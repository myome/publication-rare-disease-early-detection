from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from xgboost import XGBClassifier, plot_importance
import shap
import polars as pl
import polars.selectors as cs


def make_train_test(disease, strict=True):
    hpo = pl.read_json(f'/data/MyOmeWorkingProject/shared/rare_disease/hpo_files/{disease}_oh.json').with_columns(status = pl.lit(1))

    data = pl.concat([controls, hpo])

    data = data.with_columns(
        pl.col('entry_date').str.to_date(),
        pl.col('person_id').cast(pl.Int64)
    )

    scores = pl.read_csv(f'/data/MyOmeWorkingProject/shared/rare_disease/phers_files/with_scores/{disease}.csv')
    scores = scores.with_columns(
        pl.col('entry_date').str.to_date(),
        pl.col('first_diag_date').str.to_date()
    ).with_columns(
        days_from_diag = (pl.col('entry_date') - pl.col('first_diag_date')).dt.total_days()
    ).select(['person_id', 'entry_date', 'score', 'days_from_diag'])


    control_scores = control_scores_all.filter(
        disease = disease).drop('disease')

    control_scores = control_scores.with_columns(
        pl.col('entry_date').str.to_date(),
    ).with_columns(
        days_from_diag = None
    ).cast(
        {"days_from_diag": pl.Int64}
    ).select(pl.col(['person_id', 'entry_date', 'score', 'days_from_diag'])
    ).sort(
        ['person_id', 'entry_date']
    )

    ascores = pl.concat([control_scores, scores])

    ascores = ascores.group_by(['person_id', 'entry_date'], maintain_order=True).agg(pl.col('score').max(), pl.col('days_from_diag').first())

    m = data.join(ascores.select(['person_id', 'entry_date', 'days_from_diag', 'score']), on=['person_id', 'entry_date'])

    mlb = MultiLabelBinarizer()
    x = pl.DataFrame(mlb.fit_transform(m['HPO_cum']))
    x.columns = mlb.classes_
    m = pl.concat([m.drop('HPO_cum'), x], how='horizontal')

    fm = pl.read_csv(f'./notes/{disease}_manual_first_mention_Oct22.csv')

    test_ids = list(fm['NFER_PID']) + list(control_scores['person_id'].unique().sample(100, seed=24))

    if not strict:
        ntest = round(scores['person_id'].n_unique() * .2)
        case_ids = list(scores['person_id'].unique().sample(ntest, seed=24))
        test_ids += case_ids

    case_count = m.filter(pl.col('status') == 1)['person_id'].n_unique()
    control_count = m.filter(pl.col('status') == 0)['person_id'].n_unique()

    m_test = m.filter(pl.col('person_id').is_in(test_ids))

    m_train = m.filter(~(pl.col('person_id').is_in(test_ids)))

    m_train = m_train.sort(['person_id', 'entry_date']).group_by("person_id", maintain_order=True).last()
    return (m_train, m_test, case_count, control_count)


strict = False
tq = .99
for disease in disease_list:

    m_train, m_test, case_count, control_count = make_train_test(disease, strict=strict)
    X_train = m_train.drop(['person_id', 'entry_date', 'days_from_diag', 'status'])
    X_test = m_test.drop(['person_id', 'entry_date', 'days_from_diag', 'status'])
    y_train = m_train['status']

    patient_count = case_count + control_count

    model = XGBClassifier(n_estimators=100, learning_rate=0.1, importance_type="gain", random_state=42)
    model.fit(X_train, y_train)

    importance_scores = model.feature_importances_
    #plot_importance(model, importance_type='gain', max_num_features=30)

    imp = pl.DataFrame({'imp': importance_scores, 'names': X_train.columns})

    score_import = imp.filter(pl.col('names') == 'score')['imp'][0]

    test = m_test.select(['person_id', 'entry_date', 'days_from_diag', 'status']).with_columns(
        pred = model.predict_proba(X_test)[:, 1]
    )

    test_control = test.filter(pl.col('status') == 0)

    thresh = test_control['pred'].quantile(tq)

    test_case = test.filter(pl.col('status') == 1)

    fm = pl.read_csv(f'./notes/{disease}_manual_first_mention_Oct22.csv')

    test_case = test_case.join(fm.select(['NFER_PID', 'first_mention_date']), left_on='person_id', right_on='NFER_PID')

    test_case = test_case.with_columns(
        pl.col('first_mention_date').str.to_datetime().dt.date()
    ).with_columns(
        (pl.col('entry_date') - pl.col('first_mention_date')).dt.total_days().alias('days_from_mention')
    )

    earlyd = test_case.filter(
        (pl.col('days_from_diag') < 0) & (pl.col('pred') >= thresh)
    )

    earlym = test_case.filter(
        (pl.col('days_from_diag') < 0) & (pl.col('days_from_mention') < 0) & (pl.col('pred') >= thresh)
    )

    n = test_case['person_id'].n_unique()
    pd = earlyd.with_columns(Disease = pl.lit(disease)).select(pl.col(['Disease', 'person_id', 'days_from_diag'])).group_by('person_id').agg(
            pl.col('Disease').first(), pl.min('days_from_diag')).group_by('Disease').agg(
            pl.median('days_from_diag').alias('Median'),
            pl.mean('days_from_diag').alias('Mean'),
            pl.std('days_from_diag').alias('Stdev'),
        Fraction_Early = (earlyd['person_id'].n_unique() / n),
        Period = pl.lit('Pre_diag')
    )

    pm = earlym.with_columns(Disease = pl.lit(disease)).select(pl.col(['Disease', 'person_id', 'days_from_mention'])).group_by('person_id').agg(
            pl.col('Disease').first(), pl.min('days_from_mention')).group_by('Disease').agg(
            pl.median('days_from_mention').alias('Median'),
            pl.mean('days_from_mention').alias('Mean'),
            pl.std('days_from_mention').alias('Stdev'),
        Fraction_Early = (earlym['person_id'].n_unique() / n),
        Period = pl.lit('Pre_Mention')
    )

    ef = pl.concat([pd, pm])

    ef = ef.with_columns(
        Score_Imp = score_import,
        Patient_Count = patient_count,
        Case_Count = case_count,
        Control_Count = control_count,
        Strict = strict,
        Spec = pl.lit(tq)
    )

    res = pl.concat([res, ef])

    imp = imp.with_columns(
        Disease = pl.lit(disease),
        Patient_Count = patient_count,
        Strict = strict,
        Spec = pl.lit(tq)
    )

    res_imp = pl.concat([res_imp, imp])
