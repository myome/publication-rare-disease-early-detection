"""
check_first_mention_dates.py

Given a set of per-patient clinical-note TSV files and a CSV of "default"
first-mention dates (e.g. from ICD codes), this script walks each patient's
notes in chronological order and tries to find an *earlier* first mention of
a target disease using regex matching.  If an earlier date is found in the
note text (e.g. "history of cystic fibrosis since 2015") it is recorded too.

Usage
-----
    python check_first_mention_dates.py \
        --disease "cystic fibrosis" \
        --notes-dir  /path/to/notes/cystic_fibrosis/ \
        --synonyms    data/rare_disease_synonyms.json \
        --dates-csv   data/cystic_fibrosis_first_mention.csv \
        --out-dir     output/cystic_fibrosis/

Input files
-----------
notes-dir   : directory of per-patient TSV files, one file per patient.
              Each TSV must have exactly three columns:
                  PATIENT_ID, NOTE_DATE, CLINICAL_DOCUMENT_TEXT
              Filenames must be <patient_id>.tsv  (patient_id is all digits).

synonyms    : JSON file – list of objects:
              [{"Disease": "cystic fibrosis", "Synonyms": ["CF", "mucoviscidosis", ...]}, ...]

dates-csv   : CSV with a header row, then lines:
              patient_id,first_mention_date
              The date may be any pandas-parseable string.
              Patients absent from this file get pd.Timestamp.max as the default.

out-dir     : directory where per-patient log files and the final summary CSV
              are written.
"""

import re
import os
import sys
import glob
import argparse
from pathlib import Path
from dateutil import parser as dateutil_parser

import pandas as pd

import utils


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def parse_date(token):
    """
    Try to parse *token* as a date.  Returns a datetime or None.
    Rejects tokens that look like pure numbers (avoids false positives
    on IDs, page numbers, etc.).
    """
    if not token or not isinstance(token, str):
        return None
    token = token.strip().strip(".,;:()")
    # Must contain at least one letter or slash/hyphen to be a date
    if re.match(r"^\d+$", token):
        return None
    try:
        return dateutil_parser.parse(token, default=None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Disease detection in a single note
# ---------------------------------------------------------------------------

def check_for_disease_regex(note_text, disease_and_synonyms, abbrevs):
    """
    Search *note_text* for any synonym (case-insensitive) or abbreviation
    (case-sensitive, whole-word).

    Returns
    -------
    (match_verdict, evidence, context, possible_earlier_date)
        match_verdict       – True if found
        evidence            – snippet of surrounding text
        context             – "Disease in Medical History" or None
        possible_earlier_date – date parsed from history phrasing, or None
    """
    note_text_lower = note_text.lower()
    all_matches = []

    for synonym in disease_and_synonyms:
        synonym_lower = synonym.lower()
        matches = list(re.finditer(re.escape(synonym_lower), note_text_lower))
        all_matches.extend(matches)

    for abbrev in abbrevs:
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        all_matches.extend(re.finditer(pattern, note_text))

    match_verdict = False
    clinical_history = False
    possible_earlier_date = None
    evidence = None
    history_evidence = None
    propose_testing = False

    for match in all_matches:
        match_verdict = True
        pos = match.start()
        s_length = match.end() - match.start()

        start = max(0, pos - 50)
        end = min(len(note_text), pos + s_length + 30)

        evidence = note_text[start:end]
        evidence_before = note_text[start : pos + s_length]
        evidence_after = note_text[pos : pos + s_length + 50]

        if re.search(r"\b(suspect|possibl|probabl|screen)\b", evidence, re.IGNORECASE):
            propose_testing = True
            print(f"Suspected mention: {evidence}")

        if re.search(r"\bhistory\b", evidence_before, re.IGNORECASE) and \
                not re.search(r"\bfamily history\b", evidence_before, re.IGNORECASE):
            clinical_history = True
            history_evidence = evidence

            # Try to find a date token after the match (e.g. "history of CF since 2015")
            tokens = re.findall(r"[\w/.-]+", evidence_after)
            for token in tokens:
                if len(token) < 4:   # too short to be a year
                    continue
                earlier_date_tmp = parse_date(token)
                if earlier_date_tmp is not None:
                    print(f"Candidate earlier date: {earlier_date_tmp}  (token: {token!r})")
                    if possible_earlier_date is None or \
                            pd.to_datetime(earlier_date_tmp) < pd.to_datetime(possible_earlier_date):
                        possible_earlier_date = earlier_date_tmp

    if history_evidence:
        evidence = history_evidence

    context = "Disease in Medical History" if clinical_history else None

    return match_verdict, evidence, context, possible_earlier_date


# ---------------------------------------------------------------------------
# Per-patient log
# ---------------------------------------------------------------------------

def log_date(patient_id, note_date, note_index, first_visit_date, context, alt_date, evidence, out_dir):
    """Write a one-line log file for this patient to *out_dir*."""
    first_mention_date = note_date

    if alt_date is not None and str(alt_date) != "None":
        try:
            if pd.to_datetime(alt_date) < pd.to_datetime(note_date):
                first_mention_date = alt_date
        except Exception:
            pass

    if str(first_mention_date) == "NaT":
        first_mention_date = ""

    if first_mention_date:
        try:
            first_mention_date = first_mention_date.replace(microsecond=0)
        except AttributeError:
            pass

    days_to_mention = ""
    if first_mention_date and first_visit_date:
        try:
            days_to_mention = max(0, (pd.to_datetime(first_mention_date) - pd.to_datetime(first_visit_date)).days)
        except Exception:
            pass

    log_path = os.path.join(out_dir, f"{patient_id}.log")
    with open(log_path, "w") as fp:
        fp.write(
            ",".join(
                [
                    str(patient_id),
                    str(first_mention_date),
                    str(note_index),
                    str(days_to_mention),
                    str(note_date),
                    str(context),
                    str(evidence),
                ]
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Main per-patient function
# ---------------------------------------------------------------------------

def get_earliest_mention_date(
    sample_clinical_file,
    disease_and_synonyms,
    original_first_mention_date,
    out_dir,
    abbrevs=None,
):
    """
    Scan all clinical notes for *patient_id* and return
    [patient_id, earliest_date] where earliest_date is the earliest
    date at which the disease appears in any note (or
    original_first_mention_date if it is not found earlier).

    Parameters
    ----------
    sample_clinical_file       : path to the patient's TSV note file
    disease_and_synonyms       : list of strings (synonyms for the disease)
    original_first_mention_date: pd.Timestamp – the default date to fall back to
    out_dir                    : directory for per-patient log files
    abbrevs                    : list of case-sensitive abbreviations (e.g. ["CF"])
    """
    if abbrevs is None:
        abbrevs = []

    basename = os.path.basename(sample_clinical_file)
    patient_id = os.path.splitext(basename)[0]

    # Normalise the cutoff date
    original_first_mention_date = pd.to_datetime(original_first_mention_date)
    try:
        original_first_mention_date = original_first_mention_date.floor("s")
    except Exception:
        pass

    try:
        df = pd.read_csv(sample_clinical_file, sep="\t", index_col=0)
    except Exception as e:
        print(f"Could not read {sample_clinical_file}: {e} – skipping")
        return None

    if df.empty:
        return None

    df["NOTE_DATE"] = pd.to_datetime(df["NOTE_DATE"])
    date_col = "NOTE_DATE"
    df = df.sort_values(date_col)  # chronological order

    num_notes = len(df)
    first_visit_date = None

    for idx, (_, row) in enumerate(df.iterrows()):
        note_date = pd.to_datetime(row[date_col])

        if first_visit_date is None:
            first_visit_date = note_date

        # Once we reach the default first-mention date, stop searching –
        # any note here or later cannot give us an *earlier* date.
        if note_date >= original_first_mention_date:
            log_date(patient_id, original_first_mention_date, idx,
                     first_visit_date, None, None, "reached cutoff", out_dir)
            return [patient_id, original_first_mention_date]

        note_text = row.get("CLINICAL_DOCUMENT_TEXT", "")
        if not isinstance(note_text, str):
            continue

        found, evidence, context, alt_mention_date = check_for_disease_regex(
            note_text, disease_and_synonyms, abbrevs
        )

        if found:
            # Discard alt_mention_date if it is later than the note itself
            # (common false positive: parsed token defaults to current year)
            if alt_mention_date is not None:
                try:
                    if pd.to_datetime(alt_mention_date) > pd.to_datetime(note_date):
                        alt_mention_date = None
                except Exception:
                    alt_mention_date = None

            log_date(patient_id, note_date, idx, first_visit_date,
                     context, alt_mention_date, evidence, out_dir)
            return [patient_id, note_date]

    # Disease not found in any note before the cutoff – keep the original date
    log_date(patient_id, original_first_mention_date, num_notes,
             first_visit_date, None, None, "not found, keep original", out_dir)
    return [patient_id, original_first_mention_date]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease",    required=True,  help="Disease name exactly as it appears in the synonyms JSON")
    ap.add_argument("--notes-dir",  required=True,  help="Directory containing per-patient TSV note files (<patient_id>.tsv)")
    ap.add_argument("--synonyms",   required=True,  help="Path to rare_disease_synonyms.json")
    ap.add_argument("--dates-csv",  required=True,  help="CSV of default first-mention dates (patient_id,first_mention_date)")
    ap.add_argument("--out-dir",    required=True,  help="Output directory for log files and summary CSV")
    args = ap.parse_args()

    disease = args.disease
    notes_dir = args.notes_dir
    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    disease_and_synonyms = utils.read_disease_synonyms(disease, args.synonyms)
    print(f"Synonyms for '{disease}': {disease_and_synonyms}")

    first_mention_dates = utils.read_first_mention_dates(args.dates_csv)

    # Disease-specific abbreviations
    abbrev_map = {
        "cystic fibrosis": ["CF"],
        "neurofibromatosis_type_1": ["NF1"],
        "tuberous_sclerosis_complex": ["TSC"],
    }
    abbrevs = abbrev_map.get(disease, [])

    results = []
    tsv_files = sorted(glob.glob(os.path.join(notes_dir, "*.tsv")))

    for tsv_file in tsv_files:
        basename = os.path.basename(tsv_file)
        patient_id = os.path.splitext(basename)[0]

        if not patient_id.isdigit():
            continue  # skip non-patient files

        if patient_id in first_mention_dates:
            default_date = pd.to_datetime(first_mention_dates[patient_id])
        else:
            default_date = pd.Timestamp.max  # no prior date known

        result = get_earliest_mention_date(
            tsv_file,
            disease_and_synonyms,
            default_date,
            out_dir,
            abbrevs=abbrevs,
        )
        if result:
            results.append(result)

    # Write summary
    if results:
        out_csv = os.path.join(out_dir, f"{disease.replace(' ', '_')}_first_mention_updated.csv")
        summary = pd.DataFrame(results, columns=["patient_id", "first_mention_date"])
        summary.to_csv(out_csv, index=False)
        print(f"\nSummary written to {out_csv}")
        print(summary.head())
    else:
        print("No results produced.")


if __name__ == "__main__":
    main()
