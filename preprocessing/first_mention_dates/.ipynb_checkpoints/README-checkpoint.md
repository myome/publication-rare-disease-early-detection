# Rare Disease NLP Pipeline



 **`check_first_mention_dates.py`** — find the *earliest* date a rare
   disease appears in clinical notes (potentially before the ICD-code date).
 **`identify_symptoms.py`** — extract HPO-coded symptoms from clinical notes
   using the OpenAI API, with hallucination checking and HPO ID validation.

---

## Setup

```bash
pip install pandas numpy openai python-dateutil
export OPENAI_API_KEY="sk-..."   # required for identify_symptoms.py
                                  # and the LLM helpers in utils.py
```

---

## File layout

```
.
├── check_first_mention_dates.py
├── identify_symptoms.py
├── utils.py
├── data/
│   ├── rare_disease_synonyms.json
│   ├── hpo_terms_and_names_with_synonyms.txt   ← see HPO section below
│   ├── cystic_fibrosis_first_mention.csv
│   ├── controls.csv
│   └── notes/
│       └── cystic_fibrosis/
│           ├── 1001.tsv          ← case patient notes
│           ├── 1002.tsv
│           ├── 1003.tsv
│           └── controls/
│               ├── 2001.tsv      ← matched control notes
│               └── 2002.tsv
└── output/                       ← created automatically
```

---

## Input formats

### Note TSV files

Tab-separated, one row per note. Three required columns:

| Column | Description |
|---|---|
| `PATIENT_ID` | Numeric patient identifier |
| `NOTE_DATE` | Date of the note (any pandas-parseable format) |
| `CLINICAL_DOCUMENT_TEXT` | Full text of the clinical note |

### Default dates CSV (`data/<disease>_first_mention.csv`)

```
patient_id,first_mention_date
1001,2021-03-15
1002,2019-07-22
```

### Disease synonyms JSON (`data/rare_disease_synonyms.json`)

```json
[
  {
    "Disease": "cystic fibrosis",
    "Synonyms": ["cystic fibrosis", "mucoviscidosis", "CF lung disease"]
  }
]
```

### Controls CSV (`data/controls.csv`)

```
Patient_ID,Control_ID
1001,2001
1001,2002
```

### HPO synonym file (`data/hpo_terms_and_names_with_synonyms.txt`)

Tab-separated, one term per line:

```
HP:0002099<TAB>Asthma<TAB>bronchial asthma|asthmatic
HP:0000083<TAB>Renal insufficiency<TAB>renal failure|kidney failure
```

A small sample file is included. For real use, download the full HPO release
from **https://hpo.jax.org/data/ontology** and convert to this format.

---

## Setup — `build_hpo_lookup.py` *(one-time setup)*

Download: HPO OBO from https://hpo.jax.org/data/ontology

Parses the public HPO OBO file and generates the lookup file required by
`identify_symptoms.py`. Run this once before your first symptom-extraction
run, or whenever you want to update to a newer HPO release.

```bash
# Download latest hp.obo and build lookup files in data/
python build_hpo_lookup.py --download --out-dir data/

# Or if you already have hp.obo locally:
python build_hpo_lookup.py --obo /path/to/hp.obo --out-dir data/

## `check_first_mention_dates.py`

Walks each patient's notes in chronological order to find a disease mention
before the default ICD-code date. If a history phrase is found
("history of cystic fibrosis since 2017"), it also tries to extract the
referenced date.

```bash
python check_first_mention_dates.py \
    --disease    "cystic fibrosis" \
    --notes-dir  data/notes/cystic_fibrosis/ \
    --synonyms   data/rare_disease_synonyms.json \
    --dates-csv  data/cystic_fibrosis_first_mention.csv \
    --out-dir    output/first_mention/cystic_fibrosis/
```

**Output — per-patient log** (`output/.../<patient_id>.log`):
```
patient_id, first_mention_date, note_index, days_from_first_visit,
note_date, context, evidence_snippet
```

**Output — summary CSV** (`output/.../<disease>_first_mention_updated.csv`):
```
patient_id,first_mention_date
1001,2019-06-10
```

### Expected output for the sample data

| patient_id | original date | updated date | reason |
|---|---|---|---|
| 1001 | 2021-03-15 | 2019-06-10 | "history of cystic fibrosis" in a 2019 note |
| 1002 | 2019-07-22 | 2018-11-03 | "possible CF" found in 2018 note |
| 1003 | 2022-01-10 | 2021-01-08 | synonym "mucoviscidosis" found in 2021 note |

---

## Script 2 — `identify_symptoms.py`

Sends each clinical note (in chunks) to the OpenAI API and asks it to
identify symptoms with their HPO codes. Results are validated against a
local HPO dictionary and hallucinations are filtered out.

```bash
python identify_symptoms.py \
    --notes-dir  data/notes/cystic_fibrosis/ \
    --hpo-file   data/hpo_terms_and_names_with_synonyms.txt \   
    --out-dir    output/symptoms/cystic_fibrosis/ \
    --workers    7
```

Output is one `.hpo.json` file per patient next to the input TSV, e.g.:

```json
[
  {
    "symptoms": [
      {
        "text": "...surrounding context from note...",
        "hpo_term": "Renal insufficiency",
        "hpo_id": "HP:0000083"
      }
    ],
    "PATIENT_ID": "1001",
    "NOTE_DATE": "2019-06-10 00:00:00"
  }
]
```

Already-completed files are skipped automatically, so re-runs are safe.

---

## `utils.py` — shared helpers

| Function | Purpose |
|---|---|
| `chunk_by_sentences(text)` | Split a note into ≤15 000-char chunks at sentence boundaries |
| `general_openai_query(...)` | Single OpenAI chat call with optional JSON mode |
| `query_note(note)` | Ask the model for likely genetic conditions in a note |
| `find_relevant_symptoms(note)` | Extract symptom descriptions (free text) |
| `deduplicate_notes(df)` | Keep longest note per (patient, date) |
| `read_disease_synonyms(disease, file)` | Load synonym list from JSON |
| `read_first_mention_dates(file)` | Load default dates CSV → dict |

---


```

**Outputs** (written to `--out-dir`):

| File | Contents |
|---|---|
| `hpo_terms_and_names.txt` | `ID <TAB> Name` — one canonical name per term |
| `hpo_terms_and_names_with_synonyms.txt` | `ID <TAB> Name/Synonym` — one row per EXACT synonym; this is the file passed to `--hpo-file` |

The hp.obo file is published by the Human Phenotype Ontology Consortium
at **https://hpo.jax.org** under an open licence free for academic use.
