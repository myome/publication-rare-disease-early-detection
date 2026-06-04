"""
identify_symptoms.py

For each patient's clinical-note TSV file, extract HPO-coded symptoms using
the OpenAI API (GPT-4o-mini by default), then validate the returned HPO IDs
against a local HPO synonym dictionary and verify that extracted text snippets
actually appear in the source note (hallucination check).

Results are written as JSON — one file per patient — and can be aggregated
for downstream analysis.

Usage
-----
    python identify_symptoms.py \
        --notes-dir   data/notes/cystic_fibrosis/ \
        --hpo-file    data/hpo_terms_and_names_with_synonyms.txt \
        --controls    data/controls.csv \
        --out-dir     output/symptoms/cystic_fibrosis/ \
        [--max-cases  500] \
        [--workers    7]

Input files
-----------
notes-dir   : directory of per-patient TSV files (PATIENT_ID, NOTE_DATE,
              CLINICAL_DOCUMENT_TEXT), same format as check_first_mention_dates.py.
              Case files: <patient_id>.tsv
              Control files: controls/<control_id>.tsv

hpo-file    : tab-separated file mapping HPO IDs to term names and synonyms.
              Format per line:
                  HP:0002099<TAB>Asthma<TAB>bronchial asthma|asthmatic
              Download from https://hpo.jax.org/data/ontology

controls    : CSV with columns Patient_ID and Control_ID (one case may have
              multiple matched controls).

out-dir     : directory where per-patient .hpo.json files are written.
              Already-completed files are skipped automatically.

Environment
-----------
    export OPENAI_API_KEY="sk-..."
"""

import copy
import csv
import glob
import json
import os
import re
import sys
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from openai import OpenAI

from utils import chunk_by_sentences

csv.field_size_limit(sys.maxsize)

# ---------------------------------------------------------------------------
# OpenAI client — key read from environment
# ---------------------------------------------------------------------------

def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Run: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


MODEL = "gpt-4o-mini"   # change to "gpt-4o" for higher accuracy


# ---------------------------------------------------------------------------
# HPO dictionary
# ---------------------------------------------------------------------------

def load_hpo_dict(hpo_file):
    """
    Load the HPO synonym file into a dict mapping every term name / synonym
    (lower-cased) to its canonical HPO ID.

    Expected file format (tab-separated):
        HP:0002099<TAB>Asthma<TAB>bronchial asthma|asthmatic

    The third column (synonyms) is optional.
    Download the latest release from https://hpo.jax.org/data/ontology
    """
    symptom2hpo = {}
    with open(hpo_file, "r", encoding="utf-8") as fp:
        for line in fp:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            hpo_id, term_name = parts[0].strip(), parts[1].strip()
            symptom2hpo[term_name.lower()] = hpo_id
            if len(parts) >= 3:
                for synonym in parts[2].split("|"):
                    s = synonym.strip()
                    if s:
                        symptom2hpo[s.lower()] = hpo_id
    return symptom2hpo


# ---------------------------------------------------------------------------
# OpenAI symptom extraction
# ---------------------------------------------------------------------------

_EXAMPLE_OUTPUT = """{
  "symptoms": [
    {
      "text": "asthma attacks",
      "hpo_term": "Asthma",
      "hpo_id": "HP:0002099"
    },
    {
      "text": "renal failure",
      "hpo_term": "Renal insufficiency",
      "hpo_id": "HP:0000083"
    }
  ]
}"""


def check_for_symptoms_openai(clinical_note):
    """
    Call the OpenAI API to extract HPO-coded symptoms from *clinical_note*.

    Returns a parsed dict like {"symptoms": [{text, hpo_term, hpo_id}, ...]},
    or None on failure.
    """
    prompt = (
        "Identify all symptoms and their HPO terms present in the clinical note below. "
        f"Output ONLY valid JSON in this exact format:\n{_EXAMPLE_OUTPUT}\n\n"
        f"Note:\n{clinical_note}"
    )

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Named Entity Recognition system in the medical domain. "
                               "Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if isinstance(content, str):
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return _extract_symptoms_from_partial_json(content)
        return content

    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


def _extract_symptoms_from_partial_json(incomplete_json):
    """
    Best-effort recovery when the model returns malformed JSON.
    Extracts any recognisable symptom objects via regex.
    """
    pattern = (
        r'"text"\s*:\s*"([^"]+)"'
        r'.*?"hpo_term"\s*:\s*"([^"]+)"'
        r'.*?"hpo_id"\s*:\s*"([^"]+)"'
    )
    symptoms = [
        {"text": m[0], "hpo_term": m[1], "hpo_id": m[2]}
        for m in re.findall(pattern, incomplete_json, re.DOTALL)
    ]
    return {"symptoms": symptoms} if symptoms else None


# ---------------------------------------------------------------------------
# Chunk-level batching and combination
# ---------------------------------------------------------------------------

def _apply_openai_batch(note_chunks):
    """Run symptom extraction on each chunk sequentially and return all results."""
    results = []
    for chunk in note_chunks:
        result = check_for_symptoms_openai(chunk)
        if result is not None:
            results.append(result)
    return results


def _combine_chunk_results(chunk_results):
    """
    Merge symptom lists from multiple chunks, de-duplicating by HPO ID and
    concatenating all supporting text snippets.
    """
    chunk_results = [r for r in chunk_results if r]
    if not chunk_results:
        return {}
    if len(chunk_results) == 1:
        return chunk_results[0]

    combined = {}
    for json_data in chunk_results:
        for symptom in json_data.get("symptoms") or []:
            hpo_id = symptom.get("hpo_id")
            if not hpo_id:
                continue
            if hpo_id in combined:
                existing_text = combined[hpo_id]["text"]
                if isinstance(existing_text, list):
                    existing_text.append(symptom["text"])
                else:
                    combined[hpo_id]["text"] = [existing_text, symptom["text"]]
            else:
                combined[hpo_id] = symptom.copy()

    return {"symptoms": list(combined.values())}


# ---------------------------------------------------------------------------
# HPO validation and hallucination check
# ---------------------------------------------------------------------------

def check_hpo_terms_json(symptom_json, note_text, symptom2hpo):
    """
    For each extracted symptom:
      1. Verify the text snippet actually appears in *note_text*
         (flags hallucinations).
      2. Look up the HPO term name in *symptom2hpo*; discard entries whose
         term name is not in the dictionary, and correct the HPO ID if it
         differs from the canonical one.

    Returns a cleaned {"symptoms": [...]} dict.
    """
    if not symptom_json.get("symptoms"):
        return {"symptoms": []}

    cleaned = []
    for symptom_dict in symptom_json["symptoms"]:
        new_symptom = copy.deepcopy(symptom_dict)

        # --- hallucination check ---
        context_texts = symptom_dict.get("text", "")
        if isinstance(context_texts, str):
            context_texts = [context_texts]

        verified_snippets = []
        for ctx in context_texts:
            if not ctx:
                continue
            if ctx.lower() not in note_text.lower():
                print(f"  [hallucination] '{ctx}' not found in note — skipping")
                continue
            # Grab surrounding context (±30 chars) for each occurrence
            for match in re.finditer(re.escape(ctx.lower()), note_text.lower()):
                start = max(0, match.start() - 30)
                end = min(len(note_text), match.end() + 30)
                verified_snippets.append(note_text[start:end])

        if not verified_snippets:
            continue  # all snippets were hallucinated

        new_symptom["text"] = ", ".join(verified_snippets)

        # --- HPO ID validation ---
        hpo_term = symptom_dict.get("hpo_term", "")
        canonical_id = symptom2hpo.get(hpo_term.lower())
        if canonical_id is None:
            print(f"  [unknown HPO term] '{hpo_term}' not in dictionary — skipping")
            continue
        new_symptom["hpo_id"] = canonical_id  # correct if model was wrong

        cleaned.append(new_symptom)

    return {"symptoms": cleaned}


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------

def get_hpo_terms_from_clinical_notes(clinical_notes_file, symptom2hpo, outfile):
    """
    Read one patient's note TSV, extract HPO symptoms from every note
    within the 3-year window before the last note date, and write results
    to *outfile* as JSON.
    """
    try:
        df = pd.read_csv(clinical_notes_file, sep="\t")
    except Exception as e:
        print(f"Could not read {clinical_notes_file}: {e}")
        return

    if df.empty:
        return

    df["NOTE_DATE"] = pd.to_datetime(df["NOTE_DATE"])
    df = df.sort_values("NOTE_DATE")

    # Restrict to the 3-year window before the last (most recent) note
    last_note_date = df["NOTE_DATE"].iloc[-1]
    three_years_ago = last_note_date - pd.DateOffset(years=3)
    df = df[df["NOTE_DATE"] >= three_years_ago]
    df = df.drop_duplicates(subset=["CLINICAL_DOCUMENT_TEXT", "NOTE_DATE"])

    num_notes = len(df)
    json_list = []

    for idx, (_, row) in enumerate(df.iterrows()):
        patient_id = row["PATIENT_ID"]
        note_text = row["CLINICAL_DOCUMENT_TEXT"]
        note_date = row["NOTE_DATE"]

        if not isinstance(note_text, str) or not note_text.strip():
            continue

        chunks = chunk_by_sentences(note_text)
        chunk_results = _apply_openai_batch(chunks)
        symptom_json = _combine_chunk_results(chunk_results)

        if symptom_json.get("symptoms"):
            symptom_json = check_hpo_terms_json(symptom_json, note_text, symptom2hpo)

        symptom_json["PATIENT_ID"] = str(patient_id)
        symptom_json["NOTE_DATE"] = note_date.strftime("%Y-%m-%d %H:%M:%S")

        json_list.append(symptom_json)
        print(f"  {patient_id}  note {idx + 1}/{num_notes}")

    with open(outfile, "w", encoding="utf-8") as fp:
        json.dump(json_list, fp, indent=2)

    print(f"Wrote {outfile}")

def build_args_list(clinical_notes_files, symptom2hpo, out_dir):
    """
    Return a list of (clinical_notes_file, symptom2hpo, outfile) tuples,
    skipping files whose output already exists.
    """
    args_list = []
    for tsv_path in clinical_notes_files:
        patient_id = os.path.splitext(os.path.basename(tsv_path))[0]
        outfile = os.path.join(out_dir, patient_id + ".hpo.json")
        if os.path.exists(outfile):
            print(f"  Already done: {outfile}")
        else:
            args_list.append((tsv_path, symptom2hpo, outfile))
    return args_list


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--notes-dir",  required=True, help="Directory of per-patient TSV note files")
    ap.add_argument("--hpo-file",   required=True, help="HPO synonym TSV (HP_ID<TAB>term<TAB>synonyms)")
    ap.add_argument("--out-dir",    required=True, help="Output directory for .hpo.json files")   
    ap.add_argument("--workers",    type=int, default=max(1, cpu_count() - 1),
                    help="Parallel worker processes (default: CPU count − 1)")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading HPO dictionary...")
    symptom2hpo = load_hpo_dict(args.hpo_file)
    print(f"  {len(symptom2hpo)} HPO term/synonym entries loaded")

    all_tsv = sorted(glob.glob(os.path.join(args.notes_dir, "*.tsv")), key=os.path.getsize)
    clinical_notes_files = [f for f in all_tsv if os.path.splitext(os.path.basename(f))[0].isdigit()]
    print(f"  {len(clinical_notes_files)} TSV files found in {args.notes_dir}")
 
    args_list = build_args_list(clinical_notes_files, symptom2hpo, args.out_dir)
    print(f"\n{len(args_list)} files to process with {args.workers} workers\n")


    if not args_list:
        print("Nothing to do.")
        return

    with Pool(args.workers) as pool:
        pool.starmap(get_hpo_terms_from_clinical_notes, args_list)


if __name__ == "__main__":
    main()
