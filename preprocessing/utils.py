"""
utils.py — shared helpers for the first-mention-date and symptom-extraction pipelines.

External dependencies:
    pip install pandas numpy matplotlib seaborn openai
"""

import re
import os
import json
import time
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import cpu_count, Pool
from openai import OpenAI


# ---------------------------------------------------------------------------
# LLM query helpers (OpenAI)
# ---------------------------------------------------------------------------

# Set OPENAI_API_KEY in your environment before running.
# e.g.  export OPENAI_API_KEY="sk-..."
MODEL = "gpt-4o-mini"  # change to "gpt-4o" for higher accuracy


def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def general_openai_query(system_prompt, prompt, temperature=0, json_output=False, tokens=2000):
    """Send a single prompt to the OpenAI chat API and return the response text (or parsed JSON)."""
    client = _get_client()
    kwargs = dict(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=tokens,
    )
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        if json_output and isinstance(content, str):
            content = json.loads(content)
        return content
    except Exception as e:
        print(f"OpenAI query error: {e}")
        return "Error"


def query_note(note):
    """Ask the model for the three most likely genetic conditions mentioned in a clinical note."""
    prompt = (
        f"What are the three most likely genetic condition(s) based on this note: {note}. "
        "Output format: [Disease1, Disease2, Disease3], otherwise NONE. "
        "Do not provide reasoning."
    )
    return general_openai_query(system_prompt="You are a doctor.", prompt=prompt, temperature=0.7)


def find_relevant_symptoms(entire_clinical_note):
    """Extract symptoms that could be manifestations of a genetic condition."""
    chunks = chunk_by_sentences(entire_clinical_note, max_chars=15000)
    symptoms = []
    for note_chunk in chunks:
        prompt = (
            "Extract symptoms that could be a manifestation of a genetic condition. "
            "Format: **Symptom:**, **Sentence:**. "
            f"If none, return **Symptom:** NONE. Do not explain or elaborate. Text: {note_chunk}"
        )
        result = general_openai_query(system_prompt="You are a doctor.", prompt=prompt, temperature=0.7)
        if result and "NONE" not in result:
            symptoms.append(result)
    return symptoms


def apply_openai_batch(df):
    """Run query_note over a DataFrame in parallel, adding a 'LLM_pred' column."""
    start = time.time()
    with Pool(max(1, cpu_count() - 1)) as pool:
        note_results = pool.map(query_note, df["CLINICAL_DOCUMENT_TEXT"])
    df = df.copy()
    df["LLM_pred"] = note_results
    print(f"Batch finished in {time.time() - start:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Text / note utilities
# ---------------------------------------------------------------------------

def chunk_by_sentences(text, max_chars=15000):
    """Split *text* into chunks at sentence boundaries, each ≤ max_chars."""
    try:
        sentences = re.split(r"(?<=[.!?])\s+", text)
    except Exception:
        return []

    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = [sentence], len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += len(sentence)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def estimate_tokens_simple(text):
    """Rough token estimate: average of character-based and word-based heuristics."""
    char_estimate = len(text) / 4
    word_estimate = len(text.split()) / 0.75
    return int((char_estimate + word_estimate) / 2)


# ---------------------------------------------------------------------------
# Note pre-processing
# ---------------------------------------------------------------------------

def deduplicate_notes(df):
    """De-duplicate notes by (patient, date), keeping the longest text."""
    df = df.copy()
    df["_note_len"] = df["CLINICAL_DOCUMENT_TEXT"].str.len()
    df = (
        df.sort_values("_note_len", ascending=False)
        .drop_duplicates(subset=["PATIENT_ID", "NOTE_DATE"], keep="first")
        .drop(columns=["_note_len"])
    )
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Disease synonym lookup
# ---------------------------------------------------------------------------

def read_disease_synonyms(disease, synonyms_file):
    """
    Return the synonym list for *disease* from *synonyms_file*.

    The JSON file should be a list of objects:
        [{"Disease": "cystic fibrosis", "Synonyms": ["CF", ...]}, ...]
    """
    with open(synonyms_file, "r") as fp:
        data = json.load(fp)
    for entry in data:
        if entry["Disease"].lower() == disease.lower():
            return entry["Synonyms"]
    print(f"WARNING: could not find synonyms for '{disease}'")
    sys.exit(1)


# ---------------------------------------------------------------------------
# First-mention-date I/O
# ---------------------------------------------------------------------------

def read_first_mention_dates(before_diagnosis_file):
    """
    Read a CSV with columns [patient_id, first_mention_date] (no header row expected
    after the first line, which is skipped).

    Returns a dict {patient_id: date_string}.
    """
    first_mention_dates = {}
    if not os.path.exists(before_diagnosis_file):
        return {}
    with open(before_diagnosis_file, "r") as fp:
        next(fp)  # skip header
        for line in fp:
            fields = line.strip().split(",")
            if len(fields) > 1:
                pid, date = fields[0], fields[1]
                first_mention_dates[pid] = date
    return first_mention_dates


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_first(x):
    """Return the first element of a stringified list, or 'NONE'."""
    import ast

    default = "NONE"
    if x is None or (isinstance(x, float) and np.isnan(x)) or x == "NONE":
        return default
    if isinstance(x, str):
        if x.lower().startswith("none"):
            return default
        if x.startswith("[") and x.endswith("]"):
            arr = x.strip("[]").split(",")
            return arr[0].strip() if arr else default
        try:
            parsed = ast.literal_eval(x)
            return parsed[0] if parsed else default
        except Exception:
            pass
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    print(f"Could not parse: {x!r}")
    return default


def convert_to_clean_array(x):
    """Convert a stringified list to a Python list, filtering out None/'NONE' entries."""
    if x is None or x == "NONE" or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        return [
            item.strip()
            for item in x.strip("[]").split(",")
            if item and item.strip().lower() != "none"
        ]
    return []


def cumulative_arrays(group):
    """Turn a sequence of arrays into a list of running cumulative arrays."""
    cumulative, running = [], []
    for arr in group:
        running.extend(arr)
        cumulative.append(running.copy())
    return cumulative


def patient_description(gender, birthdate, first_mention_date):
    """Return a human-readable age/gender description at the time of first mention."""
    age_days = (pd.to_datetime(first_mention_date) - pd.to_datetime(birthdate)).days
    age_years = age_days // 365
    age_months = age_days // 30
    if age_years > 2:
        return f"{age_years}-year-old {gender} patient"
    elif age_months >= 1:
        return f"{age_months}-month-old {gender} patient"
    else:
        return f"{age_days}-day-old {gender} patient"


# ---------------------------------------------------------------------------
# Debugging helpers
# ---------------------------------------------------------------------------

def find_sets(obj, path="root"):
    """Recursively find any Python sets hidden in a nested structure (useful for debugging JSON serialisation issues)."""
    if isinstance(obj, set):
        print(f"Found set at: {path}")
    elif isinstance(obj, dict):
        for key, val in obj.items():
            find_sets(val, f"{path}.{key}")
    elif isinstance(obj, list):
        for i, val in enumerate(obj):
            find_sets(val, f"{path}[{i}]")
