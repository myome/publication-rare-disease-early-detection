"""
build_hpo_lookup.py

Parse the HPO OBO file (hp.obo) and produce two TSV lookup files used by
identify_symptoms.py:

    hpo_terms_and_names.txt
        HP:0002099  Asthma

    hpo_terms_and_names_with_synonyms.txt   (used by the pipeline)
        HP:0002099  Asthma  bronchial asthma|asthmatic

Usage
-----
    # 1. Download the latest HPO release (one-time step):
    python build_hpo_lookup.py --download

    # 2. Parse and build the lookup files:
    python build_hpo_lookup.py --obo hp.obo --out-dir data/

Both steps at once:
    python build_hpo_lookup.py --download --out-dir data/

The hp.obo file is published by the Human Phenotype Ontology Consortium
under a custom open licence: https://hpo.jax.org/app/license
It is freely available for academic and non-commercial use.

Source: https://github.com/obophenotype/human-phenotype-ontology
"""

import argparse
import csv
import os
import re
import urllib.request
from pathlib import Path

HPO_OBO_URL = "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/hp.obo"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_obo(dest_path):
    """Download the latest hp.obo release to *dest_path*."""
    print(f"Downloading {HPO_OBO_URL} ...")
    urllib.request.urlretrieve(HPO_OBO_URL, dest_path)
    size_mb = os.path.getsize(dest_path) / 1_000_000
    print(f"Saved {dest_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_obo(obo_path, terms_file, synonyms_file):
    """
    Parse *obo_path* and write:
      - *terms_file*    : ID <TAB> Name
      - *synonyms_file* : ID <TAB> Name/Synonym  (one row per name or EXACT synonym)

    Only [Term] blocks whose IDs start with "HP:" are written; obsolete terms
    are skipped.
    """
    terms_written = 0
    synonym_rows_written = 0

    with open(obo_path, "r", encoding="utf-8") as f_in, \
         open(terms_file, "w", newline="", encoding="utf-8") as f_terms, \
         open(synonyms_file, "w", newline="", encoding="utf-8") as f_syn:

        terms_writer = csv.writer(f_terms, delimiter="\t")
        terms_writer.writerow(["ID", "Name"])

        syn_writer = csv.writer(f_syn, delimiter="\t")
        syn_writer.writerow(["ID", "Name"])

        current_id = None
        current_name = None
        current_synonyms = []
        in_term = False
        is_obsolete = False

        def _flush():
            nonlocal terms_written, synonym_rows_written
            if not (in_term and current_id and current_id.startswith("HP:") and current_name and not is_obsolete):
                return
            terms_writer.writerow([current_id, current_name])
            terms_written += 1
            syn_writer.writerow([current_id, current_name])
            synonym_rows_written += 1
            for syn in current_synonyms:
                syn_writer.writerow([current_id, syn])
                synonym_rows_written += 1

        for line in f_in:
            line = line.strip()

            if line == "[Term]":
                _flush()
                current_id = None
                current_name = None
                current_synonyms = []
                in_term = True
                is_obsolete = False

            elif line.startswith("[") and line.endswith("]"):
                # Any other stanza type (e.g. [Typedef]) — flush and exit term mode
                _flush()
                in_term = False

            elif in_term:
                if line.startswith("id:"):
                    current_id = line.split("id:", 1)[1].strip()

                elif line.startswith("name:"):
                    current_name = line.split("name:", 1)[1].strip()

                elif line.startswith("is_obsolete:"):
                    is_obsolete = line.split("is_obsolete:", 1)[1].strip().lower() == "true"

                elif line.startswith("synonym:") and "EXACT" in line:
                    match = re.search(r'synonym:\s+"([^"]+)"', line)
                    if match:
                        current_synonyms.append(match.group(1))

        _flush()  # last term in file

    print(f"Terms written      : {terms_written:,}  → {terms_file}")
    print(f"Synonym rows written: {synonym_rows_written:,}  → {synonyms_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--obo",
        default="hp.obo",
        help="Path to hp.obo (default: hp.obo in current directory)",
    )
    ap.add_argument(
        "--out-dir",
        default="data",
        help="Directory for output TSV files (default: data/)",
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download the latest hp.obo before parsing",
    )
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if args.download:
        download_obo(args.obo)

    if not os.path.exists(args.obo):
        ap.error(
            f"{args.obo} not found. Run with --download to fetch it, "
            "or pass --obo /path/to/existing/hp.obo"
        )

    terms_file = os.path.join(args.out_dir, "hpo_terms_and_names.txt")
    synonyms_file = os.path.join(args.out_dir, "hpo_terms_and_names_with_synonyms.txt")

    print(f"Parsing {args.obo} ...")
    parse_obo(args.obo, terms_file, synonyms_file)
    print("Done.")


if __name__ == "__main__":
    main()
