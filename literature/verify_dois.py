"""
Verifies DOIs for literature/papers.csv against the real CrossRef REST API
(https://api.crossref.org/works), per CLAUDE.md's rule against inventing
citations/DOIs. No API key required.

For each row currently marked TO_VERIFY (the one row with a real DOI already
is left untouched), this queries CrossRef with a bibliographic query built
from Title + first author + Year, and only accepts a candidate as a match if:
  - normalized title similarity (difflib.SequenceMatcher) > 0.85, AND
  - the candidate's publication year is within +/-1 of the recorded Year.
Journal name is compared as a soft/informational signal only (recorded in
verification_note), since conference/journal name formatting varies too much
across sources to use as a hard gate, and the task explicitly says "if
possible" for the journal check.

If no candidate clears the bar, DOI is set to UNVERIFIED_DOI and
verification_note explains why (no results / best candidate below
threshold, with its similarity score / year mismatch), rather than guessing.

Only the DOI and verification_note columns are modified. Every other column
is passed through unchanged.
"""
import csv
import re
import sys
import time
import difflib
import requests

CSV_PATH = "papers.csv"  # run from inside literature/
API_URL = "https://api.crossref.org/works"
TITLE_SIM_THRESHOLD = 0.85
YEAR_TOLERANCE = 1
REQUEST_DELAY_SEC = 1.0
HEADERS = {"User-Agent": "AdaptiveELearningResearch-DOIVerifier/1.0 (+https://api.crossref.org)"}


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def title_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def first_author_surname(authors_field: str) -> str:
    first = authors_field.split(",")[0].replace(" et al.", "").strip()
    parts = first.split()
    return parts[-1] if parts else first


def candidate_year(item: dict):
    for key in ["published-print", "published-online", "issued", "created"]:
        node = item.get(key)
        if node and node.get("date-parts") and node["date-parts"][0]:
            y = node["date-parts"][0][0]
            if isinstance(y, int):
                return y
    return None


def query_crossref(title: str, author_surname: str, year: str):
    query = f"{title} {author_surname} {year}"
    params = {"query.bibliographic": query, "rows": 5}
    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("items", [])


def best_match(title: str, year_str: str, items: list):
    try:
        year_int = int(year_str)
    except ValueError:
        year_int = None

    best = None
    for item in items:
        cand_titles = item.get("title") or []
        if not cand_titles:
            continue
        cand_title = cand_titles[0]
        sim = title_similarity(title, cand_title)
        cy = candidate_year(item)
        year_ok = (year_int is not None and cy is not None and abs(cy - year_int) <= YEAR_TOLERANCE)
        record = {
            "doi": item.get("DOI"),
            "cand_title": cand_title,
            "sim": sim,
            "cand_year": cy,
            "year_ok": year_ok,
            "container": (item.get("container-title") or [""])[0],
        }
        if best is None or sim > best["sim"]:
            best = record
    return best


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "verification_note" not in fieldnames:
        fieldnames = fieldnames + ["verification_note"]

    n_confirmed = 0
    n_unverified = 0
    unverified_details = []

    for i, row in enumerate(rows):
        if row["DOI"] != "TO_VERIFY":
            row.setdefault("verification_note", "")
            continue

        title = row["Title"]
        year = row["Year"]
        authors = row["Authors"]
        surname = first_author_surname(authors)

        print(f"[{i+1}/{len(rows)}] Querying CrossRef for: {title[:70]}... "
              f"(author={surname}, year={year})")

        try:
            items = query_crossref(title, surname, year)
        except requests.RequestException as e:
            row["DOI"] = "UNVERIFIED_DOI"
            row["verification_note"] = f"CrossRef request failed: {e}"
            n_unverified += 1
            unverified_details.append((title, row["verification_note"]))
            time.sleep(REQUEST_DELAY_SEC)
            continue

        if not items:
            row["DOI"] = "UNVERIFIED_DOI"
            row["verification_note"] = "CrossRef returned zero results for this bibliographic query."
            n_unverified += 1
            unverified_details.append((title, row["verification_note"]))
            time.sleep(REQUEST_DELAY_SEC)
            continue

        match = best_match(title, year, items)
        if match and match["sim"] > TITLE_SIM_THRESHOLD and match["year_ok"]:
            row["DOI"] = match["doi"]
            note = (f"Confirmed via CrossRef: title_sim={match['sim']:.3f}, "
                    f"candidate_year={match['cand_year']}, "
                    f"container='{match['container']}'")
            row["verification_note"] = note
            n_confirmed += 1
            print(f"    -> CONFIRMED {match['doi']} (sim={match['sim']:.3f})")
        else:
            row["DOI"] = "UNVERIFIED_DOI"
            if match:
                reason = (f"Best CrossRef candidate below confidence threshold: "
                          f"title_sim={match['sim']:.3f} (need > {TITLE_SIM_THRESHOLD}), "
                          f"candidate_year={match['cand_year']} vs recorded {year} "
                          f"(year_ok={match['year_ok']}), candidate_title='{match['cand_title']}', "
                          f"candidate_doi='{match['doi']}'")
            else:
                reason = "No candidate with a title in the CrossRef response."
            row["verification_note"] = reason
            n_unverified += 1
            unverified_details.append((title, reason))
            print(f"    -> UNVERIFIED ({reason[:100]}...)")

        time.sleep(REQUEST_DELAY_SEC)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== SUMMARY ===")
    print(f"Confirmed real DOIs via CrossRef: {n_confirmed}")
    print(f"Left as UNVERIFIED_DOI: {n_unverified}")
    if unverified_details:
        print("\nUNVERIFIED_DOI papers and reasons:")
        for title, reason in unverified_details:
            print(f" - {title}\n   reason: {reason}")


if __name__ == "__main__":
    main()
