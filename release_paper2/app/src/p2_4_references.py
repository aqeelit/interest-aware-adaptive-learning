"""
Paper 2, Step 4 (2026-10-14): verify the references listed in Appendix B of
manuscript/paper2_draft_2026-10-14.md via the CrossRef REST API (by DOI) and the arXiv API (by id).

The entries below are transcribed from Appendix B (claimed authors / year / venue / identifier exactly as written
there; nothing added). Rules:
  VERIFIED   every identifier of the entry resolves AND the claimed first-author surname, every other claimed
             author surname, the claimed year, and the claimed venue keyword (if any) match the record.
  MISMATCH   the identifier resolves but a claimed field differs, or the draft gives too little to confirm the
             intended source (listed for the owner; never corrected silently).
  NOT_FOUND  no DOI / arXiv id in Appendix B, or the identifier does not resolve.
No DOI is invented; entries without an identifier are not searched.
Outputs: p2_references_verified.csv, p2_references.bib (VERIFIED only), p2_references_raw.json.
"""
import json
import os
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd

from utils import RESULTS_STATS as _RS

RESULTS_STATS = os.environ.get("P2_REF_OUT", _RS)  # test runs write to /tmp

UA = "p2-reference-check/1.0 (research reproducibility script)"
ENTRIES = [
    # key, claimed authors (surnames, as in Appendix B; "et al." -> first only), year, venue keyword, doi, arxiv, level
    ("lim2026", ["Lim"], 2026, "IJACSA", None, None, "FULL TEXT"),
    ("le2026", ["Le", "Abel", "Laforge"], 2026, None, None, "2605.25794", "FULL TEXT"),
    ("kuzilek2017", ["Kuzilek"], 2017, "Scientific Data", "10.1038/sdata.2017.171", None, "ABSTRACT/RECORD"),
    ("gasevic2016", ["Gašević"], 2016, None, "10.1016/j.iheduc.2015.10.002", None, "ABSTRACT/RECORD"),
    ("conijn2017", ["Conijn"], 2017, None, "10.1109/TLT.2016.2616312", None, "ABSTRACT/RECORD"),
    ("kleter2024", ["Kleter", "Matzat", "Conijn"], 2024, None, "10.1109/TLT.2024.3443079", None, "ABSTRACT/RECORD"),
    ("cui2020", ["Cui", "Chen", "Shiri"], 2020, None, "10.1108/ILS-05-2019-0041", None, "ABSTRACT/RECORD"),
    ("liu2015", ["Liu", "Froissard", "Richards", "Atif"], 2015, "AMCIS", None, None, "ABSTRACT/RECORD"),
    ("marras2021", ["Marras", "Vignoud", "Käser"], 2021, "EDM", None, None, "ABSTRACT/RECORD"),
    ("schwerter2026", ["Schwerter"], 2026, None, "10.1016/j.compedu.2026.105670", "2604.22812", "ABSTRACT/RECORD"),
    ("riazy2019", ["Riazy", "Simbeck"], 2019, None, "10.18420/delfi2019_305", None, "ABSTRACT/RECORD"),
    ("jovanovic_unk", ["Jovanović"], None, None, None, None, "ABSTRACT/RECORD (venue and year not verified)"),
    ("waheed2020", ["Waheed"], 2020, None, "10.1016/j.chb.2019.106189", None, "CITED ONLY"),
    ("alzawqari2022", ["Al-Zawqari"], 2022, None, "10.1016/j.caeai.2022.100103", None, "CITED ONLY"),
    ("kapoor2023", ["Kapoor", "Narayanan"], 2023, None, "10.1016/j.patter.2023.100804", None, "CITED ONLY"),
    ("esbenshade2024", ["Esbenshade", "Vitale", "Baker"], 2024, "EDM", None, None, "CITED ONLY"),
    ("karimihaghighi2021", ["Karimi-Haghighi"], 2021, "LAK", None, None, "CITED ONLY"),
    ("arxiv181106369", [], None, None, None, "1811.06369", "INCOMPLETE"),
    ("abroca_power_2025", [], 2025, "ABROCA", None, "2501.04683", "INCOMPLETE"),
    ("she_meta_47_12", [], None, "Studies in Higher Education", None, None, "INCOMPLETE"),
]


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return 404, ""
            time.sleep(3)
        except Exception:  # noqa: BLE001
            time.sleep(3)
    return None, ""


def crossref(doi):
    st, body = get("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="/"))
    if st != 200:
        return None, st
    m = json.loads(body)["message"]
    year = None
    for k in ["published-print", "published-online", "issued"]:
        if m.get(k, {}).get("date-parts") and m[k]["date-parts"][0][0]:
            year = m[k]["date-parts"][0][0]
            break
    years = sorted({m[k]["date-parts"][0][0] for k in ["published-print", "published-online", "issued"]
                    if m.get(k, {}).get("date-parts") and m[k]["date-parts"][0][0]})
    return {"source": "crossref", "title": (m.get("title") or [""])[0],
            "authors": [a.get("family", a.get("name", "")) for a in m.get("author", [])],
            "authors_full": [f"{a.get('family', a.get('name', ''))}, {a.get('given', '')}".strip(", ")
                             for a in m.get("author", [])],
            "year": year, "all_years": years, "venue": (m.get("container-title") or [""])[0],
            "volume": m.get("volume", ""), "issue": m.get("issue", ""), "pages": m.get("page", ""),
            "doi": m.get("DOI", doi), "type": m.get("type", "")}, st


def arxiv(aid):
    st, body = get("https://export.arxiv.org/api/query?id_list=" + aid)
    if st != 200:
        return None, st
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(body)
    e = root.find("a:entry", ns)
    if e is None or e.find("a:title", ns) is None or (e.find("a:id", ns).text or "").endswith("api/errors"):
        return None, 404
    names = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
    return {"source": "arxiv", "title": re.sub(r"\s+", " ", e.find("a:title", ns).text).strip(),
            "authors": [n.split()[-1] for n in names], "authors_full": names,
            "year": int(e.find("a:published", ns).text[:4]), "all_years": [int(e.find("a:published", ns).text[:4])],
            "venue": "arXiv", "volume": "", "issue": "", "pages": "", "doi": "", "arxiv": aid, "type": "preprint"}, st


def compare(entry, rec):
    key, authors, year, venue, *_ = entry
    issues = []
    rec_auth = [norm(a) for a in rec["authors"]]
    if authors:
        if not rec_auth or norm(authors[0]) not in rec_auth[0] and rec_auth[0] not in norm(authors[0]):
            issues.append(f"first author: draft '{authors[0]}' vs record '{rec['authors'][0] if rec['authors'] else ''}'")
        for a in authors[1:]:
            if not any(norm(a) in r or r in norm(a) for r in rec_auth if r):
                issues.append(f"author '{a}' not in record")
    else:
        issues.append("draft gives no author/title; intended source cannot be confirmed")
    if year is not None and year not in rec["all_years"]:
        issues.append(f"year: draft {year} vs record {rec['all_years']}")
    if venue and rec["source"] == "crossref" and venue.lower() not in rec["venue"].lower():
        issues.append(f"venue: draft '{venue}' vs record '{rec['venue']}'")
    if venue and rec["source"] == "arxiv" and venue.lower() not in rec["title"].lower():
        issues.append(f"draft keyword '{venue}' not in arXiv title")
    return issues


def bib(key, rec_list):
    r = rec_list[0]
    auth = " and ".join(r["authors_full"])
    if r["source"] == "crossref":
        f = {"author": auth, "title": r["title"], "journal": r["venue"], "year": r["year"], "volume": r["volume"],
             "number": r["issue"], "pages": r["pages"], "doi": r["doi"]}
        typ = "article" if r["type"] == "journal-article" else "inproceedings" if "proceedings" in r["type"] else "misc"
        if typ == "inproceedings":
            f["booktitle"] = f.pop("journal")
        ax = [x for x in rec_list[1:] if x["source"] == "arxiv"]
        if ax:
            f["eprint"], f["archiveprefix"] = ax[0]["arxiv"], "arXiv"
    else:
        typ = "misc"
        f = {"author": auth, "title": r["title"], "year": r["year"], "eprint": r["arxiv"], "archiveprefix": "arXiv",
             "howpublished": "arXiv preprint"}
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items() if v not in ("", None))
    return f"@{typ}{{{key},\n{body}\n}}\n"


def main():
    rows, raw, bibs = [], {}, []
    for e in ENTRIES:
        key, authors, year, venue, doi, aid, level = e
        recs, issues, statuses = [], [], []
        for kind, ident in [("doi", doi), ("arxiv", aid)]:
            if not ident:
                continue
            rec, st = crossref(ident) if kind == "doi" else arxiv(ident)
            raw[f"{key}:{kind}:{ident}"] = rec if rec else {"http_status": st}
            if rec is None:
                statuses.append("NOT_FOUND")
                msg = f"{kind} {ident} not found at {'CrossRef' if kind == 'doi' else 'arXiv'} (HTTP {st})"
                if kind == "doi":  # registration check only (e.g. DataCite DOIs); status stays NOT_FOUND
                    hst, hbody = get("https://doi.org/api/handles/" + ident)
                    reg = hst == 200 and '"responseCode":1' in hbody.replace(" ", "")
                    msg += f"; doi.org handle {'EXISTS (not a CrossRef DOI?)' if reg else 'not found'}"
                issues.append(msg)
            else:
                recs.append(rec)
                iss = compare(e, rec)
                statuses.append("MISMATCH" if iss else "VERIFIED")
                issues += [f"{kind}: {i}" for i in iss]
            time.sleep(1)
        if not doi and not aid:
            status, notes = "NOT_FOUND", "no DOI or arXiv id in Appendix B; not searched (no DOI invented)"
        elif "NOT_FOUND" in statuses:
            status, notes = "NOT_FOUND", "; ".join(issues)
        elif "MISMATCH" in statuses:
            status, notes = "MISMATCH", "; ".join(issues)
        else:
            status, notes = "VERIFIED", ""
        r0 = recs[0] if recs else {}
        rows.append({"key": key, "doi": doi or "", "arxiv": aid or "", "title": r0.get("title", ""),
                     "authors": "; ".join(r0.get("authors_full", [])), "year": r0.get("year", ""),
                     "venue": r0.get("venue", ""), "status": status, "notes": notes,
                     "draft_claim": f"{', '.join(authors) or '(none)'} ({year or 'n/a'}) {venue or ''}".strip(),
                     "appendix_b_level": level})
        if status == "VERIFIED":
            bibs.append(bib(key, recs))
        print(key, status, notes, flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_STATS, exist_ok=True)
    path = os.path.join(RESULTS_STATS, "p2_references_verified.csv")
    for p in [path, os.path.join(RESULTS_STATS, "p2_references.bib"), os.path.join(RESULTS_STATS, "p2_references_raw.json")]:
        assert not os.path.exists(p), p
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("# Appendix B of manuscript/paper2_draft_2026-10-14.md checked against CrossRef (DOI) and arXiv (id).\n")
        f.write("# Rules in app/src/p2_4_references.py docstring. Mismatches are listed, never corrected.\n")
        df.to_csv(f, index=False)
    with open(os.path.join(RESULTS_STATS, "p2_references.bib"), "w", encoding="utf-8") as f:
        f.write("% VERIFIED entries only (p2_references_verified.csv); fields from CrossRef / arXiv records.\n\n")
        f.write("\n".join(bibs))
    with open(os.path.join(RESULTS_STATS, "p2_references_raw.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    print(df["status"].value_counts().to_string(), flush=True)


if __name__ == "__main__":
    main()
