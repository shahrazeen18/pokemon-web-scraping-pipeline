"""
Automated Web Scraping & Data Extraction Pipeline
Source: https://pokemondb.net/pokedex/all  (public Pokemon stats database)

Pipeline steps:
  1. FETCH     - download the raw HTML (polite headers, retries, robots.txt check, cached to disk)
  2. EXTRACT   - pull raw text for each Pokemon row with BeautifulSoup
  3. CLEAN     - strip whitespace artifacts, split name/form, split types, convert numbers
  4. VALIDATE  - check stats add up to the listed total, drop duplicates / bad rows
  5. SAVE      - write to CSV and SQLite database

Run:   python pokemon_scraper.py
Offline re-run (uses cached HTML):   python pokemon_scraper.py --from-file data/raw/pokedex_all.html
"""

import argparse
import logging
import re
import sqlite3
import time
from pathlib import Path
from urllib import robotparser

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://pokemondb.net/pokedex/all"
USER_AGENT = "Mozilla/5.0 (student-project; educational scraping pipeline)"
DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "raw" / "pokedex_all.html"
CSV_FILE = DATA_DIR / "pokemon_clean.csv"
DB_FILE = DATA_DIR / "pokemon.db"

STAT_COLS = ["hp", "attack", "defense", "sp_atk", "sp_def", "speed"]

# Pokedex number ranges per generation (used as a derived field)
GENERATIONS = [
    (1, 151), (152, 251), (252, 386), (387, 493), (494, 649),
    (650, 721), (722, 809), (810, 905), (906, 1025),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("pokemon_pipeline")


# 1. FETCH
def allowed_by_robots(url: str) -> bool:
    """Check robots.txt using our own User-Agent, then apply its real rules."""
    try:
        resp = requests.get(
            "https://pokemondb.net/robots.txt",
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
    except requests.RequestException as exc:
        log.warning("Could not download robots.txt (%s). Continuing.", exc)
        return True

    if resp.status_code != 200:
        log.warning("robots.txt returned status %s. Continuing with one polite request.", resp.status_code)
        return True

    rp = robotparser.RobotFileParser()
    rp.parse(resp.text.splitlines())
    allowed = rp.can_fetch(USER_AGENT, url)
    if not allowed:
        log.error("robots.txt disallows this URL for our User-Agent.")
    return allowed

def fetch_html(url: str, retries: int = 3, delay: float = 2.0) -> str:
    """Download the page with retries. Saves the raw HTML so we never need to re-hit the site."""
    if not allowed_by_robots(url):
        raise PermissionError(f"robots.txt does not allow scraping {url}")

    headers = {"User-Agent": USER_AGENT}
    for attempt in range(1, retries + 1):
        try:
            log.info("Fetching %s (attempt %d/%d)", url, attempt, retries)
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
            RAW_FILE.write_text(resp.text, encoding="utf-8")
            log.info("Saved raw HTML to %s (%d KB)", RAW_FILE, len(resp.text) // 1024)
            return resp.text
        except requests.RequestException as exc:
            log.warning("Request failed: %s", exc)
            if attempt < retries:
                time.sleep(delay * attempt)  # back off a bit more each time
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts")


# 2. EXTRACT
def extract_rows(html: str) -> list[dict]:
    """Pull the RAW text of each table row (no cleaning yet)."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="pokedex")
    if table is None:
        raise ValueError("Could not find the pokedex table. The site layout may have changed.")

    records = []
    for tr in table.find("tbody").find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 10:
            continue  # skip malformed rows

        name_cell = tds[1]
        main_name = name_cell.find("a", class_="ent-name")
        form_tag = name_cell.find("small")

        records.append({
            "raw_number": tds[0].get_text(),
            "raw_name": main_name.get_text() if main_name else name_cell.get_text(),
            "raw_form": form_tag.get_text() if form_tag else "",
            "raw_types": [a.get_text() for a in tds[2].find_all("a")],
            "raw_total": tds[3].get_text(),
            "raw_stats": [td.get_text() for td in tds[4:10]],
        })
    log.info("Extracted %d raw rows", len(records))
    return records


# 3. CLEAN  (text filtering logic)
def clean_text(text: str) -> str:
    """Remove whitespace artifacts: newlines, tabs, non-breaking spaces, repeated spaces."""
    text = text.replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def to_int(text: str):
    """Pull the first integer out of messy text, e.g. ' #0025\n' -> 25."""
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def get_generation(dex_number: int):
    for gen, (low, high) in enumerate(GENERATIONS, start=1):
        if low <= dex_number <= high:
            return gen
    return None


def get_form_category(form: str) -> str:
    """Classify the form text with simple pattern matching."""
    if not form:
        return "Base"
    f = form.lower()
    if re.search(r"\bmega\b", f):
        return "Mega"
    if re.search(r"\bgigantamax\b", f):
        return "Gigantamax"
    if re.search(r"\b(alolan|galarian|hisuian|paldean)\b", f):
        return "Regional"
    return "Other form"


def clean_records(records: list[dict]) -> pd.DataFrame:
    cleaned = []
    for r in records:
        dex = to_int(r["raw_number"])
        name = clean_text(r["raw_name"])
        form = clean_text(r["raw_form"])
        types = [clean_text(t) for t in r["raw_types"] if clean_text(t)]
        stats = [to_int(s) for s in r["raw_stats"]]

        if dex is None or not name or len(stats) != 6 or None in stats:
            log.debug("Skipping bad row: %s", r)
            continue

        row = {
            "dex_number": dex,
            "name": name,
            "form": form if form else None,
            "form_category": get_form_category(form),
            "type_1": types[0] if types else None,
            "type_2": types[1] if len(types) > 1 else None,
            "total": to_int(r["raw_total"]),
            "generation": get_generation(dex),
        }
        row.update(dict(zip(STAT_COLS, stats)))
        cleaned.append(row)

    return pd.DataFrame(cleaned)


# 4. VALIDATE
def validate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["dex_number", "name", "form"]).reset_index(drop=True)
    if len(df) < before:
        log.info("Removed %d duplicate rows", before - len(df))

    # The six stats should add up to the listed total
    mismatch = df[df[STAT_COLS].sum(axis=1) != df["total"]]
    if len(mismatch):
        log.warning("%d rows where stats do not add up to total. Dropping them.", len(mismatch))
        df = df.drop(mismatch.index).reset_index(drop=True)

    log.info("Validation passed: %d clean records", len(df))
    return df


# 5. SAVE
def save_outputs(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # utf-8-sig so Excel shows accents correctly (e.g. Flabebe with the accent)
    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig")
    log.info("Saved CSV: %s", CSV_FILE)

    with sqlite3.connect(DB_FILE) as conn:
        df.to_sql("pokemon", conn, if_exists="replace", index=False)
    log.info("Saved SQLite database: %s (table: pokemon)", DB_FILE)


# MAIN
def run(from_file: str | None = None) -> pd.DataFrame:
    if from_file:
        log.info("Reading cached HTML from %s", from_file)
        html = Path(from_file).read_text(encoding="utf-8")
    else:
        html = fetch_html(URL)

    raw = extract_rows(html)
    df = clean_records(raw)
    df = validate(df)
    save_outputs(df)

    print("\n--- Preview ---")
    print(df.head(10).to_string(index=False))
    print(f"\nTotal records: {len(df)}")
    print("\nAverage attack by primary type (top 5):")
    print(df.groupby("type_1")["attack"].mean().round(1).sort_values(ascending=False).head(5))
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pokemon web scraping pipeline")
    parser.add_argument("--from-file", help="Use a previously saved HTML file instead of downloading")
    args = parser.parse_args()
    run(args.from_file)