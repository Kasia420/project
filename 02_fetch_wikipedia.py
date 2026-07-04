"""Etap 4 — pobieranie dopasowanych artykułów Wikipedii EN/PL.

    pip install wikipedia-api
    python src/02_fetch_wikipedia.py
"""
import sys
from pathlib import Path

# config.py leży piętro wyżej niż src/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import wikipediaapi

from config import (LANGUAGES, PROMPTS_CSV, WIKI_ARTICLES_CSV, WIKI_MISSING_CSV,
                    WIKI_USER_AGENT)


def _append_row(path, row):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def fetch_wikipedia():
    """Pobiera dopasowane artykuły EN i PL dla każdego tematu."""
    prompts = pd.read_csv(PROMPTS_CSV).fillna("")   
    wikis = {
        "en": wikipediaapi.Wikipedia(user_agent=WIKI_USER_AGENT, language="en"),
        "pl": wikipediaapi.Wikipedia(user_agent=WIKI_USER_AGENT, language="pl"),
    }

    done = set()
    if WIKI_ARTICLES_CSV.exists():
        prev = pd.read_csv(WIKI_ARTICLES_CSV)
        done = set(zip(prev["prompt_id"], prev["language"]))

    n_ok, n_missing = 0, 0
    for _, p in prompts.iterrows():
        for lang in LANGUAGES:
            key = (p["prompt_id"], lang)
            if key in done:
                print(f"{p['prompt_id']} {lang}  (pomijam, już pobrane)")
                continue

            title = p["wiki_title_en"] if lang == "en" else p["wiki_title_pl"]
            title = str(title).strip()
            if not title:
                print(f"{p['prompt_id']} {lang}  (brak tytułu — zapisuję jako missing)")
                _append_row(WIKI_MISSING_CSV, {
                    "prompt_id": p["prompt_id"], "language": lang,
                    "topic": p["topic"], "title": "", "reason": "no_title_in_prompts_csv"})
                n_missing += 1
                continue

            page = wikis[lang].page(title)
            if not page.exists():
                print(f"{p['prompt_id']} {lang}  BRAK: '{title}'")
                _append_row(WIKI_MISSING_CSV, {
                    "prompt_id": p["prompt_id"], "language": lang,
                    "topic": p["topic"], "title": title, "reason": "page_not_found"})
                n_missing += 1
                continue

            text = page.text
            _append_row(WIKI_ARTICLES_CSV, {
                "prompt_id": p["prompt_id"], "domain": p["domain"], "topic": p["topic"],
                "language": lang, "wiki_title": page.title, "wiki_url": page.fullurl,
                "summary": page.summary, "full_text": text,
                "n_chars": len(text), "n_words": len(text.split()),
            })
            print(f"{p['prompt_id']} {lang}  OK: '{page.title}' ({len(text.split())} słów)")
            n_ok += 1

    print(f"\nGotowe. {n_ok} artykułów -> {WIKI_ARTICLES_CSV}")
    if n_missing:
        print(f"{n_missing} nierozwiązanych tytułów -> {WIKI_MISSING_CSV} "
              f"(popraw w prompts.csv i uruchom ponownie)")


if __name__ == "__main__":
    fetch_wikipedia()
