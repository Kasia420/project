"""Etap 4 — zbieranie odpowiedzi Claude (wersja wielo-plikowa).
:
    python src/01_collect_llm.py
"""
import os
import sys
import time
import datetime as dt
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import (ANTHROPIC_MODEL, LANGUAGES, LLM_RESPONSES_CSV, MAX_RETRIES,
                    MAX_TOKENS, PROMPTS_CSV, RETRY_BACKOFF_SEC, TEMPERATURE)


def get_anthropic_caller():
    """Zwraca (model_id, fn) dla Anthropic, albo None gdy brak klucza/SDK."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        print("  ! brak SDK anthropic (pip install anthropic) — pomijam Claude")
        return None
    client = anthropic.Anthropic()

    def call(prompt_text):
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()

    return (ANTHROPIC_MODEL, call)


def call_with_retry(fn, prompt_text):
    """Wywołanie API z wykładniczym ponawianiem."""
    delay = RETRY_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(prompt_text)
        except Exception as e:
            if attempt == MAX_RETRIES:
                print(f"    x nieudane po {attempt} próbach: {e}")
                return None
            print(f"    . ponawiam {attempt}/{MAX_RETRIES - 1} za {delay}s ({type(e).__name__})")
            time.sleep(delay)
            delay *= 2
    return None


def _append_row(path, row):
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def collect_llm():
    """Pyta Claude każdym promptem po EN i PL; zapisuje do data/llm_responses.csv."""
    prompts = pd.read_csv(PROMPTS_CSV)

    caller = get_anthropic_caller()
    if not caller:
        print("Brak ANTHROPIC_API_KEY (albo SDK). Ustaw klucz i uruchom ponownie.")
        sys.exit(1)
    model_id, call = caller

    done = set()
    if LLM_RESPONSES_CSV.exists():
        prev = pd.read_csv(LLM_RESPONSES_CSV)
        done = set(zip(prev["prompt_id"], prev["provider"], prev["language"]))

    total = len(prompts) * len(LANGUAGES)
    n = 0
    print(f"Zbieram: claude | prompty: {len(prompts)} | języki: {LANGUAGES} "
          f"| temp={TEMPERATURE} max_tokens={MAX_TOKENS}\n")

    for _, p in prompts.iterrows():
        for lang in LANGUAGES:
            n += 1
            key = (p["prompt_id"], "claude", lang)
            tag = f"[{n}/{total}] {p['prompt_id']} claude {lang}"
            if key in done:
                print(f"{tag}  (pomijam, już zebrane)")
                continue
            prompt_text = p["prompt_en"] if lang == "en" else p["prompt_pl"]
            print(f"{tag}  pytam...")
            text = call_with_retry(call, prompt_text)
            if text is None:
                continue
            _append_row(LLM_RESPONSES_CSV, {
                "prompt_id": p["prompt_id"], "domain": p["domain"], "topic": p["topic"],
                "provider": "claude", "model": model_id, "language": lang,
                "prompt_text": prompt_text, "response_text": text,
                "n_chars": len(text), "n_words": len(text.split()),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })

    print(f"\nGotowe. Odpowiedzi zapisane w {LLM_RESPONSES_CSV}")


if __name__ == "__main__":
    collect_llm()
