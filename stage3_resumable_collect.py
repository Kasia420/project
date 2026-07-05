"""Etap 3 — odporne zbieranie: ponawianie, zapis po każdej odpowiedzi, wznawianie.

Motywacja: pełny przebieg z etapu 2 wywalił się w połowie na timeout i wszystkie
zebrane odpowiedzi przepadły. Stąd trzy zmiany:

  1. call_with_retry() — wykładnicze ponawianie (5s, 10s, 20s...) zamiast time.sleep
  2. _append_row()    — dopisywanie do CSV po KAŻDEJ odpowiedzi, nie na końcu
  3. zbiór `done`     — przy starcie czytam, co już mam, i pomijam zebrane pary
                        (skrypt można bezpiecznie przerwać i odpalić ponownie)

Dodatkowo:
  * wyciąganie tekstu przez filtrowanie bloków po typie — czasem pierwszy blok
    odpowiedzi NIE jest tekstem, więc msg.content[0].text potrafi się wysypać
  * get_anthropic_caller() — czytelny komunikat, gdy brak klucza albo SDK
  * ścieżki przez pathlib względem pliku, dane w podkatalogu data/

prompts.csv nadal generuje skrypt z etapu 2 (ta sama lista próbna).
Na Windowsie przy krzykach o kodowaniu: set PYTHONIOENCODING=utf-8

Uruchamianie:
    python stage3_resumable_collect.py
"""
import os
import sys
import time
import datetime as dt
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
#  Konfiguracja (na razie na górze pliku; jak urośnie — wyniosę do config.py)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
PROMPTS_CSV = ROOT / "prompts.csv"
DATA_DIR = ROOT / "data"
LLM_RESPONSES_CSV = DATA_DIR / "llm_responses.csv"

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0.0
MAX_TOKENS = 600
LANGUAGES = ["en", "pl"]

MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5

DATA_DIR.mkdir(exist_ok=True)


def get_anthropic_caller():
    """Zwraca (model_id, fn) dla Anthropic, albo None gdy brak klucza/SDK."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        print("  ! brak SDK anthropic (pip install anthropic)")
        return None
    client = anthropic.Anthropic()

    def call(prompt_text):
        msg = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            messages=[{"role": "user", "content": prompt_text}],
        )
        return "".join(b.text for b in msg.content
                       if getattr(b, "type", None) == "text").strip()

    return (MODEL, call)


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
    """Dopisuje jeden wiersz do CSV; nagłówek tylko przy tworzeniu pliku."""
    df = pd.DataFrame([row])
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False, encoding="utf-8")


def collect():
    prompts = pd.read_csv(PROMPTS_CSV)

    caller = get_anthropic_caller()
    if not caller:
        print("Brak ANTHROPIC_API_KEY (albo SDK). Ustaw klucz i uruchom ponownie.")
        sys.exit(1)
    model_id, call = caller

    # wznawialność: co już zebrane?
    done = set()
    if LLM_RESPONSES_CSV.exists():
        prev = pd.read_csv(LLM_RESPONSES_CSV)
        done = set(zip(prev["prompt_id"], prev["language"]))

    total = len(prompts) * len(LANGUAGES)
    n = 0
    print(f"Zbieram | prompty: {len(prompts)} | języki: {LANGUAGES} "
          f"| temp={TEMPERATURE} max_tokens={MAX_TOKENS}\n")

    for _, p in prompts.iterrows():
        for lang in LANGUAGES:
            n += 1
            key = (p["prompt_id"], lang)
            tag = f"[{n}/{total}] {p['prompt_id']} {lang}"
            if key in done:
                print(f"{tag}  (pomijam, już zebrane)")
                continue
            prompt_text = p["prompt_en"] if lang == "en" else p["prompt_pl"]
            print(f"{tag}  pytam...")
            text = call_with_retry(call, prompt_text)
            if text is None:
                continue   # nieudane wiersze zbiorę przy kolejnym uruchomieniu
            _append_row(LLM_RESPONSES_CSV, {
                "prompt_id": p["prompt_id"], "domain": p["domain"], "topic": p["topic"],
                "model": model_id, "language": lang,
                "prompt_text": prompt_text, "response_text": text,
                "n_chars": len(text), "n_words": len(text.split()),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })

    print(f"\nGotowe. Odpowiedzi zapisane w {LLM_RESPONSES_CSV}")


if __name__ == "__main__":
    collect()
