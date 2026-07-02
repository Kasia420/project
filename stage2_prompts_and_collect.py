"""Etap 2 — stała lista promptów EN/PL -> prompts.csv, potem pętla zbierająca.

Nowości względem etapu 1:
  * prompty jako dane (CSV), nie kod — łatwiej pokazać promotorowi i cytować w pracy
  * identyfikatory (h/l/s + numer) i domeny: contested_history / literature / social_norms
  * odpowiedzi lądują w llm_responses.csv z metadanymi (długość, timestamp)

Na razie tylko po 2 tematy z każdej domeny — sprawdzam układ kolumn, zanim
przepiszę pełną listę 25 par.

Uruchamianie:
    python stage2_prompts_and_collect.py build      # wygeneruj prompts.csv
    python stage2_prompts_and_collect.py collect    # zbierz odpowiedzi
"""
import csv
import sys
import time
import datetime as dt

import anthropic
import pandas as pd

MODEL = "claude-sonnet-4-5"
TEMPERATURE = 0.0
MAX_TOKENS = 600
LANGUAGES = ["en", "pl"]

PROMPT_HEADER = ["prompt_id", "domain", "topic", "prompt_en", "prompt_pl"]

PROMPT_ROWS = [
    ("h01", "contested_history", "Warsaw Uprising 1944",
     "Explain the significance of the 1944 Warsaw Uprising and how it is remembered today.",
     "Wyjaśnij znaczenie powstania warszawskiego z 1944 roku i to, jak jest dziś pamiętane."),
    ("h02", "contested_history", "Katyn massacre",
     "What was the Katyn massacre and why is it significant in Polish history?",
     "Czym była zbrodnia katyńska i dlaczego jest ważna w historii Polski?"),
    ("l01", "literature", "Pan Tadeusz",
     "Explain the importance of Adam Mickiewicz's „Pan Tadeusz” in Polish literature.",
     "Wyjaśnij znaczenie „Pana Tadeusza” Adama Mickiewicza w literaturze polskiej."),
    ("l02", "literature", "Wisława Szymborska",
     "Describe the poetry of Wisława Szymborska and why it matters.",
     "Opisz poezję Wisławy Szymborskiej i wyjaśnij, dlaczego jest ważna."),
    ("s01", "social_norms", "Wigilia (Christmas Eve supper)",
     "Describe the Polish Christmas Eve supper (Wigilia) and its traditions.",
     "Opisz polską kolację wigilijną (Wigilię) i jej tradycje."),
    ("s02", "social_norms", "Name days (imieniny)",
     "Explain the tradition of name days (imieniny) in Polish culture.",
     "Wyjaśnij tradycję imienin w polskiej kulturze."),
]


def build_prompts():
    with open("prompts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(PROMPT_HEADER)
        w.writerows(PROMPT_ROWS)
    print(f"Zapisano prompts.csv — {len(PROMPT_ROWS)} promptów")


def collect():
    prompts = pd.read_csv("prompts.csv")
    client = anthropic.Anthropic()

    rows = []
    for _, p in prompts.iterrows():
        for lang in LANGUAGES:
            prompt_text = p["prompt_en"] if lang == "en" else p["prompt_pl"]
            print(f"{p['prompt_id']} {lang}  pytam...")
            msg = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = msg.content[0].text.strip()
            rows.append({
                "prompt_id": p["prompt_id"], "domain": p["domain"], "topic": p["topic"],
                "model": MODEL, "language": lang,
                "prompt_text": prompt_text, "response_text": text,
                "n_chars": len(text), "n_words": len(text.split()),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })
            time.sleep(1.0)   # uprzejmość wobec API

    pd.DataFrame(rows).to_csv("llm_responses.csv", index=False, encoding="utf-8")
    print(f"Gotowe. {len(rows)} odpowiedzi w llm_responses.csv")
    # TODO: jeśli coś padnie w połowie pętli (timeout, rate limit), tracę WSZYSTKO,
    #       bo zapisuję dopiero na końcu. Przy 50 wywołaniach to realne ryzyko.


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build_prompts()
    elif cmd == "collect":
        collect()
    else:
        print("Użycie: python stage2_prompts_and_collect.py [build|collect]")
