"""Etap 1 — szybki test: czy API działa i czy Claude sensownie odpowiada po polsku.

Zero struktury, zero zapisu do pliku. Cel: zobaczyć jedną parę odpowiedzi EN/PL
na ten sam temat i ocenić, czy pomysł w ogóle ma sens.

Uruchamianie:
    set ANTHROPIC_API_KEY=sk-ant-...     (Windows)  /  export ... (Linux/Mac)
    python stage1_api_poc.py
"""
import anthropic

client = anthropic.Anthropic()   # klucz bierze z zmiennej ANTHROPIC_API_KEY

TEST_PROMPTS = [
    ("en", "Explain the significance of the 1944 Warsaw Uprising and how it is remembered today."),
    ("pl", "Wyjaśnij znaczenie powstania warszawskiego z 1944 roku i to, jak jest dziś pamiętane."),
]

for lang, prompt in TEST_PROMPTS:
    print("=" * 70)
    print(f"[{lang}] {prompt}\n")
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=600,        # przycinam, żeby odpowiedzi były porównywalnej długości
        temperature=0.0,       # deterministycznie — chcę powtarzalne odpowiedzi
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text     # naiwnie zakładam, że pierwszy blok to tekst
    print(text)
    print(f"\n--- {len(text.split())} słów, {len(text)} znaków")
