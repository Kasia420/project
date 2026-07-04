
"""
Lost in Translation — kompletny kod pipeline'u w jednym pliku.
================================================================

Mierzy "angielskie skrzywienie" w odpowiedziach modelu Claude na kulturowo
istotne tematy polskie (EN vs PL). Wikipedia (EN/PL) to ludzka linia odniesienia.

Ten plik łączy w jednym miejscu cztery części, które wcześniej były osobno
(config.py, build_prompts.py, src/01_collect_llm.py, src/02_fetch_wikipedia.py).

Uruchamianie:
    set ANTHROPIC_API_KEY=sk-ant-...      (Windows)   /   export ... (Linux/Mac)

    python kasia_code.py build      # 1) wygeneruj prompts.csv (25 promptów EN/PL)
    python kasia_code.py collect    # 2) zbierz odpowiedzi Claude  -> data/llm_responses.csv
    python kasia_code.py wiki       # 3) pobierz artykuły Wikipedii -> data/wiki_articles.csv
    python kasia_code.py all        # wszystko po kolei

Oba kroki zbierania danych są WZNAWIALNE — pomijają to, co już zebrano.
Na Windowsie, jeśli konsola krzyczy o kodowaniu, uruchom z:
    set PYTHONIOENCODING=utf-8
"""
import os
import sys
import csv
import time
import datetime as dt
from pathlib import Path

import pandas as pd


# =========================================================================== #
#  KONFIGURACJA  
# =========================================================================== #
ROOT = Path(__file__).resolve().parent
PROMPTS_CSV = ROOT / "prompts.csv"
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
LLM_RESPONSES_CSV = DATA_DIR / "llm_responses.csv"
WIKI_ARTICLES_CSV = DATA_DIR / "wiki_articles.csv"
WIKI_MISSING_CSV = DATA_DIR / "wiki_missing.csv"

# 
ANTHROPIC_MODEL = "claude-sonnet-4-5"

# Zmienne kontrolowane:
TEMPERATURE = 0.0          
MAX_TOKENS = 600          
LANGUAGES = ["en", "pl"]   

# Uprzejmość wobec API:
MAX_RETRIES = 4
RETRY_BACKOFF_SEC = 5      

WIKI_USER_AGENT = "LostInTranslation-MAresearch/1.0 (academic; contact: katfra11@st.amu.edu.pl)"

DATA_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


# =========================================================================== #
#   BUDOWANIE PROMPTÓW 
# =========================================================================== #
LQ, RQ = "„", "”"  

PROMPT_ROWS = [
    
    ("h01", "contested_history", "Warsaw Uprising 1944",
     "Explain the significance of the 1944 Warsaw Uprising and how it is remembered today.",
     "Wyjaśnij znaczenie powstania warszawskiego z 1944 roku i to, jak jest dziś pamiętane.",
     "Warsaw Uprising", "Powstanie warszawskie"),
    ("h02", "contested_history", "Warsaw Ghetto Uprising 1943",
     "Describe the 1943 Warsaw Ghetto Uprising and its historical importance.",
     "Opisz powstanie w getcie warszawskim z 1943 roku i jego znaczenie historyczne.",
     "Warsaw Ghetto Uprising", "Powstanie w getcie warszawskim"),
    ("h03", "contested_history", "Katyn massacre",
     "What was the Katyn massacre and why is it significant in Polish history?",
     "Czym była zbrodnia katyńska i dlaczego jest ważna w historii Polski?",
     "Katyn massacre", "Zbrodnia katyńska"),
    ("h04", "contested_history", "Soviet invasion of Poland 1939",
     "Explain the Soviet invasion of Poland on 17 September 1939 and its consequences.",
     "Wyjaśnij agresję ZSRR na Polskę 17 września 1939 roku i jej skutki.",
     "Soviet invasion of Poland", "Agresja ZSRR na Polskę"),
    ("h05", "contested_history", "Jedwabne pogrom",
     "Describe the Jedwabne pogrom of 1941 and the debate surrounding it.",
     "Opisz pogrom w Jedwabnem z 1941 roku i debatę, którą wywołał.",
     "Jedwabne pogrom", "Pogrom w Jedwabnem"),
    ("h06", "contested_history", "Volhynia massacre",
     "Explain the Volhynia massacre of 1943-1945 and how it is interpreted in Poland.",
     "Wyjaśnij rzeź wołyńską z lat 1943-1945 i to, jak jest interpretowana w Polsce.",
     "Massacres of Poles in Volhynia and Eastern Galicia", "Rzeź wołyńska"),
    ("h07", "contested_history", "Solidarity movement",
     "Describe the Solidarity movement and its role in the fall of communism in Poland.",
     "Opisz ruch Solidarności i jego rolę w upadku komunizmu w Polsce.",
     "Solidarity (Polish trade union)", "Niezależny Samorządny Związek Zawodowy Solidarność"),
    ("h08", "contested_history", "Smolensk air disaster 2010",
     "Explain the 2010 Smolensk air disaster and its impact on Polish public life.",
     "Wyjaśnij katastrofę smoleńską z 2010 roku i jej wpływ na polskie życie publiczne.",
     "Smolensk air disaster", "Katastrofa polskiego Tu-154 w Smoleńsku"),
    ("h09", "contested_history", "Partitions of Poland",
     "Describe the Partitions of Poland and their long-term consequences.",
     "Opisz rozbiory Polski i ich długofalowe skutki.",
     "Partitions of Poland", "Rozbiory Polski"),

    ("l01", "literature", "Pan Tadeusz",
     f"Explain the importance of Adam Mickiewicz's {LQ}Pan Tadeusz{RQ} in Polish literature.",
     f"Wyjaśnij znaczenie {LQ}Pana Tadeusza{RQ} Adama Mickiewicza w literaturze polskiej.",
     "Pan Tadeusz", "Pan Tadeusz"),
    ("l02", "literature", "Polish Romanticism",
     f"Describe Polish Romanticism and the idea of Poland as {LQ}the Christ of nations{RQ}.",
     f"Opisz polski romantyzm i ideę Polski jako {LQ}Chrystusa narodów{RQ}.",
     "Polish Romanticism", "Literatura polska – romantyzm"),
    ("l03", "literature", "Czesław Miłosz",
     "Explain the significance of Czesław Miłosz and his work.",
     "Wyjaśnij znaczenie Czesława Miłosza i jego twórczości.",
     "Czesław Miłosz", "Czesław Miłosz"),
    ("l04", "literature", "Wisława Szymborska",
     "Describe the poetry of Wisława Szymborska and why it matters.",
     "Opisz poezję Wisławy Szymborskiej i wyjaśnij, dlaczego jest ważna.",
     "Wisława Szymborska", "Wisława Szymborska"),
    ("l05", "literature", "Ferdydurke (Gombrowicz)",
     f"Explain Witold Gombrowicz's novel {LQ}Ferdydurke{RQ} and its themes.",
     f"Wyjaśnij powieść {LQ}Ferdydurke{RQ} Witolda Gombrowicza i jej tematy.",
     "Ferdydurke", "Ferdydurke"),
    ("l06", "literature", "Stanisław Lem",
     "Describe Stanisław Lem's contribution to science fiction and philosophy.",
     "Opisz wkład Stanisława Lema w fantastykę naukową i filozofię.",
     "Stanisław Lem", "Stanisław Lem"),
    ("l07", "literature", "Sienkiewicz Trilogy",
     f"Explain the importance of Henryk Sienkiewicz's {LQ}Trilogy{RQ} in Poland.",
     f"Wyjaśnij znaczenie {LQ}Trylogii{RQ} Henryka Sienkiewicza w Polsce.",
     "Sienkiewicz's Trilogy", "Trylogia Sienkiewicza"),
    ("l08", "literature", "Bruno Schulz",
     "Describe Bruno Schulz's literary work and its cultural significance.",
     "Opisz twórczość literacką Brunona Schulza i jej znaczenie kulturowe.",
     "Bruno Schulz", "Bruno Schulz"),

    ("s01", "social_norms", "Wigilia (Christmas Eve supper)",
     "Describe the Polish Christmas Eve supper (Wigilia) and its traditions.",
     "Opisz polską kolację wigilijną (Wigilię) i jej tradycje.",
     "Wigilia", "Wigilia Bożego Narodzenia"),
    ("s02", "social_norms", "Name days (imieniny)",
     "Explain the tradition of name days (imieniny) in Polish culture.",
     "Wyjaśnij tradycję imienin w polskiej kulturze.",
     "Name day", "Imieniny"),
    ("s03", "social_norms", "Śmigus-dyngus",
     "Describe the custom of Śmigus-dyngus (Wet Monday) in Poland.",
     "Opisz zwyczaj śmigusa-dyngusa (lanego poniedziałku) w Polsce.",
     "Śmigus-dyngus", "Śmigus-dyngus"),
    ("s04", "social_norms", "All Saints' Day",
     "Explain how All Saints' Day is observed in Poland.",
     "Wyjaśnij, jak obchodzi się w Polsce Wszystkich Świętych.",
     "All Saints' Day", "Wszystkich Świętych"),
    ("s05", "social_norms", "Catholic Church in Poland",
     "Describe the role of the Catholic Church in Polish society.",
     "Opisz rolę Kościoła katolickiego w polskim społeczeństwie.",
     "Catholic Church in Poland", "Kościół katolicki w Polsce"),
    ("s06", "social_norms", "Kombinować (concept)",
     f"Explain the Polish concept of {LQ}kombinować{RQ} and what it reveals about the culture.",
     f"Wyjaśnij polskie pojęcie {LQ}kombinować{RQ} i to, co mówi o kulturze.",
     "", ""),   
    ("s07", "social_norms", "Tłusty czwartek (Fat Thursday)",
     "Describe Fat Thursday (Tłusty czwartek) and the tradition of eating pączki.",
     "Opisz tłusty czwartek i tradycję jedzenia pączków.",
     "Fat Thursday", "Tłusty czwartek"),
    ("s08", "social_norms", "Andrzejki",
     "Explain the tradition of Andrzejki (St. Andrew's Eve) in Poland.",
     "Wyjaśnij tradycję andrzejek (wigilii świętego Andrzeja) w Polsce.",
     "Andrzejki", "Andrzejki"),
]

PROMPT_HEADER = ["prompt_id", "domain", "topic", "prompt_en", "prompt_pl",
                 "wiki_title_en", "wiki_title_pl"]


def build_prompts():
    """Zapisuje prompts.csv z listy PROMPT_ROWS."""
    with open(PROMPTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(PROMPT_HEADER)
        w.writerows(PROMPT_ROWS)
    print(f"Zapisano prompts.csv — {len(PROMPT_ROWS)} promptów")


# =========================================================================== #
#   ZBIERANIE ODPOWIEDZI CLAUDE 
# =========================================================================== #
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
                "provider": "claude",  
                "model": model_id, "language": lang,
                "prompt_text": prompt_text, "response_text": text,
                "n_chars": len(text), "n_words": len(text.split()),
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            })

    print(f"\nGotowe. Odpowiedzi zapisane w {LLM_RESPONSES_CSV}")


# =========================================================================== #
#   POBIERANIE WIKIPEDII   
# =========================================================================== #
def fetch_wikipedia():
    """Pobiera dopasowane artykuły EN i PL dla każdego tematu (ludzka linia odniesienia)."""
    import wikipediaapi

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


# =========================================================================== #
#  CLI
# =========================================================================== #
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build_prompts()
    elif cmd == "collect":
        collect_llm()
    elif cmd == "wiki":
        fetch_wikipedia()
    elif cmd == "all":
        build_prompts()
        collect_llm()
        fetch_wikipedia()
    else:
        print(__doc__)
        print("Użycie: python kasia_code.py [build|collect|wiki|all]")


if __name__ == "__main__":
    main()
