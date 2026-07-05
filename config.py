"""Etap 4 — wspólna konfiguracja projektu.

Struktura projektu na tym etapie:
    config.py                  <- ten plik
    build_prompts.py           <- generuje prompts.csv (pełne 25 par)
    src/01_collect_llm.py      <- zbiera odpowiedzi Claude
    src/02_fetch_wikipedia.py  <- pobiera artykuły Wikipedii (linia odniesienia)
    src/03..05_*.py            <- (planowane) podobieństwo, sentyment, wykresy

Stałe wyniesione tutaj, żeby oba skrypty zbierające dzieliły jedną prawdę
o ścieżkach i parametrach.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROMPTS_CSV = ROOT / "prompts.csv"
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"          # na przyszłe skrypty 03-05 (wykresy)
LLM_RESPONSES_CSV = DATA_DIR / "llm_responses.csv"
WIKI_ARTICLES_CSV = DATA_DIR / "wiki_articles.csv"
WIKI_MISSING_CSV = DATA_DIR / "wiki_missing.csv"

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
