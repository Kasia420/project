# -*- coding: utf-8 -*-
"""
Lost in Translation — analiza i wizualizacja  (krok 4-6 pipeline'u)
===================================================================

Wejście  (z kroków collect/wiki):
    data/llm_responses.csv   — 50 odpowiedzi Claude (25 promptów x EN/PL)
    data/wiki_articles.csv   — 48 artykułów Wikipedii (24 tematy x EN/PL) = linia odniesienia

Co liczy:
    1. Podobieństwo semantyczne (cosine na wielojęzycznych embeddingach):
         - sim_en_pl        : EN-odpowiedź  vs  PL-odpowiedź   (homogenizacja)
         - sim_*_resp_*_wiki: odpowiedź vs artykuł Wikipedii (po obu stronach)
         - native_pull      : sim(odpowiedź, wiki w TYM SAMYM języku)
                              - sim(odpowiedź, wiki w DRUGIM języku)
           Hipoteza "angielskiej soczewki": native_pull(PL) < native_pull(EN),
           tzn. polskie odpowiedzi są SŁABIEJ zakotwiczone w polskim dyskursie,
           niż angielskie w angielskim.
    2. Sentyment (wielojęzyczny XLM-RoBERTa): polarity = P(pos) - P(neg),
       porównanie EN vs PL parami i wg domeny.
    3. Statystyka: testy par Wilcoxona (n=25 / 24), mediany, wielkości efektu.
    4. Rysunki -> figures/  (każdy z tytułem, osiami, legendą; podpisy w captions.txt).

Uruchamianie (na własnej maszynie — pierwszy raz pobiera modele, ~1-2 GB):
    pip install -r requirements.txt
    python kasia_code_analyze.py

Modele można podmienić niżej (EMBED_MODEL / SENTIMENT_MODEL).
Ciężkie importy (torch, sentence-transformers, transformers) są LENIWE — ładują
się dopiero gdy faktycznie liczymy embeddingi/sentyment, więc samo wczytanie
pliku i logika danych działa bez nich.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
#  Konfiguracja
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

LLM_RESPONSES_CSV = DATA_DIR / "llm_responses.csv"
WIKI_ARTICLES_CSV = DATA_DIR / "wiki_articles.csv"

PER_PROMPT_CSV = DATA_DIR / "analysis_per_prompt.csv"
STATS_TXT = DATA_DIR / "stats_summary.txt"
CAPTIONS_TXT = FIGURES_DIR / "captions.txt"

# LaBSE jest projektowany pod międzyjęzykowe podobieństwo zdań (pary tłumaczeń),
# więc dobrze pasuje do porównań EN<->PL. Alternatywa, lżejsza:
#   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBED_MODEL = "sentence-transformers/LaBSE"
SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

DOMAIN_ORDER = ["contested_history", "literature", "social_norms"]
DOMAIN_LABELS = {
    "contested_history": "Contested history",
    "literature": "Literature",
    "social_norms": "Social norms",
}


# --------------------------------------------------------------------------- #
#  Wczytanie i sparowanie danych (działa bez ciężkich bibliotek)
# --------------------------------------------------------------------------- #
def load_paired():
    """Zwraca DataFrame: jeden wiersz na prompt, kolumny EN/PL dla odpowiedzi
    i (jeśli istnieje) streszczeń Wikipedii."""
    if not LLM_RESPONSES_CSV.exists():
        sys.exit(f"Brak {LLM_RESPONSES_CSV}. Uruchom najpierw collect/wiki.")
    r = pd.read_csv(LLM_RESPONSES_CSV)
    w = pd.read_csv(WIKI_ARTICLES_CSV) if WIKI_ARTICLES_CSV.exists() else pd.DataFrame()

    resp = (r.pivot_table(index=["prompt_id", "domain", "topic"],
                          columns="language", values="response_text",
                          aggfunc="first")
              .reset_index()
              .rename(columns={"en": "resp_en", "pl": "resp_pl"}))
    resp.columns.name = None

    # długości (zmienna kontrolna / opisowa)
    rl = (r.pivot_table(index="prompt_id", columns="language",
                        values="n_words", aggfunc="first")
            .reset_index()
            .rename(columns={"en": "words_en", "pl": "words_pl"}))
    rl.columns.name = None
    out = resp.merge(rl, on="prompt_id", how="left")

    if not w.empty:
        ws = (w.pivot_table(index="prompt_id", columns="language",
                            values="summary", aggfunc="first")
                .reset_index()
                .rename(columns={"en": "wiki_en", "pl": "wiki_pl"}))
        ws.columns.name = None
        out = out.merge(ws, on="prompt_id", how="left")
    else:
        out["wiki_en"] = np.nan
        out["wiki_pl"] = np.nan

    out["domain"] = pd.Categorical(out["domain"], categories=DOMAIN_ORDER, ordered=True)
    return out.sort_values(["domain", "prompt_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
#  Embeddingi + podobieństwo  (leniwy import)
# --------------------------------------------------------------------------- #
def _embedder():
    from sentence_transformers import SentenceTransformer
    print(f"  ladowanie modelu embeddingow: {EMBED_MODEL} ...")
    return SentenceTransformer(EMBED_MODEL)


def _encode(model, texts):
    """Zwraca znormalizowane wektory; puste teksty -> wektor zerowy (cos=0)."""
    texts = ["" if (t is None or (isinstance(t, float) and np.isnan(t))) else str(t)
             for t in texts]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(emb)


def _cos(a, b):
    """Cosine dla znormalizowanych wektorów = iloczyn skalarny, wierszami."""
    return np.sum(a * b, axis=1)


def add_similarity(df):
    """Dokłada kolumny podobieństwa do df (jeden wiersz na prompt)."""
    model = _embedder()
    e_resp_en = _encode(model, df["resp_en"])
    e_resp_pl = _encode(model, df["resp_pl"])
    e_wiki_en = _encode(model, df["wiki_en"])
    e_wiki_pl = _encode(model, df["wiki_pl"])

    df = df.copy()
    df["sim_en_pl"] = _cos(e_resp_en, e_resp_pl)                # homogenizacja
    df["sim_en_resp_en_wiki"] = _cos(e_resp_en, e_wiki_en)
    df["sim_pl_resp_pl_wiki"] = _cos(e_resp_pl, e_wiki_pl)
    df["sim_en_resp_pl_wiki"] = _cos(e_resp_en, e_wiki_pl)
    df["sim_pl_resp_en_wiki"] = _cos(e_resp_pl, e_wiki_en)

    # "native pull" — o ile bliżej odpowiedź jest do wiki we WŁASNYM jezyku
    # niz do wiki w drugim jezyku. Brak baseline (s06) -> NaN.
    has_base = df["wiki_en"].notna() & df["wiki_pl"].notna()
    df["native_pull_en"] = np.where(
        has_base, df["sim_en_resp_en_wiki"] - df["sim_en_resp_pl_wiki"], np.nan)
    df["native_pull_pl"] = np.where(
        has_base, df["sim_pl_resp_pl_wiki"] - df["sim_pl_resp_en_wiki"], np.nan)
    return df


# --------------------------------------------------------------------------- #
#  Sentyment  (leniwy import)
# --------------------------------------------------------------------------- #
def add_sentiment(df):
    from transformers import pipeline
    print(f"  ladowanie modelu sentymentu: {SENTIMENT_MODEL} ...")
    clf = pipeline("sentiment-analysis", model=SENTIMENT_MODEL,
                   top_k=None, truncation=True, max_length=512)

    def polarity(text):
        if text is None or (isinstance(text, float) and np.isnan(text)) or str(text).strip() == "":
            return np.nan
        scores = {d["label"].lower(): d["score"] for d in clf(str(text))[0]}
        # etykiety modelu: negative / neutral / positive
        return scores.get("positive", 0.0) - scores.get("negative", 0.0)

    df = df.copy()
    df["sent_en"] = df["resp_en"].map(polarity)
    df["sent_pl"] = df["resp_pl"].map(polarity)
    df["sent_diff"] = df["sent_pl"] - df["sent_en"]   # +: PL bardziej pozytywne
    return df


# --------------------------------------------------------------------------- #
#  Statystyka
# --------------------------------------------------------------------------- #
def _wilcoxon(a, b, label, lines):
    from scipy.stats import wilcoxon
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)
    med_a, med_b = np.median(a), np.median(b)
    try:
        stat, p = wilcoxon(a, b)
        # wielkość efektu: rank-biserial dla testu par
        diff = a - b
        nz = diff[diff != 0]
        from scipy.stats import rankdata
        ranks = rankdata(np.abs(nz))
        rbc = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum() if len(nz) else 0.0
        lines.append(f"{label}: n={n}  median(A)={med_a:+.3f}  median(B)={med_b:+.3f}  "
                     f"W={stat:.1f}  p={p:.4f}  rank-biserial={rbc:+.3f}")
    except ValueError as e:
        lines.append(f"{label}: n={n}  median(A)={med_a:+.3f}  median(B)={med_b:+.3f}  "
                     f"(Wilcoxon nie policzony: {e})")


def run_stats(df):
    lines = ["LOST IN TRANSLATION — podsumowanie statystyczne",
             "=" * 52, ""]

    lines.append("[1] HOMOGENIZACJA: cosine(EN-odpowiedz, PL-odpowiedz)")
    lines.append(f"    cala proba: median={df['sim_en_pl'].median():.3f}  "
                 f"mean={df['sim_en_pl'].mean():.3f}  sd={df['sim_en_pl'].std():.3f}")
    for d in DOMAIN_ORDER:
        s = df.loc[df["domain"] == d, "sim_en_pl"]
        lines.append(f"    {DOMAIN_LABELS[d]:<18}: median={s.median():.3f} (n={len(s)})")
    lines.append("")

    if "native_pull_en" in df:
        lines.append("[2] ANGIELSKA SOCZEWKA: native_pull(EN) vs native_pull(PL)")
        lines.append("    native_pull = sim(odpowiedz, wiki-wlasny-jezyk) - sim(odpowiedz, wiki-obcy-jezyk)")
        lines.append(f"    median native_pull(EN)={df['native_pull_en'].median():+.3f}  "
                     f"native_pull(PL)={df['native_pull_pl'].median():+.3f}")
        _wilcoxon(df["native_pull_en"], df["native_pull_pl"],
                  "    Wilcoxon native_pull EN vs PL", lines)
        lines.append("")

    if "sent_en" in df:
        lines.append("[3] SENTYMENT: polarity = P(pos)-P(neg)")
        lines.append(f"    median EN={df['sent_en'].median():+.3f}  "
                     f"PL={df['sent_pl'].median():+.3f}")
        _wilcoxon(df["sent_pl"], df["sent_en"], "    Wilcoxon sentyment PL vs EN", lines)
        for d in DOMAIN_ORDER:
            s = df.loc[df["domain"] == d]
            lines.append(f"    {DOMAIN_LABELS[d]:<18}: EN={s['sent_en'].median():+.3f}  "
                         f"PL={s['sent_pl'].median():+.3f}  diff={s['sent_diff'].median():+.3f}")
        lines.append("")

    lines.append("[4] DLUGOSC ODPOWIEDZI (kontrola):")
    lines.append(f"    median slow EN={df['words_en'].median():.0f}  "
                 f"PL={df['words_pl'].median():.0f}")

    text = "\n".join(lines)
    STATS_TXT.write_text(text, encoding="utf-8")
    print("\n" + text + f"\n\n-> zapisano {STATS_TXT}")


# --------------------------------------------------------------------------- #
#  Rysunki
# --------------------------------------------------------------------------- #
def make_figures(df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")

    captions = []
    dlab = df.assign(domain_label=df["domain"].map(DOMAIN_LABELS))

    # FIG 1 — homogenizacja EN<->PL wg domeny
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=dlab, x="domain_label", y="sim_en_pl", ax=ax,
                order=[DOMAIN_LABELS[d] for d in DOMAIN_ORDER], color="#cfe3f2")
    sns.stripplot(data=dlab, x="domain_label", y="sim_en_pl", ax=ax,
                  order=[DOMAIN_LABELS[d] for d in DOMAIN_ORDER],
                  color="#1f4e79", size=7, jitter=0.18, alpha=0.8)
    ax.set_xlabel(""); ax.set_ylabel("Cosine(EN response, PL response)")
    ax.set_title("Cross-lingual response homogenization by domain")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig1_homogenization.png", dpi=150)
    plt.close(fig)
    captions.append("Figure 1. Cosine similarity between the English and Polish answers to the "
                    "same prompt, per domain (each point = one topic, n=25). Higher = the model "
                    "delivers more nearly identical content regardless of question language.")

    # FIG 2 — sentyment EN vs PL (parowany scatter z linia y=x)
    if "sent_en" in df:
        fig, ax = plt.subplots(figsize=(7.5, 7))
        for d in DOMAIN_ORDER:
            s = dlab[dlab["domain"] == d]
            ax.scatter(s["sent_en"], s["sent_pl"], s=90, alpha=0.85,
                       label=DOMAIN_LABELS[d])
        lim = [-1, 1]
        ax.plot(lim, lim, "--", color="grey", lw=1.5, label="EN = PL")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("Sentiment polarity (EN response)")
        ax.set_ylabel("Sentiment polarity (PL response)")
        ax.set_title("Per-topic sentiment: English vs Polish")
        ax.legend(fontsize=12, loc="lower right")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig2_sentiment.png", dpi=150)
        plt.close(fig)
        captions.append("Figure 2. Sentiment polarity (P(pos)-P(neg)) of each topic's English vs "
                        "Polish answer. Points above the dashed EN=PL line are more positive in "
                        "Polish; below, more positive in English.")

    # FIG 3 — angielska soczewka: native_pull EN vs PL
    if "native_pull_en" in df:
        np_long = df.melt(id_vars=["prompt_id", "domain"],
                          value_vars=["native_pull_en", "native_pull_pl"],
                          var_name="side", value_name="native_pull").dropna()
        np_long["side"] = np_long["side"].map(
            {"native_pull_en": "English answers", "native_pull_pl": "Polish answers"})
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=np_long, x="side", y="native_pull", ax=ax,
                    palette=["#d9d2e9", "#f2cfcf"])
        sns.stripplot(data=np_long, x="side", y="native_pull", ax=ax,
                      color="#333333", size=6, jitter=0.15, alpha=0.7)
        ax.axhline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel(""); ax.set_ylabel("Native pull\n(own-language − other-language wiki sim)")
        ax.set_title("Anchoring to native-language discourse")
        fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig3_native_pull.png", dpi=150)
        plt.close(fig)
        captions.append("Figure 3. 'Native pull' = how much closer an answer is to the Wikipedia "
                        "article in its OWN language than to the article in the other language "
                        "(n=24; s06 has no baseline). If Polish answers show less native pull than "
                        "English answers, Polish output is comparatively less anchored in Polish "
                        "discourse — the English-lens signal.")

    # FIG 4 — dlugosc odpowiedzi (kontrola)
    fig, ax = plt.subplots(figsize=(8, 6))
    wl = df.melt(id_vars=["domain"], value_vars=["words_en", "words_pl"],
                 var_name="lang", value_name="words")
    wl["lang"] = wl["lang"].map({"words_en": "EN", "words_pl": "PL"})
    wl["domain_label"] = wl["domain"].map(DOMAIN_LABELS)
    sns.barplot(data=wl, x="domain_label", y="words", hue="lang", ax=ax,
                order=[DOMAIN_LABELS[d] for d in DOMAIN_ORDER], errorbar="sd")
    ax.set_xlabel(""); ax.set_ylabel("Response length (words)")
    ax.set_title("Response length by language and domain")
    ax.legend(title="Prompt language")
    fig.tight_layout(); fig.savefig(FIGURES_DIR / "fig4_length.png", dpi=150)
    plt.close(fig)
    captions.append("Figure 4. Mean response length (words) by domain and prompt language; error "
                    "bars = SD. Control check that the matched max-token cap kept lengths "
                    "comparable across languages.")

    CAPTIONS_TXT.write_text("\n\n".join(captions), encoding="utf-8")
    print(f"-> zapisano {len(captions)} rysunkow do {FIGURES_DIR} (+ captions.txt)")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main():
    print("Wczytuje i paruje dane ...")
    df = load_paired()
    print(f"  {len(df)} promptow sparowanych "
          f"({df[['wiki_en','wiki_pl']].notna().all(axis=1).sum()} z baseline Wikipedii)")

    print("Licze podobienstwo semantyczne ...")
    df = add_similarity(df)

    print("Licze sentyment ...")
    df = add_sentiment(df)

    df.to_csv(PER_PROMPT_CSV, index=False, encoding="utf-8")
    print(f"-> zapisano metryki per prompt: {PER_PROMPT_CSV}")

    run_stats(df)
    make_figures(df)
    print("\nGotowe. Wyniki w data/ , rysunki w figures/.")


if __name__ == "__main__":
    main()