"""
Narrative DNA of The Great Gatsby 
"""
 
import re
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt

# -
URL = "https://www.gutenberg.org/cache/epub/64317/pg64317-images.html"
headers = {"User-Agent": "AMU-Research-Bot/1.0 (researcher@amu.edu.pl)"}
 
print(f"Agent dispatched to: {URL} ...")
response = requests.get(URL, headers=headers, timeout=60) 
if response.status_code != 200:
    raise SystemExit(f"Request failed with status {response.status_code}")
print("Success! The server opened the vault.")
time.sleep(1)  
 
soup = BeautifulSoup(response.text, "html.parser")
full_text = soup.get_text(separator="\n")
 
# -
START = "*** START OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"
END   = "*** END OF THE PROJECT GUTENBERG EBOOK THE GREAT GATSBY ***"
narrative = full_text[full_text.find(START) + len(START): full_text.find(END)]
 
chapter_re = re.compile(r"(?m)^\s*(IX|VIII|VII|VI|V|IV|III|II|I)\s*$")
parts = chapter_re.split(narrative)
bodies = parts[2::2]                                
chapters = [b for b in bodies if len(b.strip()) > 1000]  
 
print(f"Chapters detected: {len(chapters)}")
assert len(chapters) == 9, "Expected 9 chapters"

# -

NAME_RE = re.compile(r'(?<![.!?:"\u201c\u201d])\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')


canonical_mapping = [
    ("Gatsby", "Jay Gatsby"),   ("Jay", "Jay Gatsby"),
    ("Buchanan", "Tom Buchanan"), ("Tom", "Tom Buchanan"),
    ("Daisy", "Daisy Fay"),
    ("Jordan", "Jordan Baker"), ("Baker", "Jordan Baker"),
    ("Carraway", "Nick Carraway"), ("Nick", "Nick Carraway"),
    ("Myrtle", "Myrtle Wilson"),
    ("George", "George Wilson"), ("Wilson", "George Wilson"),
    ("Wolfshiem", "Meyer Wolfshiem"), ("Meyer", "Meyer Wolfshiem"),
    ("New York", "New York"), ("West Egg", "West Egg"), ("East Egg", "East Egg"),
    ("Long Island", "Long Island"), ("Chicago", "Chicago"),
]

def resolve(entity):
    for keyword, canonical in canonical_mapping:
        if re.search(r"\b" + re.escape(keyword) + r"\b", entity):
            return canonical
    return None   
rows = []
for ch_num, body in enumerate(chapters, start=1):

    text = re.sub(r"\s+", " ", body)

    counts = {}
    for phrase in NAME_RE.findall(text):
        name = resolve(phrase)
        if name is None:
            continue
        counts[name] = counts.get(name, 0) + 1

    for name, freq in counts.items():
        rows.append({"Chapter": ch_num, "Name": name, "Frequency": freq})

df = pd.DataFrame(rows)

# -
df = df.groupby(["Chapter", "Name"], as_index=False)["Frequency"].sum()
 
# -
df = df.sort_values(["Chapter", "Frequency"], ascending=[True, False]).reset_index(drop=True)
df.to_csv("Kasia_DNAscores.csv", index=False)
print("Saved Kasia_DNAscores.csv")
print(df.head(15).to_string(index=False))
 
# -
top10 = df.groupby("Name")["Frequency"].sum().nlargest(10).index.tolist()
 
pivot = (df[df["Name"].isin(top10)]
         .pivot_table(index="Name", columns="Chapter", values="Frequency", fill_value=0)
         .reindex(columns=range(1, 10), fill_value=0))
 
fig, ax = plt.subplots(figsize=(13, 8))
cmap = plt.get_cmap("tab10")
 
for i, name in enumerate(top10):
    ax.plot(pivot.columns, pivot.loc[name],
            marker="o", linewidth=2, color=cmap(i % 10), label=name)
 
for ch in range(1, 10):
    sub = df[(df["Chapter"] == ch) & (df["Name"].isin(top10))]
    for _, r in sub.nlargest(5, "Frequency").iterrows():
        
        label_text = r["Name"].split()[-1]
        ax.annotate(label_text, (ch, r["Frequency"]),
                    fontsize=8, fontweight="bold",
                    xytext=(0, 6), textcoords="offset points", ha="center")
 
ax.set_title("Narrative DNA of The Great Gatsby - Top 10 across 9 Chapters")
ax.set_xlabel("Chapter")
ax.set_ylabel("Frequency")
ax.set_xticks(range(1, 10))
ax.grid(axis="y", linestyle="--", alpha=0.4)
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("Kasia_DNAplot.png", dpi=150)
print("Saved Kasia_DNAplot.png")
plt.show()