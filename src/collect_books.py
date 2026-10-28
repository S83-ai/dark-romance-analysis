import time

import pandas as pd
import requests

QUERIES = [
    "mafia romance", "dark romance", "enemies to lovers mafia", "arranged marriage mafia",
    "bratva romance", "mafia boss romance", "italian mafia romance", "russian mafia romance",
    "cartel romance", "billionaire mafia", "forced marriage romance", "marriage of convenience mafia",
    "dark romance revenge", "captive romance", "possessive alpha romance", "crime family romance",
    "mob boss romance", "dark mafia romance series", "irish mafia romance", "bratva bride",
    "mafia princess", "anti hero romance", "morally grey romance", "kidnapped by mafia",
    "mafia wedding", "underworld romance", "gangster romance", "mafia hitman romance",
    "dark billionaire romance", "organized crime romance",
]
PAGES_PER_QUERY = 4
URL = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "dark-romance-analysis (student project)"}
FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year", "number_of_pages_median",
    "ratings_average", "ratings_count", "want_to_read_count", "already_read_count",
    "currently_reading_count", "readinglog_count", "edition_count", "subject",
])


def fetch(params, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            print("  Request failed:", e)
            time.sleep(5)
            continue
        if r.status_code == 200:
            return r.json()
        print("  Error", r.status_code, "- retrying")
        time.sleep(5 * (attempt + 1))
    return None


rows = []
for i, q in enumerate(QUERIES, start=1):
    for page in range(1, PAGES_PER_QUERY + 1):
        params = {"q": q, "fields": FIELDS, "limit": 100, "page": page}
        data = fetch(params)
        if data is None:
            continue
        docs = data.get("docs", [])
        if not docs:
            break
        for d in docs:
            d["query"] = q
            rows.append(d)
        time.sleep(2)
    print(f"[{i}/{len(QUERIES)}] '{q}' done. Rows so far: {len(rows)}")

if not rows:
    print("No books collected. Check the messages above.")
    raise SystemExit

df = pd.DataFrame(rows)

for col in ["author_name", "subject"]:
    if col in df.columns:
        df[col] = df[col].apply(lambda v: "|".join(v) if isinstance(v, list) else v)

df = df.drop_duplicates(subset="key")

print()
print("Total unique books:", len(df))
print("With rating:", df["ratings_average"].notna().sum())
print("With want-to-read count:", df["want_to_read_count"].notna().sum())
print("With page count:", df["number_of_pages_median"].notna().sum())
print("With subjects:", df["subject"].notna().sum())
df.to_csv("data/books_raw.csv", index=False)
print("Saved to data/books_raw.csv")