import time

import pandas as pd
import requests

QUERIES = ["mafia romance", "dark romance", "enemies to lovers mafia", "arranged marriage mafia", "bratva romance"]
URL = "https://openlibrary.org/search.json"
HEADERS = {"User-Agent": "dark-romance-analysis (student project)"}
FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year", "number_of_pages_median",
    "ratings_average", "ratings_count", "want_to_read_count", "already_read_count",
    "currently_reading_count", "readinglog_count", "subject",
])

rows = []
for q in QUERIES:
    for page in range(1, 4):
        params = {"q": q, "fields": FIELDS, "limit": 100, "page": page}
        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as e:
            print("Request failed:", e)
            continue
        if r.status_code != 200:
            print("Error", r.status_code, "for", q, page)
            continue
        for d in r.json().get("docs", []):
            d["query"] = q
            rows.append(d)
        time.sleep(2)

if not rows:
    print("No books collected. Check the error messages above.")
    raise SystemExit

df = pd.DataFrame(rows)
for col in ["key", "ratings_average", "ratings_count", "want_to_read_count",
            "number_of_pages_median", "first_publish_year", "subject"]:
    if col not in df.columns:
        df[col] = None
df = df.drop_duplicates(subset="key")

print("Total unique books:", len(df))
print("With rating:", df["ratings_average"].notna().sum())
print("With 5+ ratings:", (df["ratings_count"].fillna(0) >= 5).sum())
print("With want-to-read count:", df["want_to_read_count"].notna().sum())
print("With page count:", df["number_of_pages_median"].notna().sum())
print("With subjects:", df["subject"].notna().sum())
df.to_csv("data/openlibrary_check.csv", index=False)