import os
import time

import pandas as pd
import requests

API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")
QUERIES = ["mafia romance", "dark romance", "enemies to lovers mafia", "arranged marriage mafia"]
URL = "https://www.googleapis.com/books/v1/volumes"

if not API_KEY:
    print("API key not found. Set GOOGLE_BOOKS_API_KEY in this terminal first.")
    raise SystemExit


def fetch(params, retries=4):
    for attempt in range(retries):
        r = requests.get(URL, params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = 10 * (attempt + 1)
            print("Rate limited (429). Waiting", wait, "seconds...")
            time.sleep(wait)
            continue
        print("Error", r.status_code, r.text[:200])
        return None
    return None


rows = []
for q in QUERIES:
    for start in range(0, 120, 40):
        params = {
            "q": q,
            "maxResults": 40,
            "startIndex": start,
            "printType": "books",
            "langRestrict": "en",
            "key": API_KEY,
        }
        data = fetch(params)
        if data is None:
            continue
        for item in data.get("items", []):
            info = item.get("volumeInfo", {})
            rows.append({
                "id": item.get("id"),
                "title": info.get("title"),
                "authors": ", ".join(info.get("authors", [])),
                "published": info.get("publishedDate"),
                "pages": info.get("pageCount"),
                "categories": ", ".join(info.get("categories", [])),
                "avg_rating": info.get("averageRating"),
                "ratings_count": info.get("ratingsCount"),
                "description": info.get("description"),
                "query": q,
            })
        time.sleep(2)

if not rows:
    print("No books collected. Check the error messages above.")
    raise SystemExit

df = pd.DataFrame(rows).drop_duplicates(subset="id")
print("Total unique books:", len(df))
print("With rating:", df["avg_rating"].notna().sum())
print("With description:", df["description"].notna().sum())
print("With page count:", df["pages"].notna().sum())
df.to_csv("data/coverage_check.csv", index=False)