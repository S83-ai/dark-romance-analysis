# %% [markdown]
# # Notebook 01: Data Cleaning and Visualization
# **Project:** What Makes a Dark Romance Book Popular?
# **Source:** Open Library search API (raw file: data/books_raw.csv)
# **Target:** `popular` = want-to-read count above the median

# %%
import re
import sys
from collections import Counter
from pathlib import Path
from IPython.display import display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

from theme import set_theme, headline, CRIMSON, GOLD, ROSE, BLOOD, IVORY, GREY
set_theme()

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save_show(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.show()


# %% [markdown]
# ## 1. Load and inspect the raw data

# %%
raw = pd.read_csv(ROOT / "data" / "books_raw.csv")
print("Shape:", raw.shape)
display(raw.head())
display(raw.describe().T)

# %%
missing = (raw.isna().mean() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(missing.index[::-1], missing.values[::-1], color=CRIMSON)
headline(ax, "What the Raw Data Is Hiding", "Percentage of missing values per column")
ax.set_xlabel("% missing")
save_show(fig, "01_missing_values.png")

# %% [markdown]
# **Interpretation (write 2-3 lines in your own words):**
# Which columns have the most missing values, and why might that be the case for self-published or newer books?

# %% [markdown]
# ## 2. Cleaning

# %%
df = raw.copy()
print("Rows before cleaning:", len(df))

# 2.1 Remove duplicate records (same key, then same title and first author)
df = df.drop_duplicates(subset="key")
df["title_clean"] = df["title"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.strip()
df["author_first"] = df["author_name"].fillna("").str.split("|").str[0].str.lower().str.strip()
df = df.sort_values("readinglog_count", ascending=False)
df = df.drop_duplicates(subset=["title_clean", "author_first"], keep="first")
print("Rows after removing duplicates:", len(df))

# %%
# 2.2 Fix data types
num_cols = ["first_publish_year", "number_of_pages_median", "ratings_average", "ratings_count",
            "want_to_read_count", "already_read_count", "currently_reading_count",
            "readinglog_count", "edition_count"]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 2.3 Handle invalid values (set to missing, impute later inside the modelling pipeline)
bad_pages = df["number_of_pages_median"].notna() & ~df["number_of_pages_median"].between(30, 1500)
print("Invalid page counts set to missing:", bad_pages.sum())
df.loc[bad_pages, "number_of_pages_median"] = np.nan

bad_year = df["first_publish_year"].notna() & ~df["first_publish_year"].between(1980, 2026)
print("Out-of-range years set to missing:", bad_year.sum())
df.loc[bad_year, "first_publish_year"] = np.nan

# %%
# 2.4 Rename columns for readability
df = df.rename(columns={
    "number_of_pages_median": "pages",
    "first_publish_year": "year",
    "ratings_average": "rating",
    "ratings_count": "n_ratings",
    "want_to_read_count": "want_to_read",
    "already_read_count": "already_read",
    "currently_reading_count": "reading_now",
    "readinglog_count": "readinglog",
    "edition_count": "n_editions",
})

# 2.5 Missing-value flags and a reliability flag for ratings
df["pages_missing"] = df["pages"].isna().astype(int)
df["year_missing"] = df["year"].isna().astype(int)
df["rating_reliable"] = (df["n_ratings"].fillna(0) >= 5).astype(int)
print("Books with a reliable rating (5+ ratings):", df["rating_reliable"].sum())

# %%
# 2.6 Turn subject tags into features
df["subject"] = df["subject"].fillna("")
df["subject_list"] = df["subject"].apply(lambda s: [t.strip().lower() for t in s.split("|") if t.strip()])
df["n_subjects"] = df["subject_list"].apply(len)

tag_counts = Counter(t for tags in df["subject_list"] for t in set(tags))
top_tags = [t for t, _ in tag_counts.most_common(25)]
for t in top_tags:
    col = "tag_" + re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    df[col] = df["subject_list"].apply(lambda tags, t=t: int(t in tags))
print("Top 25 tags turned into yes/no columns:", len(top_tags))

# %%
# 2.7 Create the target: popular = want-to-read count above the median
# Books with no want-to-read count are NOT assumed to be zero; they get no target and are excluded from modelling.
median_wtr = df["want_to_read"].median()
df["popular"] = np.where(df["want_to_read"].notna(), (df["want_to_read"] > median_wtr).astype(float), np.nan)
print("Median want-to-read count:", median_wtr)
print("Books usable for modelling:", df["popular"].notna().sum())
print(df["popular"].value_counts(normalize=True).round(3))

# %% [markdown]
# **Interpretation (write 2-3 lines):**
# Explain each cleaning step and why you made that choice (duplicates, invalid values, missing-value flags, target definition).

# %% [markdown]
# ## 3. Visualization

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(np.log10(df["want_to_read"].dropna() + 1), bins=40, color=CRIMSON, edgecolor=BLACK if False else "#0B0B0F")
ax.axvline(np.log10(median_wtr + 1), color=GOLD, linestyle="--", linewidth=2, label=f"Median = {median_wtr:.0f}")
headline(ax, "Who Gets Wanted?", "Distribution of want-to-read counts (log scale)")
ax.set_xlabel("log10(want-to-read count + 1)")
ax.set_ylabel("Number of books")
ax.legend()
save_show(fig, "02_want_to_read_distribution.png")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["pages"].dropna(), bins=40, color=GOLD, edgecolor="#0B0B0F")
headline(ax, "How Long Is a Dark Romance?", "Distribution of page counts")
ax.set_xlabel("Pages")
ax.set_ylabel("Number of books")
save_show(fig, "03_pages_distribution.png")

# %%
yearly = df.dropna(subset=["year"]).groupby("year").size()
yearly = yearly[yearly.index >= 2000]
fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(yearly.index, yearly.values, color=BLOOD, alpha=0.8)
ax.plot(yearly.index, yearly.values, color=ROSE, linewidth=2.5)
headline(ax, "The Rise of the Genre", "Books in the dataset by first publication year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of books")
save_show(fig, "04_publication_trend.png")

# %%
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df.loc[df["rating_reliable"] == 1, "rating"], bins=20, color=ROSE, edgecolor="#0B0B0F")
headline(ax, "What Readers Think", "Average rating of books with 5+ ratings")
ax.set_xlabel("Average rating")
ax.set_ylabel("Number of books")
save_show(fig, "05_rating_distribution.png")

# %%
top15 = pd.Series(tag_counts).sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top15.index[::-1], top15.values[::-1], color=GOLD)
headline(ax, "The Language of the Genre", "15 most common subject tags")
ax.set_xlabel("Number of books")
save_show(fig, "06_top_subject_tags.png")

# %%
model_view = df.dropna(subset=["popular", "year"]).copy()
model_view["period"] = pd.cut(model_view["year"], bins=[1979, 2009, 2014, 2019, 2026],
                              labels=["Before 2010", "2010-2014", "2015-2019", "2020-2026"])
share = model_view.groupby("period", observed=True)["popular"].mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(share.index.astype(str), share.values, color=[BLOOD, CRIMSON, ROSE, GOLD][:len(share)])
headline(ax, "Does Age Buy Popularity?", "Share of popular books by publication period")
ax.set_ylabel("% of books marked popular")
save_show(fig, "07_popular_share_by_period.png")

# %% [markdown]
# **Interpretation (write 2-3 lines per chart):**
# Describe the shape of each distribution, the trend in publication year, and whether older books are more often popular (this is the limitation to state in the report).

# %% [markdown]
# ## 4. Save the cleaned dataset

# %%
out = df.drop(columns=["subject_list", "title_clean", "author_first"])
out.to_csv(ROOT / "data" / "books_clean.csv", index=False)
print("Clean dataset shape:", out.shape)
print("Saved to data/books_clean.csv") 