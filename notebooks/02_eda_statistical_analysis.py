# %% [markdown]
# # Notebook 02: EDA and Statistical Analysis
# **Project:** What Makes a Dark Romance Book Popular?
# **Input:** data/books_clean.csv (from notebook 01)
# **Target:** `popular` = want-to-read count above the median (3+ readers)

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy import stats

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

from theme import set_theme, headline, CRIMSON, GOLD, ROSE, BLOOD, IVORY, GREY, DIVERGING_CMAP
set_theme()

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def save_show(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.show()


# %%
clean = pd.read_csv(ROOT / "data" / "books_clean.csv")
df = clean[clean["popular"].notna()].copy()
df["popular"] = df["popular"].astype(int)
df["label"] = df["popular"].map({0: "Not popular", 1: "Popular"})
df["rating_rel"] = df["rating"].where(df["rating_reliable"] == 1)
df["log_wtr"] = np.log10(df["want_to_read"] + 1)
ORDER = ["Not popular", "Popular"]

print("Books used for analysis:", df.shape[0])
print(df["label"].value_counts())

# %% [markdown]
# **Important note on data leakage:** `want_to_read`, `already_read`, `reading_now`, `readinglog` and `n_ratings`
# are reader-activity counts. `want_to_read` defines the target, and the others move with it, so they are
# **excluded from the model features** in notebook 03. The features used are `pages`, `year`, `n_editions`,
# `n_subjects`, the subject-tag columns and the missing-value flags.

# %% [markdown]
# ## 1. Summary statistics

# %%
num_features = ["pages", "year", "n_editions", "n_subjects"]
summary = df.groupby("label")[num_features].agg(["count", "mean", "median", "std"]).round(2).T
display(summary)
display(df[num_features + ["rating_rel"]].describe().T.round(2))

# %%
rows = []
for c in ["pages", "year", "n_editions", "n_subjects", "rating_rel"]:
    s = df[c].dropna()
    _, p_norm = stats.normaltest(s)
    rows.append({"feature": c, "n": len(s), "skewness": round(float(stats.skew(s)), 2), "normality_p": p_norm})
normality = pd.DataFrame(rows)
display(normality)

# %% [markdown]
# **Interpretation (write 2-3 lines):**
# Which features are skewed and not normal? This is why Mann-Whitney U and Kruskal-Wallis are used below
# alongside the t-test and ANOVA.

# %% [markdown]
# ## 2. Relationships between variables

# %%
corr_cols = ["pages", "year", "n_editions", "n_subjects", "rating_rel", "popular"]
corr = df[corr_cols].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, center=0, vmin=-1, vmax=1,
            linewidths=1, linecolor="#0B0B0F", annot_kws={"color": IVORY}, ax=ax)
headline(ax, "Correlation of Desire", "Spearman correlation between features and popularity")
save_show(fig, "08_correlation_heatmap.png")

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col, title in zip(axes, ["pages", "year", "n_editions"],
                          ["Pages", "Publication year", "Number of editions"]):
    data = [df.loc[df["label"] == lab, col].dropna() for lab in ORDER]
    bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                    medianprops={"color": IVORY, "linewidth": 2},
                    whiskerprops={"color": IVORY}, capprops={"color": IVORY},
                    boxprops={"edgecolor": IVORY})
    for patch, color in zip(bp["boxes"], [BLOOD, GOLD]):
        patch.set_facecolor(color)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(ORDER)
    ax.set_title(title)
fig.suptitle("Popular vs Not Popular", color=GOLD, fontsize=16, fontweight="bold")
save_show(fig, "09_boxplots_by_popularity.png")

# %% [markdown]
# **Interpretation (write 2-3 lines):**
# Which features move with popularity? Does the correlation with publication year support the
# "older books collect more want-to-read adds" limitation?

# %% [markdown]
# ## 3. Hypothesis tests (alpha = 0.05)

# %%
results = []


def log_result(hypothesis, test, stat, p, effect_name, effect):
    results.append({
        "hypothesis": hypothesis,
        "test": test,
        "statistic": round(float(stat), 3),
        "p_value": float(p),
        "effect_size": f"{effect_name} = {effect:.3f}",
        "decision (alpha 0.05)": "Reject H0" if p < 0.05 else "Fail to reject H0",
    })


# H1: popular and not-popular books differ on numeric features (Mann-Whitney U, because of skew)
for col, name in [("pages", "page count"), ("year", "publication year"),
                  ("n_editions", "number of editions"), ("n_subjects", "number of subject tags")]:
    a = df.loc[df["popular"] == 1, col].dropna()
    b = df.loc[df["popular"] == 0, col].dropna()
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    r = 2 * u / (len(a) * len(b)) - 1
    print(f"{name}: median popular = {a.median():.1f}, not popular = {b.median():.1f}, "
          f"U = {u:.0f}, p = {p:.4g}, rank-biserial r = {r:.3f}")
    log_result(f"H0: {name} is the same for popular and not-popular books", "Mann-Whitney U", u, p, "rank-biserial r", r)

# %%
# H2: average rating differs between popular and not-popular books (Welch t-test, books with 5+ ratings)
a = df.loc[(df["popular"] == 1) & df["rating_rel"].notna(), "rating_rel"]
b = df.loc[(df["popular"] == 0) & df["rating_rel"].notna(), "rating_rel"]
t, p = stats.ttest_ind(a, b, equal_var=False)
d = (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
print(f"Mean rating: popular = {a.mean():.2f} (n={len(a)}), not popular = {b.mean():.2f} (n={len(b)})")
print(f"Welch t = {t:.3f}, p = {p:.4g}, Cohen's d = {d:.3f}")
u2, p2 = stats.mannwhitneyu(a, b, alternative="two-sided")
print(f"Robustness check (Mann-Whitney U): p = {p2:.4g}")
log_result("H0: mean rating is the same for popular and not-popular books (5+ ratings)",
           "Welch t-test", t, p, "Cohen's d", d)

# %%
# H3: subject tags and popularity (Chi-square, Bonferroni-adjusted)
def chi_binary(col):
    table = pd.crosstab(df[col], df["popular"])
    if table.shape != (2, 2):
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    v = np.sqrt(chi2 / table.values.sum())
    with_rate = table.loc[1, 1] / table.loc[1].sum()
    without_rate = table.loc[0, 1] / table.loc[0].sum()
    return chi2, p, v, with_rate, without_rate, int(table.loc[1].sum())


tag_rows = []
for c in [c for c in df.columns if c.startswith("tag_")]:
    res = chi_binary(c)
    if res is None:
        continue
    chi2, p, v, wr, wor, n_with = res
    tag_rows.append({"tag": c.replace("tag_", "").replace("_", " "), "books_with_tag": n_with,
                     "popular_rate_with_tag": wr, "popular_rate_without": wor,
                     "chi2": chi2, "p_value": p, "cramers_v": v})

tag_tests = pd.DataFrame(tag_rows).sort_values("p_value").reset_index(drop=True)
tag_tests["p_bonferroni"] = (tag_tests["p_value"] * len(tag_tests)).clip(upper=1)
tag_tests["significant"] = tag_tests["p_bonferroni"] < 0.05
display(tag_tests.round(4))
print("Tags significant after Bonferroni correction:", int(tag_tests["significant"].sum()), "of", len(tag_tests))

best = tag_tests.iloc[0]
log_result(f"H0: the tag '{best['tag']}' is unrelated to popularity (Bonferroni-adjusted p)",
           "Chi-square test", best["chi2"], best["p_bonferroni"], "Cramer's V", best["cramers_v"])

# %%
top = tag_tests.head(10).iloc[::-1]
y = np.arange(len(top))
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(y + 0.2, top["popular_rate_with_tag"] * 100, height=0.4, color=GOLD, label="Books with the tag")
ax.barh(y - 0.2, top["popular_rate_without"] * 100, height=0.4, color=BLOOD, label="Books without the tag")
ax.axvline(df["popular"].mean() * 100, color=IVORY, linestyle="--", linewidth=1.5, label="Overall rate")
ax.set_yticks(y)
ax.set_yticklabels(top["tag"])
ax.set_xlabel("% of books that are popular")
headline(ax, "Tags That Tempt Readers", "Popular rate with vs without each tag (10 smallest p-values)")
ax.legend()
save_show(fig, "10_tag_popularity_rates.png")

# %%
# H4: is missing page count related to popularity? (Chi-square)
res = chi_binary("pages_missing")
if res is not None:
    chi2, p, v, wr, wor, n_with = res
    print(f"Popular rate when pages missing = {wr:.1%}, when pages present = {wor:.1%}")
    print(f"Chi-square = {chi2:.3f}, p = {p:.4g}, Cramer's V = {v:.3f}")
    log_result("H0: a missing page count is unrelated to popularity", "Chi-square test", chi2, p, "Cramer's V", v)

# %%
# H5: want-to-read counts differ across publication periods (ANOVA and Kruskal-Wallis on log counts)
df["period"] = pd.cut(df["year"], bins=[1979, 2009, 2014, 2019, 2026],
                      labels=["Before 2010", "2010-2014", "2015-2019", "2020-2026"])
period_df = df.dropna(subset=["period"])
groups = [g["log_wtr"].values for _, g in period_df.groupby("period", observed=True)]

f, p_f = stats.f_oneway(*groups)
grand = period_df["log_wtr"].mean()
ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
ss_total = ((period_df["log_wtr"] - grand) ** 2).sum()
eta2 = ss_between / ss_total
h, p_kw = stats.kruskal(*groups)
print(f"One-way ANOVA: F = {f:.3f}, p = {p_f:.4g}, eta squared = {eta2:.3f}")
print(f"Kruskal-Wallis: H = {h:.3f}, p = {p_kw:.4g}")
log_result("H0: mean log want-to-read count is equal across the four publication periods",
           "One-way ANOVA", f, p_f, "eta squared", eta2)
log_result("H0: want-to-read counts have the same distribution across the four publication periods",
           "Kruskal-Wallis", h, p_kw, "eta squared (ANOVA)", eta2)

try:
    print(stats.tukey_hsd(*groups))
except Exception as e:
    print("Tukey post-hoc test not available:", e)

# %%
period_stats = period_df.groupby("period", observed=True)["log_wtr"].agg(["mean", "sem", "count"])
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(period_stats.index.astype(str), period_stats["mean"], yerr=period_stats["sem"],
       color=[BLOOD, CRIMSON, ROSE, GOLD][:len(period_stats)], ecolor=IVORY, capsize=6)
headline(ax, "Does Age Buy Attention?", "Mean log10(want-to-read + 1) by publication period, with standard error")
ax.set_ylabel("Mean log10(want-to-read + 1)")
save_show(fig, "11_want_to_read_by_period.png")

# %% [markdown]
# **Interpretation (write 2-3 lines per test):**
# State each hypothesis in words, the p-value, and what it means for the project. For the period test,
# say whether it confirms that publication year must be controlled for or at least named as a limitation.

# %% [markdown]
# ## 4. Results table and summary of findings

# %%
results_df = pd.DataFrame(results)
pd.set_option("display.max_colwidth", 90)
display(results_df)
results_df.to_csv(ROOT / "reports" / "statistical_tests.csv", index=False)
print("Saved to reports/statistical_tests.csv")

# %% [markdown]
# ### Summary of findings (write 4-5 bullet points in your own words)
# - Which features are most strongly linked to popularity?
# - Which subject tags stand out after the Bonferroni correction?
# - Does publication year confound popularity, and how will the models handle it?
# - Features carried into notebook 03: `pages`, `year`, `n_editions`, `n_subjects`, tag columns, missing flags.
# - Excluded to avoid leakage: `want_to_read`, `already_read`, `reading_now`, `readinglog`, `n_ratings`.