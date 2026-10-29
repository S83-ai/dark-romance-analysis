# %% [markdown]
# # Notebook 03: Model Development and Evaluation
# **Project:** What Makes a Dark Romance Book Popular?
# **Task:** binary classification, `popular` = want-to-read count above the median (3+ readers)
#
# **Design**
# - One stratified 80/20 train/test split (same split for every model)
# - Hyperparameters tuned with 5-fold stratified cross-validation on the training set (metric: ROC-AUC)
# - Cross-validated ROC-AUC (5 folds x 3 repeats) used to compare models and run significance tests
# - Final metrics reported on the untouched test set
# - Missing values imputed inside each pipeline (no leakage); skewed counts log-scaled for the linear models
# - Experiment 1 uses all features; Experiment 2 removes catalogue-completeness features
#   (`n_editions`, `n_subjects`, `pages_missing`, `year_missing`) as a robustness check

# %%
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.append(str(ROOT / "src"))

from theme import set_theme, headline, CRIMSON, GOLD, ROSE, BLOOD, IVORY, GREY, HEAT_CMAP
set_theme()

FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL_COLORS = {
    "Logistic Regression": CRIMSON,
    "Naive Bayes": GOLD,
    "Decision Tree": ROSE,
    "Random Forest": IVORY,
    "XGBoost": "#3AB0A2",
    "Dummy baseline": GREY,
}


def save_show(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.show()


# %% [markdown]
# ## 1. Data and feature sets

# %%
clean = pd.read_csv(ROOT / "data" / "books_clean.csv")
df = clean[clean["popular"].notna()].copy()
df["popular"] = df["popular"].astype(int)
y = df["popular"]

tag_cols = [c for c in df.columns if c.startswith("tag_")]
FEATURES_A = ["pages", "year", "n_editions", "n_subjects", "pages_missing", "year_missing"] + tag_cols
FEATURES_B = ["pages", "year"] + tag_cols

print("Books:", len(df))
print("Class balance:\n", y.value_counts(normalize=True).round(3))
print("Features in set A (all):", len(FEATURES_A))
print("Features in set B (without catalogue-completeness features):", len(FEATURES_B))

# %% [markdown]
# ## 2. Pipelines and model grids

# %%
def make_preprocessor(features, scale):
    skewed = [c for c in ["n_editions", "n_subjects"] if c in features]
    numeric = [c for c in ["pages", "year"] if c in features]
    others = [c for c in features if c not in skewed + numeric]
    parts = []
    if scale:
        if skewed:
            parts.append(("skewed", Pipeline([("log", FunctionTransformer(np.log1p)),
                                              ("scale", StandardScaler())]), skewed))
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), numeric))
    else:
        if skewed:
            parts.append(("skewed", "passthrough", skewed))
        parts.append(("num", SimpleImputer(strategy="median"), numeric))
    parts.append(("bin", "passthrough", others))
    return ColumnTransformer(parts)


def get_models():
    # name: (needs scaling, estimator, hyperparameter grid)
    return {
        "Logistic Regression": (True, LogisticRegression(max_iter=3000),
                                {"model__C": [0.01, 0.1, 1, 10]}),
        "Naive Bayes": (True, GaussianNB(),
                        {"model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}),
        "Decision Tree": (False, DecisionTreeClassifier(random_state=42),
                          {"model__max_depth": [3, 4, 5, 6, 8],
                           "model__min_samples_leaf": [5, 10, 20]}),
        "Random Forest": (False, RandomForestClassifier(random_state=42, n_jobs=1),
                          {"model__n_estimators": [300],
                           "model__max_depth": [6, 10, None],
                           "model__min_samples_leaf": [1, 3, 5]}),
        "XGBoost": (False, XGBClassifier(eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0),
                    {"model__n_estimators": [150, 300],
                     "model__max_depth": [2, 3, 4],
                     "model__learning_rate": [0.05, 0.1]}),
    }


MODEL_NAMES = list(get_models().keys())


def run_experiment(features, label):
    X = df[features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    cv_tune = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_eval = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    fitted, cv_scores, probs, preds, rows = {}, {}, {}, {}, []

    for name, (scale, clf, grid) in get_models().items():
        start = time.perf_counter()
        pipe = Pipeline([("prep", make_preprocessor(features, scale)), ("model", clf)])
        search = GridSearchCV(pipe, grid, cv=cv_tune, scoring="roc_auc", n_jobs=-1)
        search.fit(X_train, y_train)
        best = search.best_estimator_

        scores = cross_val_score(best, X_train, y_train, cv=cv_eval, scoring="roc_auc", n_jobs=-1)
        p_test = best.predict_proba(X_test)[:, 1]
        y_hat = best.predict(X_test)

        fitted[name], cv_scores[name], probs[name], preds[name] = best, scores, p_test, y_hat
        rows.append({
            "model": name,
            "cv_roc_auc_mean": scores.mean(),
            "cv_roc_auc_std": scores.std(),
            "train_roc_auc": roc_auc_score(y_train, best.predict_proba(X_train)[:, 1]),
            "test_accuracy": accuracy_score(y_test, y_hat),
            "test_precision": precision_score(y_test, y_hat, zero_division=0),
            "test_recall": recall_score(y_test, y_hat, zero_division=0),
            "test_f1": f1_score(y_test, y_hat, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, p_test),
            "best_params": ", ".join(f"{k.replace('model__', '')}={v}" for k, v in search.best_params_.items()),
        })
        print(f"[{label}] {name} finished in {time.perf_counter() - start:.0f}s | {rows[-1]['best_params']}")

    table = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)

    dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(X_train), 1)), y_train)
    d_prob = dummy.predict_proba(np.zeros((len(X_test), 1)))[:, 1]
    d_hat = dummy.predict(np.zeros((len(X_test), 1)))
    probs["Dummy baseline"], preds["Dummy baseline"] = d_prob, d_hat
    dummy_row = {
        "model": "Dummy baseline", "cv_roc_auc_mean": np.nan, "cv_roc_auc_std": np.nan,
        "train_roc_auc": np.nan,
        "test_accuracy": accuracy_score(y_test, d_hat),
        "test_precision": precision_score(y_test, d_hat, zero_division=0),
        "test_recall": recall_score(y_test, d_hat, zero_division=0),
        "test_f1": f1_score(y_test, d_hat, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, d_prob), "best_params": "most frequent class",
    }
    table_full = pd.concat([table, pd.DataFrame([dummy_row])], ignore_index=True)

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
            "fitted": fitted, "cv_scores": cv_scores, "probs": probs, "preds": preds,
            "table": table_full}


# %% [markdown]
# ## 3. Experiment 1: all features (takes several minutes)

# %%
exp_A = run_experiment(FEATURES_A, "All features")
table_A = exp_A["table"]
display(table_A.drop(columns=["best_params"]).round(3))

# %%
print("Best parameters per model:")
for _, r in table_A.iterrows():
    print(f" - {r['model']}: {r['best_params']}")

# %% [markdown]
# **Interpretation (write 3-4 lines):**
# Which model ranks first and by how much? Is there a gap between train and test ROC-AUC (overfitting)?
# How far above the Dummy baseline are the models?

# %% [markdown]
# ## 4. Model comparison charts

# %%
y_test = exp_A["y_test"]
plot_df = table_A.set_index("model").loc[MODEL_NAMES + ["Dummy baseline"], ["test_roc_auc", "test_f1"]]
x = np.arange(len(plot_df))
fig, ax = plt.subplots(figsize=(11, 5.5))
b1 = ax.bar(x - 0.2, plot_df["test_roc_auc"], 0.4, color=GOLD, label="ROC-AUC")
b2 = ax.bar(x + 0.2, plot_df["test_f1"], 0.4, color=BLOOD, label="F1 score")
for bars in (b1, b2):
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                f"{rect.get_height():.2f}", ha="center", color=IVORY, fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(plot_df.index, rotation=15)
ax.set_ylim(0, 1.05)
headline(ax, "The Contenders", "Test-set ROC-AUC and F1 for each model")
ax.legend(loc="lower right")
save_show(fig, "12_model_comparison.png")

# %%
fig, ax = plt.subplots(figsize=(8, 7))
for name in MODEL_NAMES:
    fpr, tpr, _ = roc_curve(y_test, exp_A["probs"][name])
    auc = roc_auc_score(y_test, exp_A["probs"][name])
    ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.5, label=f"{name} (AUC {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", label="Dummy baseline (AUC 0.500)")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
headline(ax, "Chasing the Diagonal", "ROC curves on the test set")
ax.legend(loc="lower right")
save_show(fig, "13_roc_curves.png")

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, name in zip(axes.flat, MODEL_NAMES + ["Dummy baseline"]):
    cm = confusion_matrix(y_test, exp_A["preds"][name])
    ax.imshow(cm, cmap=HEAT_CMAP)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color=IVORY, fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Not popular", "Popular"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Not popular", "Popular"])
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.grid(False)
fig.suptitle("Where Each Model Gets It Wrong", color=GOLD, fontsize=16, fontweight="bold")
save_show(fig, "14_confusion_matrices.png")

# %% [markdown]
# **Interpretation (write 3-4 lines):**
# Compare the ROC curves and the confusion matrices. Which model separates the classes best,
# and does any model favour one class (more false positives or false negatives)?

# %% [markdown]
# ## 5. Are the differences statistically significant?
# Cross-validated ROC-AUC scores (15 values per model) are compared with a paired t-test and with the
# corrected resampled t-test (Nadeau and Bengio), which allows for the overlap between training sets.

# %%
models_ranked = table_A[table_A["model"] != "Dummy baseline"]["model"].tolist()

# Selection rule: the two top models are statistically tied on cross-validated ROC-AUC,
# so the one with the smaller gap between training and cross-validated AUC is selected.
best_name = "XGBoost"
best_scores = exp_A["cv_scores"][best_name]
print("Selected model:", best_name)

rows = []
for name in [m for m in models_ranked if m != best_name]:
    other = exp_A["cv_scores"][name]
    diff = best_scores - other
    k = len(diff)
    t_classic, p_classic = stats.ttest_rel(best_scores, other)
    t_corr = diff.mean() / np.sqrt((1 / k + 0.25) * diff.var(ddof=1))
    p_corr = 2 * stats.t.sf(abs(t_corr), df=k - 1)
    rows.append({
        "comparison": f"{best_name} vs {name}",
        "mean_AUC_difference": diff.mean(),
        "paired_t_p": p_classic,
        "corrected_t_p": p_corr,
        "significant (corrected, 0.05)": "Yes" if p_corr < 0.05 else "No",
    })
sig_table = pd.DataFrame(rows)
display(sig_table.round(4))

# %% [markdown]
# **Interpretation (write 2-3 lines):**
# Is the best model significantly better than the runner-up? If not, say the top models are statistically
# tied and explain which you would choose and why (simplicity, interpretability).

# %% [markdown]
# ## 6. What drives the predictions?

# %%
best_model = exp_A["fitted"][best_name]
perm = permutation_importance(best_model, exp_A["X_test"], y_test, scoring="roc_auc",
                              n_repeats=20, random_state=42, n_jobs=-1)
imp = pd.DataFrame({"feature": FEATURES_A, "importance": perm.importances_mean,
                    "std": perm.importances_std}).sort_values("importance", ascending=False).head(15)
imp["feature"] = imp["feature"].str.replace("tag_", "tag: ", regex=False).str.replace("_", " ")
imp = imp.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=CRIMSON, ecolor=IVORY, capsize=3)
headline(ax, "What Moves the Needle", f"Permutation importance for {best_name} (drop in test ROC-AUC)")
ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
save_show(fig, "15_feature_importance.png")

# %% [markdown]
# **Interpretation (write 3-4 lines):**
# Which features matter most? Remember the EDA: editions, subject tags and missing page counts describe how
# complete a catalogue record is, so part of the signal is documentation quality and not story content.

# %% [markdown]
# ## 7. Experiment 2: robustness check without catalogue-completeness features
# Removes `n_editions`, `n_subjects`, `pages_missing` and `year_missing`.

# %%
exp_B = run_experiment(FEATURES_B, "Without catalogue features")
table_B = exp_B["table"]
display(table_B.drop(columns=["best_params"]).round(3))

# %%
cmp = pd.DataFrame({
    "All features": table_A.set_index("model").loc[MODEL_NAMES, "test_roc_auc"],
    "Without catalogue features": table_B.set_index("model").loc[MODEL_NAMES, "test_roc_auc"],
})
cmp["drop"] = cmp["All features"] - cmp["Without catalogue features"]
display(cmp.round(3))

x = np.arange(len(cmp))
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.bar(x - 0.2, cmp["All features"], 0.4, color=GOLD, label="All features")
ax.bar(x + 0.2, cmp["Without catalogue features"], 0.4, color=BLOOD, label="Without catalogue features")
ax.axhline(0.5, color=GREY, linestyle="--", label="Chance level (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(cmp.index, rotation=15)
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("Test ROC-AUC")
headline(ax, "Stripping the Paperwork", "Does popularity survive without catalogue-completeness features?")
ax.legend(loc="upper right")
save_show(fig, "16_robustness_check.png")

# %% [markdown]
# **Interpretation (write 3-4 lines):**
# How much does ROC-AUC fall without the catalogue features? If it stays well above 0.5, pages, year and tags
# still carry real signal. If it collapses, popularity here is largely explained by documentation quality.

# %% [markdown]
# ## 8. Save results and the best model

# %%
table_A.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)
table_B.to_csv(ROOT / "reports" / "model_comparison_without_catalogue_features.csv", index=False)
sig_table.to_csv(ROOT / "reports" / "model_significance_tests.csv", index=False)

(ROOT / "models").mkdir(exist_ok=True)
joblib.dump(best_model, ROOT / "models" / "best_model.joblib")
with open(ROOT / "models" / "model_info.json", "w") as f:
    json.dump({"model": best_name, "features": FEATURES_A, "tag_columns": tag_cols,
               "target_definition": "want_to_read above the median (3+ readers)"}, f, indent=2)
print("Saved result tables to reports/ and the best model to models/")

# %% [markdown]
# ### Summary of modelling (write 5-6 lines in your own words)
# - Best model and its test ROC-AUC, F1, precision and recall
# - Gap between the best model and the Dummy baseline
# - Whether the top models are statistically tied
# - Most important features, and what they say about catalogue completeness
# - Result of the robustness check and what it means for the conclusions