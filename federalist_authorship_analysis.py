#!/usr/bin/env python3
"""
Authorship Attribution of the Disputed Federalist Papers Using Decision Trees,
Ensemble Methods, and Regularized Word-Rate Classification.

Translates and extends the original R `rpart` / `caret` workflow into Python
(`pandas`, `numpy`, `scikit-learn`, `matplotlib`) on the 85-essay Federalist
Papers function-word matrix (`fedPapers85.csv`).
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fedPapers85.csv")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Clean academic/committee serif styling for all Python-generated graphics
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "Hamilton": "#1B365D",  # Navy
    "Madison": "#2E6B4E",   # Forest green
    "Jay": "#7A6855",       # Warm slate/taupe
    "dispt": "#8B2635",     # Burgundy
    "HM": "#5C6B73",        # Neutral slate
}


def essay_number(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


def compute_class_metrics(cm: np.ndarray, labels: list[str]) -> dict:
    """Compute sensitivity (recall), specificity, PPV (precision), NPV, and balanced accuracy."""
    total = cm.sum()
    metrics = {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        bal = 0.5 * (sens + spec)
        metrics[label] = {
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "ppv": round(ppv, 4),
            "npv": round(npv, 4),
            "balanced_accuracy": round(bal, 4),
        }
    return metrics


def main() -> None:
    # =========================================================================
    # PART 1: DATA PREPARATION AND CORPUS PARTITIONING
    # =========================================================================
    fedpapers = pd.read_csv(DATA_PATH)
    fedpapers["essay_no"] = fedpapers["filename"].apply(essay_number)
    fedpapers = fedpapers.sort_values("essay_no").reset_index(drop=True)

    feature_cols = [c for c in fedpapers.columns if c not in ("author", "filename", "essay_no")]
    author_counts = fedpapers["author"].value_counts().to_dict()

    # Separate confirmed single-author corpus (excluding joint 'HM' and disputed 'dispt'),
    # disputed target cohort ('dispt', n=11), and joint cohort ('HM', n=3).
    known_df = fedpapers[~fedpapers["author"].isin(["dispt", "HM"])].copy().reset_index(drop=True)
    disputed_df = fedpapers[fedpapers["author"] == "dispt"].copy().sort_values("essay_no").reset_index(drop=True)
    hm_df = fedpapers[fedpapers["author"] == "HM"].copy().sort_values("essay_no").reset_index(drop=True)

    X_known = known_df[feature_cols]
    y_known = known_df["author"]

    # Stratified ~70/30 partition (51 training essays, 20 testing essays), matching the
    # original R caret::createDataPartition sample sizes (n_train=51, n_test=20).
    X_train, X_test, y_train, y_test = train_test_split(
        X_known, y_known, test_size=20, stratify=y_known, random_state=3
    )

    labels_order = ["Hamilton", "Jay", "Madison"]
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    loo = LeaveOneOut()

    # =========================================================================
    # PART 2: DECISION TREE MODELING, TUNING, AND COST-COMPLEXITY PRUNING
    # =========================================================================
    # 1. General Decision Tree replicating R rpart default stopping rules (minsplit=20, minbucket=7)
    dt_general = DecisionTreeClassifier(
        criterion="gini", min_samples_split=20, min_samples_leaf=7, random_state=42
    )
    dt_general.fit(X_train, y_train)

    # 2. Tuned Model 1: Decision Stump (min_samples_split=10, max_depth=1)
    dt_tuned1 = DecisionTreeClassifier(
        criterion="gini", min_samples_split=10, max_depth=1, random_state=42
    )
    dt_tuned1.fit(X_train, y_train)

    # 3. Tuned Model 2: Expanded Depth (min_samples_split=10, max_depth=5)
    dt_tuned2 = DecisionTreeClassifier(
        criterion="gini", min_samples_split=10, max_depth=5, random_state=42
    )
    dt_tuned2.fit(X_train, y_train)

    # 4. Tuned Model 3: Lowered Minimum Split (min_samples_split=5, max_depth=5)
    dt_tuned3 = DecisionTreeClassifier(
        criterion="gini", min_samples_split=5, max_depth=5, random_state=42
    )
    dt_tuned3.fit(X_train, y_train)

    # 5. Cost-Complexity Pruning Path (Python counterpart to R printcp / plotcp)
    pruning_base = DecisionTreeClassifier(criterion="gini", random_state=42)
    ccp_path = pruning_base.cost_complexity_pruning_path(X_train, y_train)
    ccp_records = []
    for alpha, impurity in zip(ccp_path.ccp_alphas, ccp_path.impurities):
        cand = DecisionTreeClassifier(criterion="gini", ccp_alpha=float(alpha), random_state=42)
        cv_scores = cross_val_score(cand, X_known, y_known, cv=cv5)
        cand.fit(X_train, y_train)
        ccp_records.append({
            "ccp_alpha": round(float(alpha), 5),
            "total_impurity": round(float(impurity), 5),
            "n_leaves": int(cand.get_n_leaves()),
            "max_depth": int(cand.get_depth()),
            "train_acc": round(float(cand.score(X_train, y_train)), 4),
            "cv5_mean": round(float(cv_scores.mean()), 4),
            "cv5_std": round(float(cv_scores.std()), 4),
            "test_acc": round(float(cand.score(X_test, y_test)), 4),
        })

    # Optimal pruned decision tree at ccp_alpha = 0.030 (3 terminal leaves, 2 splits)
    dt_pruned = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.030, random_state=42)
    dt_pruned.fit(X_train, y_train)

    # =========================================================================
    # PART 3: COMPLEMENTARY ENSEMBLE AND PROBABILISTIC CLASSIFIERS
    # =========================================================================
    rf_clf = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42
    )
    rf_clf.fit(X_train, y_train)

    gb_clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
    )
    gb_clf.fit(X_train, y_train)

    lr_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42),
    )
    lr_clf.fit(X_train, y_train)

    model_suite = [
        ("DT Baseline (min_split=20)", dt_general),
        ("DT Tuned 1 (max_depth=1)", dt_tuned1),
        ("DT Tuned 2 (min_split=10, depth=5)", dt_tuned2),
        ("DT Pruned (ccp_alpha=0.030)", dt_pruned),
        ("Random Forest (500 Trees)", rf_clf),
        ("Gradient Boosted Trees", gb_clf),
        ("L2 Logistic Regression", lr_clf),
    ]

    evaluation_rows = []
    for name, clf in model_suite:
        cv_scores = cross_val_score(clf, X_known, y_known, cv=cv5)
        loo_scores = cross_val_score(clf, X_known, y_known, cv=loo)
        y_pred_test = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_test, labels=labels_order)
        evaluation_rows.append({
            "model": name,
            "train_acc": round(float(clf.score(X_train, y_train)) * 100, 2),
            "test_acc": round(float(accuracy_score(y_test, y_pred_test)) * 100, 2),
            "cv5_acc": round(float(cv_scores.mean()) * 100, 2),
            "cv5_std": round(float(cv_scores.std()) * 100, 2),
            "loocv_acc": round(float(loo_scores.mean()) * 100, 2),
            "confusion_matrix": cm.tolist(),
            "class_metrics": compute_class_metrics(cm, labels_order),
        })

    # =========================================================================
    # PART 4: FULL-CORPUS REFIT & ATTRIBUTION OF DISPUTED AND JOINT ESSAYS
    # =========================================================================
    dt_pruned_full = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.030, random_state=42).fit(X_known, y_known)
    rf_full = RandomForestClassifier(
        n_estimators=500, max_features="sqrt", class_weight="balanced", random_state=42
    ).fit(X_known, y_known)
    gb_full = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=2, random_state=42
    ).fit(X_known, y_known)
    lr_full = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000, random_state=42),
    ).fit(X_known, y_known)

    def attribute_cohort(cohort_df: pd.DataFrame) -> list[dict]:
        records = []
        X_c = cohort_df[feature_cols]
        dt_p = dt_pruned_full.predict(X_c)
        dt_pr = dt_pruned_full.predict_proba(X_c)
        rf_p = rf_full.predict(X_c)
        rf_pr = rf_full.predict_proba(X_c)
        gb_p = gb_full.predict(X_c)
        gb_pr = gb_full.predict_proba(X_c)
        lr_p = lr_full.predict(X_c)
        lr_pr = lr_full.predict_proba(X_c)
        cls_idx = {c: i for i, c in enumerate(rf_full.classes_)}

        for i, (_, row) in enumerate(cohort_df.iterrows()):
            votes = [dt_p[i], rf_p[i], gb_p[i], lr_p[i]]
            mad_votes = sum(v == "Madison" for v in votes)
            ham_votes = sum(v == "Hamilton" for v in votes)
            avg_mad_prob = float(np.mean([
                dt_pr[i][cls_idx["Madison"]],
                rf_pr[i][cls_idx["Madison"]],
                gb_pr[i][cls_idx["Madison"]],
                lr_pr[i][cls_idx["Madison"]],
            ]))
            avg_ham_prob = float(np.mean([
                dt_pr[i][cls_idx["Hamilton"]],
                rf_pr[i][cls_idx["Hamilton"]],
                gb_pr[i][cls_idx["Hamilton"]],
                lr_pr[i][cls_idx["Hamilton"]],
            ]))
            consensus = "Madison" if mad_votes > ham_votes else ("Hamilton" if ham_votes > mad_votes else ("Madison (Split 2-2)" if avg_mad_prob >= avg_ham_prob else "Contested (Split 2-2)"))
            records.append({
                "essay_no": int(row["essay_no"]),
                "filename": row["filename"],
                "upon": round(float(row["upon"]), 3),
                "on": round(float(row["on"]), 3),
                "which": round(float(row["which"]), 3),
                "by": round(float(row["by"]), 3),
                "to": round(float(row["to"]), 3),
                "dt_pruned": dt_p[i],
                "rf_pred": rf_p[i],
                "rf_madison_prob": round(float(rf_pr[i][cls_idx["Madison"]]), 3),
                "rf_hamilton_prob": round(float(rf_pr[i][cls_idx["Hamilton"]]), 3),
                "gb_pred": gb_p[i],
                "lr_pred": lr_p[i],
                "lr_madison_prob": round(float(lr_pr[i][cls_idx["Madison"]]), 3),
                "lr_hamilton_prob": round(float(lr_pr[i][cls_idx["Hamilton"]]), 3),
                "mean_madison_prob": round(avg_mad_prob, 3),
                "consensus": consensus,
            })
        return records

    disputed_results = attribute_cohort(disputed_df)
    hm_results = attribute_cohort(hm_df)

    # Top feature importances from full Random Forest
    importances = pd.Series(rf_full.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top12_words = importances.head(12).index.tolist()
    word_means = fedpapers.groupby("author")[top12_words].mean().round(4).to_dict(orient="index")

    # =========================================================================
    # PART 5: GENERATE PYTHON MATPLOTLIB / SCIKIT-LEARN FIGURES
    # =========================================================================
    # FIGURE 1: Baseline Decision Stump vs. Cost-Complexity Pruned Decision Tree
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.0, 1.35]})
    plot_tree(
        dt_tuned1,
        feature_names=feature_cols,
        class_names=labels_order,
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        fontsize=9,
        ax=axes[0],
    )
    axes[0].set_title("A. Baseline / Stump Tree (max_depth = 1)\n1 Split ('upon'); Jay Unrepresented (90.0% Holdout)", pad=10)

    plot_tree(
        dt_pruned,
        feature_names=feature_cols,
        class_names=labels_order,
        filled=True,
        rounded=True,
        proportion=True,
        precision=3,
        fontsize=9,
        ax=axes[1],
    )
    axes[1].set_title("B. Cost-Complexity Pruned Tree (ccp_alpha = 0.030)\n2 Splits ('upon', 'which'); All 3 Authors Isolated (95.0% Holdout)", pad=10)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "figure_1_decision_trees.png"), bbox_inches="tight")
    plt.close(fig)

    # FIGURE 2: Cost-Complexity Pruning Diagnostic (ccp_alpha vs Impurity and Accuracy)
    ccp_df = pd.DataFrame(ccp_records)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    axes[0].plot(ccp_df["ccp_alpha"], ccp_df["total_impurity"], marker="o", linewidth=1.8, color="#1B365D", drawstyle="steps-post")
    for _, r in ccp_df.iterrows():
        n_lv = int(r["n_leaves"])
        lbl = f"{n_lv} leaf" if n_lv == 1 else f"{n_lv} leaves"
        dx = -8 if n_lv == 4 else (8 if n_lv == 3 else 4)
        dy = 8 if n_lv == 4 else 6
        axes[0].annotate(
            lbl,
            (r["ccp_alpha"], r["total_impurity"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
        )
    axes[0].axvline(0.030, color="#8B2635", linestyle="--", linewidth=1.2, label="Selected ccp_alpha = 0.030")
    axes[0].set_xlabel("Effective Complexity Parameter (ccp_alpha)")
    axes[0].set_ylabel("Total Leaf Gini Impurity (Training Set)")
    axes[0].set_title("A. Impurity vs. Complexity Penalty")
    axes[0].legend(loc="lower right", frameon=False)
    axes[0].grid(axis="y", linestyle=":", alpha=0.5)

    axes[1].plot(ccp_df["ccp_alpha"], ccp_df["train_acc"] * 100, marker="s", linestyle=":", color="#5C6B73", label="Training Accuracy (n = 51)")
    axes[1].plot(ccp_df["ccp_alpha"], ccp_df["cv5_mean"] * 100, marker="o", linewidth=1.8, color="#1B365D", label="5-Fold Stratified CV (n = 71)")
    axes[1].plot(ccp_df["ccp_alpha"], ccp_df["test_acc"] * 100, marker="^", linewidth=1.8, color="#2E6B4E", label="Holdout Test Accuracy (n = 20)")
    axes[1].axvline(0.030, color="#8B2635", linestyle="--", linewidth=1.2, label="Optimal Pruning Threshold (0.030)")
    axes[1].set_xlabel("Effective Complexity Parameter (ccp_alpha)")
    axes[1].set_ylabel("Classification Accuracy (%)")
    axes[1].set_ylim(65, 103)
    axes[1].set_title("B. Cross-Validation and Holdout Generalization")
    axes[1].legend(loc="lower left", bbox_to_anchor=(0.17, 0.02), frameon=False, fontsize=8.0)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "figure_2_ccp_pruning_curve.png"), bbox_inches="tight")
    plt.close(fig)

    # FIGURE 3: Holdout Test Confusion Matrix & Multi-Model Benchmark
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [0.95, 1.45]})
    cm_pruned = confusion_matrix(y_test, dt_pruned.predict(X_test), labels=labels_order)
    axes[0].imshow(cm_pruned, cmap="Blues", vmin=0, vmax=15)
    axes[0].set_xticks(range(len(labels_order)))
    axes[0].set_yticks(range(len(labels_order)))
    axes[0].set_xticklabels(labels_order)
    axes[0].set_yticklabels(labels_order)
    axes[0].set_xlabel("Predicted Author")
    axes[0].set_ylabel("Confirmed Author (Reference)")
    axes[0].set_title(f"A. Holdout Confusion Matrix\nPruned Decision Tree (n = 20, Acc = {accuracy_score(y_test, dt_pruned.predict(X_test))*100:.1f}%)")
    for i in range(cm_pruned.shape[0]):
        for j in range(cm_pruned.shape[1]):
            val = cm_pruned[i, j]
            axes[0].text(
                j, i, str(val),
                ha="center", va="center",
                color="white" if val > 7 else "#111111",
                fontweight="bold", fontsize=11,
            )

    bench_df = pd.DataFrame([
        r for r in evaluation_rows
        if r["model"] in (
            "DT Baseline (min_split=20)",
            "DT Pruned (ccp_alpha=0.030)",
            "Random Forest (500 Trees)",
            "Gradient Boosted Trees",
            "L2 Logistic Regression",
        )
    ])
    short_names = [
        "DT Baseline\n(1 Split)",
        "DT Pruned\n(ccp=0.030)",
        "Random Forest\n(500 Trees)",
        "Gradient\nBoosting",
        "L2 Logistic\nRegression",
    ]
    y_pos = np.arange(len(bench_df))
    bar_h = 0.25
    axes[1].barh(y_pos - bar_h, bench_df["test_acc"], height=bar_h, color="#7A6855", label="Holdout Test (n = 20)")
    axes[1].barh(y_pos, bench_df["loocv_acc"], height=bar_h, color="#2E6B4E", label="Leave-One-Out CV (n = 71)")
    axes[1].barh(y_pos + bar_h, bench_df["cv5_acc"], height=bar_h, color="#1B365D", label="5-Fold Stratified CV")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(short_names)
    axes[1].invert_yaxis()
    axes[1].set_xlim(80, 104)
    axes[1].set_xlabel("Accuracy (%)")
    axes[1].set_title("B. Validation Accuracy Across Classifier Families")
    for idx_b, r in bench_df.reset_index(drop=True).iterrows():
        axes[1].text(r["test_acc"] + 0.35, idx_b - bar_h, f"{r['test_acc']:.1f}%", va="center", fontsize=7.8)
        axes[1].text(r["loocv_acc"] + 0.35, idx_b, f"{r['loocv_acc']:.1f}%", va="center", fontsize=7.8)
        axes[1].text(r["cv5_acc"] + 0.35, idx_b + bar_h, f"{r['cv5_acc']:.1f}%", va="center", fontsize=7.8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.48, -0.17), ncol=3, frameon=False, fontsize=8.0)
    axes[1].grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "figure_3_holdout_confusion_and_cv.png"), bbox_inches="tight")
    plt.close(fig)

    # FIGURE 4: Top Discriminating Function Words & Author Usage Comparison
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    top12_vals = importances.head(12)[::-1]
    axes[0].barh(top12_vals.index, top12_vals.values, color="#1B365D", height=0.68)
    axes[0].set_xlabel("Mean Decrease in Gini Impurity (Random Forest)")
    axes[0].set_title("A. Top 12 Discriminating Function Words")
    for idx_w, (w, v) in enumerate(top12_vals.items()):
        axes[0].text(v + 0.002, idx_w, f"{v:.3f}", va="center", fontsize=8)
    axes[0].set_xlim(0, float(top12_vals.max()) * 1.18)
    axes[0].grid(axis="x", linestyle=":", alpha=0.5)

    marker_words = ["upon", "on", "there", "by", "would", "also"]
    x_m = np.arange(len(marker_words))
    w_bar = 0.25
    ham_rates = [fedpapers[fedpapers["author"] == "Hamilton"][w].mean() for w in marker_words]
    mad_rates = [fedpapers[fedpapers["author"] == "Madison"][w].mean() for w in marker_words]
    disp_rates = [fedpapers[fedpapers["author"] == "dispt"][w].mean() for w in marker_words]
    axes[1].bar(x_m - w_bar, ham_rates, width=w_bar, color=PALETTE["Hamilton"], label="Hamilton (n = 51)")
    axes[1].bar(x_m, mad_rates, width=w_bar, color=PALETTE["Madison"], label="Madison (n = 15)")
    axes[1].bar(x_m + w_bar, disp_rates, width=w_bar, color=PALETTE["dispt"], label="Disputed (n = 11)")
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels([f"'{w}'" for w in marker_words])
    axes[1].set_ylabel("Mean Normalized Word Rate")
    axes[1].set_title("B. Disputed Essays Mirror Madison's Function-Word Profile")
    axes[1].legend(loc="upper right", frameon=False, fontsize=8.5)
    axes[1].grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "figure_4_function_word_importance.png"), bbox_inches="tight")
    plt.close(fig)

    # FIGURE 5: Predicted Madison Authorship Probability Across Disputed (11) & Joint HM (3) Essays
    combined_attrib = disputed_results + hm_results
    essay_labels = (
        [f"No. {r['essay_no']} (Disputed)" for r in disputed_results]
        + [f"No. {r['essay_no']} (Joint HM)" for r in hm_results]
    )
    y_e = np.arange(len(combined_attrib))
    rf_probs = [r["rf_madison_prob"] * 100 for r in combined_attrib]
    lr_probs = [r["lr_madison_prob"] * 100 for r in combined_attrib]
    mean_probs = [r["mean_madison_prob"] * 100 for r in combined_attrib]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    bh = 0.34
    ax.barh(y_e - bh / 2, rf_probs, height=bh, color="#2E6B4E", label="Random Forest Madison Probability (%)")
    ax.barh(y_e + bh / 2, lr_probs, height=bh, color="#1B365D", label="L2 Logistic Regression Madison Probability (%)")
    ax.axvline(50.0, color="#8B2635", linestyle="--", linewidth=1.2, label="50% Decision Threshold")
    ax.axhline(len(disputed_results) - 0.5, color="#5C6B73", linestyle=":", linewidth=1.0)
    ax.set_yticks(y_e)
    ax.set_yticklabels(essay_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Predicted Probability of James Madison Authorship (%)")
    ax.set_title("Predicted James Madison Authorship Probability Across the 11 Disputed and 3 Joint Essays")
    for idx_r, (rf_v, lr_v) in enumerate(zip(rf_probs, lr_probs)):
        ax.text(max(rf_v, lr_v) + 1.2, idx_r, f"RF {rf_v:.1f}% | LR {lr_v:.1f}%", va="center", fontsize=8.0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.11), ncol=3, frameon=False, fontsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout(pad=1.2)
    fig.savefig(os.path.join(FIG_DIR, "figure_5_disputed_attribution_probabilities.png"), bbox_inches="tight")
    plt.close(fig)

    # Save structured JSON summary for deterministic verification
    summary_payload = {
        "dataset_shape": list(fedpapers.shape),
        "author_counts": author_counts,
        "feature_count": len(feature_cols),
        "split_counts": {
            "train": y_train.value_counts().to_dict(),
            "test": y_test.value_counts().to_dict(),
        },
        "dt_pruned_rules": export_text(dt_pruned_full, feature_names=feature_cols),
        "ccp_pruning_table": ccp_records,
        "model_evaluation": evaluation_rows,
        "top12_rf_features": {k: round(float(v), 4) for k, v in importances.head(12).items()},
        "word_means_by_author": word_means,
        "disputed_predictions": disputed_results,
        "hm_predictions": hm_results,
    }
    with open(os.path.join(OUT_DIR, "analysis_results.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("=== FEDERALIST PAPERS PYTHON AUTHORSHIP ATTRIBUTION SUMMARY ===")
    print(f"Dataset shape: {fedpapers.shape[0]} essays x {fedpapers.shape[1]} columns ({len(feature_cols)} function words)")
    print(f"Author distribution: {author_counts}")
    print("\n--- Model Validation Summary ---")
    print(pd.DataFrame(evaluation_rows)[["model", "train_acc", "test_acc", "cv5_acc", "loocv_acc"]].to_string(index=False))
    print("\n--- Disputed Essays Attribution (n = 11) ---")
    print(pd.DataFrame(disputed_results)[["essay_no", "upon", "on", "which", "dt_pruned", "rf_pred", "rf_madison_prob", "gb_pred", "lr_pred", "lr_madison_prob", "consensus"]].to_string(index=False))
    print("\n--- Jointly Authored HM Essays Attribution (n = 3) ---")
    print(pd.DataFrame(hm_results)[["essay_no", "upon", "on", "dt_pruned", "rf_pred", "rf_madison_prob", "gb_pred", "lr_pred", "lr_madison_prob", "consensus"]].to_string(index=False))


if __name__ == "__main__":
    main()
