#!/usr/bin/env python3
"""
One-Page Graphical Summary Infographic: Federalist Papers Authorship Attribution.

Generates a single-page (8.5 x 11.0 in, 300 DPI PNG + vector PDF) quantitative
visual summary directly from `fedPapers85.csv` and `outputs/analysis_results.json`
using `pandas`, `numpy`, `scikit-learn`, and `matplotlib`. Contains purely
graphical data visualizations with zero portraits or external illustrations.
"""

import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "fedPapers85.csv")
JSON_PATH = os.path.join(BASE_DIR, "outputs", "analysis_results.json")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Typography and committee palette
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.titlesize": 9.8,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

NAVY = "#1B365D"       # Hamilton / primary structural accent
GREEN = "#2E6B4E"      # Madison
TAUPE = "#7A6855"      # Jay
BURGUNDY = "#8B2635"   # Disputed / contested callout
SLATE = "#5C6B73"      # Neutral secondary
LIGHT_BG = "#F8FAFC"   # Subtle card background
BORDER = "#CBD5E1"     # Hairline card border
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#475569"


def essay_number(filename: str) -> int:
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else -1


def draw_tree_box(ax, cx, cy, w, h, title, lines, edge_color, fill_color, header_fill=None):
    """Draw a clean vector node card on normalized axis coordinates."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2),
        w,
        h,
        boxstyle="round,pad=0.010,rounding_size=0.020",
        linewidth=1.2,
        edgecolor=edge_color,
        facecolor=fill_color,
        zorder=3,
    )
    ax.add_patch(box)

    if header_fill:
        hdr_h = h * 0.33
        hdr = FancyBboxPatch(
            (cx - w / 2, cy + h / 2 - hdr_h),
            w,
            hdr_h,
            boxstyle="round,pad=0.010,rounding_size=0.020",
            linewidth=0,
            facecolor=header_fill,
            zorder=4,
        )
        ax.add_patch(hdr)
        ax.text(
            cx,
            cy + h / 2 - hdr_h * 0.45,
            title,
            ha="center",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color="white",
            zorder=5,
        )
        body_top = cy + h / 2 - hdr_h - 0.045
        step = (h * 0.42) / max(len(lines), 1)
    else:
        ax.text(
            cx,
            cy + h * 0.28,
            title,
            ha="center",
            va="center",
            fontsize=7.9,
            fontweight="bold",
            color=edge_color,
            zorder=5,
        )
        body_top = cy + h * 0.02
        step = (h * 0.46) / max(len(lines), 1)

    for idx, line in enumerate(lines):
        ax.text(
            cx,
            body_top - idx * step,
            line,
            ha="center",
            va="center",
            fontsize=7.0,
            color=TEXT_DARK,
            zorder=5,
        )


def main() -> None:
    # Load primary dataset and deterministic model output JSON
    fedpapers = pd.read_csv(DATA_PATH)
    fedpapers["essay_no"] = fedpapers["filename"].apply(essay_number)
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)

    feature_cols = [c for c in fedpapers.columns if c not in ("author", "filename", "essay_no")]
    known_df = fedpapers[~fedpapers["author"].isin(["dispt", "HM"])].copy().reset_index(drop=True)
    X_known = known_df[feature_cols]
    y_known = known_df["author"]

    # Fit full-corpus pruned CART tree (ccp_alpha = 0.030) to read exact node stats
    dt_full = DecisionTreeClassifier(criterion="gini", ccp_alpha=0.030, random_state=42)
    dt_full.fit(X_known, y_known)
    tree_ = dt_full.tree_
    classes = list(dt_full.classes_)  # ['Hamilton', 'Jay', 'Madison']

    root_feat = feature_cols[tree_.feature[0]]
    root_thr = float(tree_.threshold[0])
    root_n = int(tree_.n_node_samples[0])
    root_gini = float(tree_.impurity[0])

    left_idx = int(tree_.children_left[0])
    right_idx = int(tree_.children_right[0])  # Hamilton leaf

    split2_feat = feature_cols[tree_.feature[left_idx]]
    split2_thr = float(tree_.threshold[left_idx])
    split2_n = int(tree_.n_node_samples[left_idx])
    split2_gini = float(tree_.impurity[left_idx])

    ll_idx = int(tree_.children_left[left_idx])   # Jay leaf
    lr_idx = int(tree_.children_right[left_idx])  # Madison leaf

    def node_counts(nid: int) -> tuple[int, int, int]:
        probs = tree_.value[nid][0]
        counts = np.rint(probs * tree_.n_node_samples[nid]).astype(int)
        d = dict(zip(classes, counts))
        return d["Hamilton"], d["Madison"], d["Jay"]

    ham_leaf_cnt = node_counts(right_idx)
    jay_leaf_cnt = node_counts(ll_idx)
    mad_leaf_cnt = node_counts(lr_idx)

    # Create single US Letter (8.5 x 11.0 in) canvas
    fig = plt.figure(figsize=(8.5, 11.0), facecolor="#FFFFFF")

    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.axis("off")
    bg_ax.set_xlim(0, 1)
    bg_ax.set_ylim(0, 1)

    # =========================================================================
    # 1. TOP BANNER
    # =========================================================================
    banner = Rectangle((0.035, 0.908), 0.93, 0.072, facecolor=NAVY, edgecolor="none", zorder=1)
    bg_ax.add_patch(banner)
    accent_bar = Rectangle((0.035, 0.904), 0.93, 0.004, facecolor=BURGUNDY, edgecolor="none", zorder=2)
    bg_ax.add_patch(accent_bar)

    bg_ax.text(
        0.055,
        0.963,
        "QUANTITATIVE STYLOMETRIC BRIEF   |   PYTHON DECISION TREES, ENSEMBLES & REGULARIZED CLASSIFICATION",
        fontsize=7.1,
        fontweight="bold",
        color="#CBD5E1",
        va="center",
    )
    bg_ax.text(
        0.055,
        0.939,
        "Who Wrote the Disputed Federalist Papers?",
        fontsize=15.5,
        fontweight="bold",
        color="#FFFFFF",
        va="center",
    )
    bg_ax.text(
        0.055,
        0.918,
        "Empirical Attribution Across 85 Constitutional Essays and 70 Normalized Function Words (fedPapers85.csv)",
        fontsize=8.3,
        color="#E2E8F0",
        va="center",
    )

    # =========================================================================
    # 2. KPI SUMMARY STRIP (4 CARDS)
    # =========================================================================
    ac = results["author_counts"]
    eval_map = {m["model"]: m for m in results["model_evaluation"]}
    pruned_m = eval_map["DT Pruned (ccp_alpha=0.030)"]
    lr_m = eval_map["L2 Logistic Regression"]

    kpis = [
        {
            "metric": f"{results['dataset_shape'][0]} Essays | {results['feature_count']} Words",
            "label": "Corpus Structure",
            "sub": f"{ac['Hamilton']} Hamilton • {ac['Madison']} Madison • {ac['Jay']} Jay\n{ac['HM']} Joint HM • {ac['dispt']} Disputed",
            "color": NAVY,
        },
        {
            "metric": f"{pruned_m['cv5_acc']:.1f}% CV  ({pruned_m['test_acc']:.0f}% Test)",
            "label": "Pruned Decision Tree",
            "sub": "ccp_alpha = 0.030 • 3 Leaves\nSplits on 'upon' & 'which' Only",
            "color": TAUPE,
        },
        {
            "metric": f"{lr_m['loocv_acc']:.1f}% LOOCV",
            "label": "Peak Validation Accuracy",
            "sub": f"L2 Logistic Regression (69/71)\n{lr_m['cv5_acc']:.1f}% 5-Fold Stratified CV",
            "color": GREEN,
        },
        {
            "metric": "10 of 11 to Madison",
            "label": "Disputed & Joint Verdict",
            "sub": "Nos. 49–54, 56–57, 62–63 + HM 18–20\nFederalist No. 55 Splits 2–2",
            "color": BURGUNDY,
        },
    ]

    card_w = 0.221
    gap = 0.0153
    x0 = 0.035
    y0 = 0.812
    card_h = 0.080

    for idx, kpi in enumerate(kpis):
        cx = x0 + idx * (card_w + gap)
        card = FancyBboxPatch(
            (cx, y0),
            card_w,
            card_h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor=LIGHT_BG,
            edgecolor=BORDER,
            linewidth=0.9,
            zorder=1,
        )
        bg_ax.add_patch(card)
        top_strip = Rectangle((cx, y0 + card_h - 0.006), card_w, 0.006, facecolor=kpi["color"], edgecolor="none", zorder=2)
        bg_ax.add_patch(top_strip)

        bg_ax.text(cx + 0.011, y0 + 0.061, kpi["label"].upper(), fontsize=6.7, fontweight="bold", color=TEXT_MUTED, va="center")
        bg_ax.text(cx + 0.011, y0 + 0.041, kpi["metric"], fontsize=10.6, fontweight="bold", color=kpi["color"], va="center")
        bg_ax.text(cx + 0.011, y0 + 0.016, kpi["sub"], fontsize=6.7, color=TEXT_DARK, va="center", linespacing=1.25)

    def add_panel_frame(x, y, w, h, title, subtitle):
        panel = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.003,rounding_size=0.008",
            facecolor="#FFFFFF",
            edgecolor=BORDER,
            linewidth=0.9,
            zorder=0.5,
        )
        bg_ax.add_patch(panel)
        bg_ax.text(x + 0.014, y + h - 0.018, title, fontsize=9.6, fontweight="bold", color=NAVY, va="center")
        bg_ax.text(x + 0.014, y + h - 0.034, subtitle, fontsize=7.1, color=TEXT_MUTED, va="center")

    # =========================================================================
    # 3. ROW 2 LEFT: PRUNED DECISION TREE SCHEMATIC (FROM LIVE SKLEARN TREE_)
    # =========================================================================
    add_panel_frame(
        0.035,
        0.492,
        0.452,
        0.306,
        "Pruned Decision Tree",
        "Fitted on all 71 confirmed essays (ccp_alpha = 0.030, 3 terminal leaves)",
    )
    ax_tree = fig.add_axes([0.042, 0.500, 0.438, 0.252])
    ax_tree.set_xlim(0, 1)
    ax_tree.set_ylim(0, 1)
    ax_tree.axis("off")

    # Root node
    draw_tree_box(
        ax_tree,
        0.50,
        0.85,
        0.44,
        0.23,
        f"Root Split: '{root_feat}' <= {root_thr:.3f}",
        [f"n = {root_n} confirmed  |  Gini = {root_gini:.3f}", "51 Hamilton • 15 Madison • 5 Jay"],
        edge_color=NAVY,
        fill_color="#F1F5F9",
    )

    # Second split node (left branch)
    draw_tree_box(
        ax_tree,
        0.26,
        0.49,
        0.41,
        0.22,
        f"Branch Split: '{split2_feat}' <= {split2_thr:.3f}",
        [f"n = {split2_n} essays  |  Gini = {split2_gini:.3f}", "1 Hamilton • 15 Madison • 5 Jay"],
        edge_color=SLATE,
        fill_color="#F8FAFC",
    )

    # Right leaf: Hamilton
    draw_tree_box(
        ax_tree,
        0.78,
        0.49,
        0.36,
        0.24,
        "Leaf 3: HAMILTON",
        [f"n = {sum(ham_leaf_cnt)}  |  Gini = {tree_.impurity[right_idx]:.3f}", f"{ham_leaf_cnt[0]} Ham • {ham_leaf_cnt[1]} Mad • {ham_leaf_cnt[2]} Jay"],
        edge_color=NAVY,
        fill_color="#EFF6FF",
        header_fill=NAVY,
    )

    # Bottom-left leaf: Jay
    draw_tree_box(
        ax_tree,
        0.175,
        0.135,
        0.32,
        0.24,
        "Leaf 1: JAY",
        [f"n = {sum(jay_leaf_cnt)}  |  Gini = {tree_.impurity[ll_idx]:.3f}", f"{jay_leaf_cnt[0]} Ham • {jay_leaf_cnt[1]} Mad • {jay_leaf_cnt[2]} Jay"],
        edge_color=TAUPE,
        fill_color="#FAF8F5",
        header_fill=TAUPE,
    )

    # Bottom-mid leaf: Madison
    draw_tree_box(
        ax_tree,
        0.535,
        0.135,
        0.33,
        0.24,
        "Leaf 2: MADISON",
        [f"n = {sum(mad_leaf_cnt)}  |  Gini = {tree_.impurity[lr_idx]:.3f}", f"{mad_leaf_cnt[0]} Ham • {mad_leaf_cnt[1]} Mad • {mad_leaf_cnt[2]} Jay"],
        edge_color=GREEN,
        fill_color="#F0FDF4",
        header_fill=GREEN,
    )

    # Connector arrows + edge labels
    arrow_kw = dict(arrowstyle="-|>,head_width=0.35,head_length=0.45", lw=1.15, color=SLATE, zorder=2)
    ax_tree.annotate("", xy=(0.26, 0.615), xytext=(0.40, 0.725), arrowprops=arrow_kw)
    ax_tree.text(0.285, 0.675, f"Yes (<= {root_thr:.3f})", fontsize=6.8, fontweight="bold", color=TEXT_DARK, ha="right")

    ax_tree.annotate("", xy=(0.76, 0.625), xytext=(0.60, 0.725), arrowprops=arrow_kw)
    ax_tree.text(0.715, 0.675, f"No (> {root_thr:.3f})", fontsize=6.8, fontweight="bold", color=NAVY, ha="left")

    ax_tree.annotate("", xy=(0.175, 0.27), xytext=(0.225, 0.37), arrowprops=arrow_kw)
    ax_tree.text(0.165, 0.325, f"<= {split2_thr:.3f}", fontsize=6.8, fontweight="bold", color=TAUPE, ha="right")

    ax_tree.annotate("", xy=(0.50, 0.27), xytext=(0.335, 0.37), arrowprops=arrow_kw)
    ax_tree.text(0.455, 0.325, f"> {split2_thr:.3f}", fontsize=6.8, fontweight="bold", color=GREEN, ha="left")

    # Boundary case callout card with ample left clearance from Leaf 2
    note_box = FancyBboxPatch(
        (0.735, 0.015),
        0.245,
        0.26,
        boxstyle="round,pad=0.008,rounding_size=0.02",
        facecolor="#FFF5F5",
        edgecolor=BURGUNDY,
        linewidth=0.85,
        zorder=3,
    )
    ax_tree.add_patch(note_box)
    ax_tree.text(
        0.858,
        0.145,
        f"Boundary Case:\nNo. 51 has\n'which' = 0.105\n(<= {split2_thr:.3f} -> Jay;\nRF 61% & LR\n99.8% Madison)",
        ha="center",
        va="center",
        fontsize=6.25,
        color=BURGUNDY,
        linespacing=1.22,
    )

    # =========================================================================
    # 4. ROW 2 RIGHT: CLASSIFIER VALIDATION BENCHMARK
    # =========================================================================
    add_panel_frame(
        0.502,
        0.492,
        0.463,
        0.306,
        "Model Benchmark",
        "5-fold stratified and leave-one-out cross-validation across models (n = 71)",
    )
    ax_bench = fig.add_axes([0.618, 0.556, 0.330, 0.190])

    selected_models = [
        ("DT Tuned 1 (max_depth=1)", "Stump Tree\n(max_depth=1)"),
        ("DT Baseline (min_split=20)", "Baseline Tree\n(min_split=20)"),
        ("DT Pruned (ccp_alpha=0.030)", "Pruned Tree\n(ccp=0.030)"),
        ("Random Forest (500 Trees)", "Random Forest\n(500 Trees)"),
        ("Gradient Boosted Trees", "Gradient\nBoosting"),
        ("L2 Logistic Regression", "L2 Logistic\nRegression"),
    ]
    b_rows = [eval_map[k] for k, _ in selected_models]
    b_labels = [lbl for _, lbl in selected_models]

    y_pos = np.arange(len(b_rows))
    bh = 0.34
    cv5_vals = [r["cv5_acc"] for r in b_rows]
    loo_vals = [r["loocv_acc"] for r in b_rows]

    ax_bench.barh(y_pos - bh / 2, cv5_vals, height=bh, color=NAVY, label="5-Fold Stratified CV (n = 71)")
    ax_bench.barh(y_pos + bh / 2, loo_vals, height=bh, color=GREEN, label="Leave-One-Out CV (n = 71)")

    ax_bench.set_yticks(y_pos)
    ax_bench.set_yticklabels(b_labels, fontsize=7.1)
    ax_bench.invert_yaxis()
    ax_bench.set_xlim(84, 102.5)
    ax_bench.set_xlabel("Cross-Validation Accuracy (%)", fontsize=7.5)
    ax_bench.grid(axis="x", linestyle=":", alpha=0.55)

    for idx_b, (c_v, l_v) in enumerate(zip(cv5_vals, loo_vals)):
        ax_bench.text(c_v + 0.3, idx_b - bh / 2, f"{c_v:.1f}%", va="center", fontsize=6.7, color=NAVY, fontweight="bold")
        ax_bench.text(l_v + 0.3, idx_b + bh / 2, f"{l_v:.1f}%", va="center", fontsize=6.7, color=GREEN, fontweight="bold")

    ax_bench.legend(loc="upper center", bbox_to_anchor=(0.40, -0.21), ncol=2, frameon=False, fontsize=6.9)

    # =========================================================================
    # 5. ROW 3 LEFT: FUNCTION-WORD RATES (DISPUTED MATCH MADISON; NO. 55 OUTLIER)
    # =========================================================================
    add_panel_frame(
        0.035,
        0.072,
        0.452,
        0.404,
        "Function-Word Rates",
        "Mean normalized rates across primary markers (with Federalist No. 55 overlaid)",
    )
    ax_words = fig.add_axes([0.088, 0.135, 0.382, 0.282])

    markers = ["upon", "on", "there", "by", "would", "also"]
    xm = np.arange(len(markers))
    wb = 0.25

    ham_m = [float(fedpapers[fedpapers["author"] == "Hamilton"][w].mean()) for w in markers]
    mad_m = [float(fedpapers[fedpapers["author"] == "Madison"][w].mean()) for w in markers]
    dsp_m = [float(fedpapers[fedpapers["author"] == "dispt"][w].mean()) for w in markers]
    no55_row = fedpapers[fedpapers["essay_no"] == 55].iloc[0]
    no55_m = [float(no55_row[w]) for w in markers]

    ax_words.bar(xm - wb, ham_m, width=wb, color=NAVY, label="Hamilton (n = 51)")
    ax_words.bar(xm, mad_m, width=wb, color=GREEN, label="Madison (n = 15)")
    ax_words.bar(xm + wb, dsp_m, width=wb, color="#65A30D", alpha=0.85, label="Disputed Mean (n = 11)")
    ax_words.scatter(
        xm + wb,
        no55_m,
        marker="D",
        s=28,
        color=BURGUNDY,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
        label="Federalist No. 55",
    )

    ax_words.set_xticks(xm)
    ax_words.set_xticklabels([f"'{w}'" for w in markers], fontsize=7.7, fontweight="bold")
    ax_words.set_ylabel("Mean Normalized Word Rate", fontsize=7.6)
    ax_words.set_ylim(0, 0.228)
    ax_words.grid(axis="y", linestyle=":", alpha=0.55)
    ax_words.legend(loc="upper right", frameon=True, facecolor="#FFFFFF", edgecolor=BORDER, fontsize=6.8)

    # Clean non-overlapping callouts placed strictly in open white space
    ax_words.annotate(
        f"Hamilton 'upon'\n({ham_m[0]:.3f}) vs. Disputed\nmean ({dsp_m[0]:.3f})",
        xy=(0 - wb, ham_m[0] + 0.002),
        xytext=(-0.38, 0.140),
        fontsize=6.4,
        color=NAVY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=0.85, color=NAVY),
    )
    ax_words.annotate(
        f"No. 55 matches\nHamilton on 'on'\n({no55_m[1]:.3f}) & 'by' ({no55_m[3]:.3f})",
        xy=(1 + wb, no55_m[1] + 0.003),
        xytext=(0.95, 0.184),
        fontsize=6.4,
        color=BURGUNDY,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=0.85, color=BURGUNDY),
    )

    bg_ax.text(
        0.048,
        0.085,
        "Disputed essays mirror Madison across all 6 markers; No. 55 diverges toward Hamilton on 'on' and 'by'.",
        fontsize=6.35,
        color=TEXT_MUTED,
        va="center",
    )

    # =========================================================================
    # 6. ROW 3 RIGHT: AUTHORSHIP PROBABILITIES ACROSS ALL 14 TARGET ESSAYS
    # =========================================================================
    add_panel_frame(
        0.502,
        0.072,
        0.463,
        0.404,
        "Authorship Probabilities",
        "Predicted James Madison probability across 11 disputed and 3 joint HM papers",
    )
    ax_prob = fig.add_axes([0.585, 0.146, 0.362, 0.278])

    combined = results["disputed_predictions"] + results["hm_predictions"]
    p_labels = (
        [f"No. {r['essay_no']} (Disp.)" for r in results["disputed_predictions"]]
        + [f"No. {r['essay_no']} (HM)" for r in results["hm_predictions"]]
    )
    yp = np.arange(len(combined))
    rf_p = [r["rf_madison_prob"] * 100 for r in combined]
    lr_p = [r["lr_madison_prob"] * 100 for r in combined]

    idx_55 = next(i for i, r in enumerate(combined) if r["essay_no"] == 55)
    ax_prob.axhspan(idx_55 - 0.46, idx_55 + 0.46, color="#FFF1F2", zorder=0)

    bh_p = 0.35
    ax_prob.barh(yp - bh_p / 2, rf_p, height=bh_p, color=GREEN, label="Random Forest Madison %")
    ax_prob.barh(yp + bh_p / 2, lr_p, height=bh_p, color=NAVY, label="L2 Logistic Reg. Madison %")
    ax_prob.axvline(50.0, color=BURGUNDY, linestyle="--", linewidth=1.0, label="50% Threshold")
    ax_prob.axhline(len(results["disputed_predictions"]) - 0.5, color=SLATE, linestyle=":", linewidth=0.9)

    ax_prob.set_yticks(yp)
    ax_prob.set_yticklabels(p_labels, fontsize=6.8)
    ax_prob.invert_yaxis()
    ax_prob.set_xlim(0, 126)
    ax_prob.set_xlabel("Madison Authorship Probability (%)", fontsize=7.4)
    ax_prob.grid(axis="x", linestyle=":", alpha=0.5)

    for idx_r, (rv, lv, rec) in enumerate(zip(rf_p, lr_p, combined)):
        mx = max(rv, lv)
        if rec["essay_no"] == 55:
            # Place text cleanly to the right of the 50% vertical dashed line with a short connector
            ax_prob.plot([mx + 0.8, 52.0], [idx_r - bh_p / 2, idx_r], color=BURGUNDY, linewidth=0.7)
            ax_prob.text(53.0, idx_r, f"Split 2–2 ({rv:.0f}% | {lv:.0f}%)", va="center", fontsize=6.25, color=BURGUNDY, fontweight="bold")
        elif rec["essay_no"] == 51:
            ax_prob.text(mx + 1.4, idx_r, f"{rv:.0f}% | {lv:.1f}%*", va="center", fontsize=6.25, color=NAVY, fontweight="bold")
        else:
            ax_prob.text(mx + 1.4, idx_r, f"{rv:.0f}% | {lv:.0f}%", va="center", fontsize=6.2, color=TEXT_DARK)

    ax_prob.legend(loc="upper center", bbox_to_anchor=(0.44, -0.155), ncol=3, frameon=False, fontsize=6.35)
    bg_ax.text(
        0.514,
        0.081,
        "*No. 51 resolved to Madison by multi-feature models (99.8% LR). Labels show: RF % | LR %.",
        fontsize=6.35,
        color=TEXT_MUTED,
        va="center",
    )

    # =========================================================================
    # 7. FOOTER SOURCE BAND
    # =========================================================================
    bg_ax.plot([0.035, 0.965], [0.056, 0.056], color=BORDER, linewidth=0.9)
    bg_ax.text(
        0.035,
        0.038,
        "Methodology & Sources: Generated programmatically in Python (generate_federalist_infographic.py) from fedPapers85.csv (70 function words across 85 essays) and",
        fontsize=6.8,
        color=TEXT_MUTED,
        va="center",
    )
    bg_ax.text(
        0.035,
        0.023,
        "validated scikit-learn models (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier, LogisticRegression). Historical baseline: Mosteller & Wallace (1964); Library of Congress.",
        fontsize=6.8,
        color=TEXT_MUTED,
        va="center",
    )

    png_out = os.path.join(FIG_DIR, "federalist_authorship_infographic.png")
    pdf_out = os.path.join(FIG_DIR, "federalist_authorship_infographic.pdf")
    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out)
    plt.close(fig)
    print(f"Saved PNG infographic: {png_out}")
    print(f"Saved PDF infographic: {pdf_out}")


if __name__ == "__main__":
    main()
