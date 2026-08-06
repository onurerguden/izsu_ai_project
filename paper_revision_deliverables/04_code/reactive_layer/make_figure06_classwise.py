import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("models/outputs/reactive/reactive_classwise_metrics.csv")
sub = df[(df["Model"] == "SVM") & (df["Experiment"] == "E3_AUGMENTED_TRAIN_MIXED_TEST")]
sub = sub.set_index("Class").loc[["Good", "Caution", "Risk"]]

metrics = ["Precision", "Recall", "F1"]
classes = ["Good", "Caution", "Risk"]
colors = ["tab:blue", "tab:orange", "tab:green"]

x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
for i, metric in enumerate(metrics):
    values = sub[metric].values
    bars = ax.bar(x + (i - 1) * width, values, width, label=metric, color=colors[i])
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=11)

ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=13)
ax.set_ylabel("Score", fontsize=13)
ax.set_ylim(0, 1.08)
ax.set_title("Class-wise Precision, Recall and F1-score (SVM, Mixed Test Set)", fontsize=13)
ax.legend(loc="lower left", fontsize=11, ncol=3)
ax.grid(axis="y", linestyle="--", alpha=0.4)
fig.tight_layout()

out1 = "models/outputs/reactive/figures/figure_06_svm_classwise_performance.png"
fig.savefig(out1, dpi=300)
out2 = "paper_revision_deliverables/07_figures_300dpi/reactive/Figure_06_SVM_classwise_performance_MixedTest.png"
fig.savefig(out2, dpi=300)
print("saved", out1, out2)
print(sub[metrics])
