import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
import os


def plot_roc(df, label):
    fpr, tpr, _ = roc_curve(df["true_label"], df["predicted_prob"])
    plt.plot(fpr, tpr, label=f"Seed {label}")


def main():
    seeds = [0, 1, 50, 100, 1000]
    for seed in seeds:
        root_path = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.join(root_path, "../../hmm_modelseeds")

        path = os.path.join(root_dir, f"/roc_{seed}.csv")
        df = pd.read_csv(path)
        plot_roc(df, seed)

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("DCMA: ROC Curves per Seed")
    plt.legend(loc="lower right")
    plt.tight_layout()
    output_path = "./NoHateZone/checpont_hmm_seeds/roc_comparison.png"
    plt.savefig(output_path, dpi=300)
    print(f"ROC curve plot saved to: {output_path}")


if __name__ == "__main__":
    main()
