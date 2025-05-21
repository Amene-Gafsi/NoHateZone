# crossmodal_random_cv.py

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, roc_curve
)
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt

# =============================================================================
# Data loading
# =============================================================================

def load_dataframe():
    df = pd.read_pickle('embeddings_hatemm_mert.pkl')
    parts = df['frameID'].str.rsplit(pat='_frame', n=1, expand=True)
    df['video_group'] = parts[0]
    return df

# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true, y_prob, threshold=0.5):
    # random predictions via fixed 0.5 threshold
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
    }

# =============================================================================
# Main
# =============================================================================

def main():
    out_dir = Path('./random_cv_metrics')
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe()
    seeds = list(range(20))
    k_folds = 5

    final_metrics = {}
    seed_rocs = {}

    for seed in seeds:
        np.random.seed(seed)

        # Test split at video_group level
        test_groups = (
            df['video_group']
              .drop_duplicates()
              .sample(frac=0.2, random_state=seed)
              .tolist()
        )
        df_test = df[df['video_group'].isin(test_groups)]
        df_train_val = df[~df['video_group'].isin(test_groups)]

        # Build folds (we only use them to draw random val scores)
        folds = list(
            GroupKFold(n_splits=k_folds)
            .split(df_train_val, groups=df_train_val['video_group'])
        )

        # Collect random validation‐set scores (unused for thresholding)
        for tr_idx, val_idx in folds:
            df_val = df_train_val.iloc[val_idx]
            # purely random scores
            _ = np.random.rand(len(df_val))  # throwaway

        # Now evaluate on TEST set with purely random 0–1 scores
        y_test = df_test['label'].values
        p_test = np.random.rand(len(df_test))

        # Fix threshold=0.5
        tm = compute_metrics(y_test, p_test, threshold=0.5)
        final_metrics[seed] = tm

        # Also collect ROC curve for this seed
        fpr, tpr, _ = roc_curve(y_test, p_test)
        seed_rocs[seed] = (fpr, tpr)

        print(f"Seed {seed} → "
              f"ACC={tm['accuracy']:.4f}  "
              f"AUC={tm['auc']:.4f}  "
              f"F1={tm['f1']:.4f}  "
              f"Prec={tm['precision']:.4f}  "
              f"Rec={tm['recall']:.4f}  "
              f"Macro-F1={tm['macro_f1']:.4f}")

    # Save per-seed metrics
    df_metrics = pd.DataFrame.from_dict(final_metrics, orient='index')
    df_metrics.index.name = 'seed'
    df_metrics.to_csv(out_dir/'test_metrics.csv')

    # Print averaged metrics
    avg = df_metrics.mean()
    print("\nAverage across seeds:")
    for m, v in avg.items():
        print(f"  {m:10s}: {v:.4f}")

    # Plot ROC per seed
    plt.figure(figsize=(8,6))
    for seed, (fpr, tpr) in seed_rocs.items():
        plt.plot(fpr, tpr, lw=1.5, label=f"Seed {seed}")
    plt.plot([0,1],[0,1],'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC per Seed (Random Classifier)")
    plt.legend(loc="lower right", ncol=2, fontsize='small')
    plt.savefig(out_dir/'roc_per_seed.png')
    plt.close()

if __name__ == '__main__':
    main()
