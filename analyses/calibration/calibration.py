import numpy as np
import os
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef, classification_report
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from collections import Counter

# ============================================
# LOGGER CLASS
# ============================================
class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# ============================================
# Helper Classes & Functions
# ============================================
class IsotonicCalibrator:
    def __init__(self, n_classes=3):
        self.n_classes = n_classes
        self.regressors = []
    def fit(self, probs, y):
        self.regressors = []
        for i in range(self.n_classes):
            y_binary = (y == i).astype(int)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(probs[:, i], y_binary)
            self.regressors.append(iso)
        return self
    def predict_proba(self, probs):
        calibrated = np.zeros_like(probs)
        for i, iso in enumerate(self.regressors):
            calibrated[:, i] = iso.predict(probs[:, i])
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1e-10
        return calibrated / row_sums

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def compute_calibration_metrics(probs, y_true, n_bins=10, n_classes=3):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    N = len(y_true)
    fixed_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, mcs = 0.0, 0.0, 0.0
    bin_stats = [] 
    for i, (bin_lower, bin_upper) in enumerate(zip(fixed_boundaries[:-1], fixed_boundaries[1:])):
        if i == 0:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        count_in_bin = np.sum(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            conf_in_bin = np.mean(confidences[in_bin])
            gap = float(np.abs(conf_in_bin - acc_in_bin))
            signed_gap = float(conf_in_bin - acc_in_bin)
            ece += gap * prop_in_bin
            mcs += signed_gap * prop_in_bin
            mce = max(mce, gap)
            bin_y_true = y_true[in_bin]
            class_counts = [int(np.sum(bin_y_true == c)) for c in range(n_classes)]
            bin_stats.append({"lower": bin_lower, "upper": bin_upper, "count": int(count_in_bin), "prop": prop_in_bin, "acc": acc_in_bin, "conf": conf_in_bin, "gap": gap, "classes": class_counts})
        else:
            bin_stats.append({"lower": bin_lower, "upper": bin_upper, "count": 0, "prop": 0.0, "acc": None, "conf": None, "gap": None, "classes": [0]*n_classes})
    
    sorted_indices = np.argsort(confidences)
    sorted_confs, sorted_accs = confidences[sorted_indices], accuracies[sorted_indices]
    ace = 0.0
    splits = np.array_split(np.arange(N), n_bins)
    for chunk in splits:
        if len(chunk) == 0: continue
        #ace += np.abs(np.mean(sorted_confs[chunk]) - np.mean(sorted_accs[chunk])) * (len(chunk)/N)
        ace += np.abs(np.mean(sorted_confs[chunk]) - np.mean(sorted_accs[chunk]))
    ace /= n_bins
    return ece, mce, ace, mcs, bin_stats


def compute_class_wise_metrics(probs, y_true, n_bins=10, n_classes=3):
    N = len(y_true)
    fixed_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    # Storage for per-class results
    class_ece = []
    class_mce = []
    class_mcs = []
    class_ace = []

    for c in range(n_classes):
        # One-vs-Rest data for current class
        c_probs = probs[:, c]
        c_trues = (y_true == c).astype(float)

        # --- Fixed Binning Metrics (ECE, MCE, MCS) ---
        c_ece_accum = 0.0
        c_mce_val = 0.0
        c_mcs_accum = 0.0

        for i in range(n_bins):
            bin_lower, bin_upper = fixed_boundaries[i], fixed_boundaries[i + 1]
            # Boundary handling: include 0.0 in the first bin
            if i == 0:
                in_bin = (c_probs >= bin_lower) & (c_probs <= bin_upper)
            else:
                in_bin = (c_probs > bin_lower) & (c_probs <= bin_upper)

            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                acc_in_bin = np.mean(c_trues[in_bin])
                conf_in_bin = np.mean(c_probs[in_bin])

                gap = np.abs(conf_in_bin - acc_in_bin)
                signed_gap = conf_in_bin - acc_in_bin

                c_ece_accum += gap * prop_in_bin
                c_mcs_accum += signed_gap * prop_in_bin
                c_mce_val = max(c_mce_val, gap)

        # --- Adaptive Binning Metric (ACE) ---
        sorted_indices = np.argsort(c_probs)
        sorted_c_confs = c_probs[sorted_indices]
        sorted_c_trues = c_trues[sorted_indices]

        c_ace_accum = 0.0
        splits = np.array_split(np.arange(N), n_bins)
        for chunk in splits:
            if len(chunk) > 0:
                c_ace_accum += np.abs(np.mean(sorted_c_confs[chunk]) - np.mean(sorted_c_trues[chunk]))

        # Store results for this class
        class_ece.append(c_ece_accum)
        class_mce.append(c_mce_val)
        class_mcs.append(c_mcs_accum)
        class_ace.append(c_ace_accum / n_bins)

    # Calculate Macro Averages (Mean of all classes)
    metrics = {
        "avg_class_ece": np.mean(class_ece),
        "avg_class_mce": np.mean(class_mce),
        "avg_class_ace": np.mean(class_ace),
        "avg_class_mcs": np.mean(class_mcs),
        "per_class_details": {
            "ece": class_ece,
            "mce": class_mce,
            "ace": class_ace,
            "mcs": class_mcs
        }
    }

    return metrics

def print_bin_details(bin_stats, title="Bin Analysis", top_n=5):
    print(f"\n{'='*110}\n {title}\n{'='*110}")
    print(f"{'Bin':<15} | {'Count':<6} | {'Prop':<6} | {'Conf':<6} | {'Acc':<6} | {'Gap':<6} | {'True Class Dist (0,1,2)'}")
    print("-" * 110)
    bins_sorted = sorted([b for b in bin_stats if b["gap"] is not None], key=lambda b: b["gap"], reverse=True)
    for b in bins_sorted[:top_n]:
        bin_range = f"[{b['lower']:.1f}, {b['upper']:.1f}]"
        print(f"{bin_range:<15} | {b['count']:<6d} | {b['prop']:<6.3f} | {b['conf']:<6.3f} | {b['acc']:<6.3f} | {b['gap']:<6.3f} | {str(b['classes'])}")
    print("="*110)


def print_class_wise_results(metrics, class_names=None, ece_thresholds={'good': 0.05, 'poor': 0.10},
                             ace_thresholds={'good': 0.06, 'poor': 0.12}):
    """Enhanced printer with MACRO evaluation for overall calibration"""
    if class_names is None:
        class_names = [f'Class {i}' for i in range(3)]

    details = metrics['per_class_details']

    print("🧮 CLASS-WISE CALIBRATION METRICS\n")

    df_classes = pd.DataFrame({
        'Class': class_names,
        'ECE': np.array(details['ece']),
        'ACE': np.array(details['ace']),
        'MCE': np.array(details['mce']),
        'MCS': np.array(details['mcs'])
    }).round(4)
    print(df_classes.to_string(index=False, float_format='%.4f'))

    print("\n🎯 EVALUATION CRITERIA")
    print(f"  ECE: Perfect=0.0, Good≤{ece_thresholds['good']}, Poor>{ece_thresholds['poor']}")
    print(f"  ACE: Perfect=0.0, Good≤{ace_thresholds['good']}, Poor>{ace_thresholds['poor']}")

    print(f"\n📊 MACRO AVERAGES (Overall Model Calibration)")
    avg_ece = metrics['avg_class_ece']
    avg_ace = metrics['avg_class_ace']

    print(f"  ECE: {avg_ece:.4f} | ", end="")
    if avg_ece == 0.0:
        print("🟢 PERFECT")
    elif avg_ece <= ece_thresholds['good']:
        print("🟢 GOOD")
    else:
        print("🔴 POOR")

    print(f"  ACE: {avg_ace:.4f} | ", end="")
    if avg_ace == 0.0:
        print("🟢 PERFECT")
    elif avg_ace <= ace_thresholds['good']:
        print("🟢 GOOD")
    else:
        print("🔴 POOR")

    print(f"  MCE: {metrics['avg_class_mce']:.4f} | MCS: {metrics['avg_class_mcs']:.4f} (direction)")

    print("\n🔍 PER-CLASS EVALUATION")
    print(" Class      ECE      ACE     MCS Direction")
    print("-" * 50)

    for i, (ece, ace, mcs) in enumerate(zip(details['ece'], details['ace'], details['mcs'])):
        # ECE status
        if ece == 0.0:
            ece_status = "🟢 PERFECT"
        elif ece <= ece_thresholds['good']:
            ece_status = "🟢 GOOD"
        else:
            ece_status = "🔴 POOR"

        # ACE status
        if ace == 0.0:
            ace_status = "🟢 PERFECT"
        elif ace <= ace_thresholds['good']:
            ace_status = "🟢 GOOD"
        else:
            ace_status = "🔴 POOR"

        # MCS direction
        if abs(mcs) <= 0.02:
            direction = "⚪ BALANCED"
        elif mcs > 0:
            direction = "🔴 OVERCONF"
        else:
            direction = "🟢 UNDERCONF"

        print(f" {class_names[i]:8s}  {ece_status:>8s}  {ace_status:>8s}  {direction}")

    # OVERALL MACRO SUMMARY
    overall_status = "🟢 WELL CALIBRATED" if (
                avg_ece <= ece_thresholds['good'] and avg_ace <= ace_thresholds['good']) else "🔴 POORLY CALIBRATED"
    print(f"\n🏆 OVERALL CALIBRATION: {overall_status}")
    print(f"   (Avg ECE={avg_ece:.4f}, Avg ACE={avg_ace:.4f})")


def plot_fold_reliability_and_scores(fold_idx, bin_stats_orig, ece_orig, mcs_orig, mce_orig, ace_orig, f1_orig,
                                     bin_stats_cal, ece_cal, mcs_cal, mce_cal, ace_cal, f1_cal, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    def plot_one(ax, bin_stats, title, ece, mcs, mce, ace, f1):
        confs = [b["conf"] for b in bin_stats if b["conf"] is not None]
        accs = [b["acc"] for b in bin_stats if b["conf"] is not None]
        ax.plot([0, 1], [0, 1], 'k--', label="Perfect")
        
        # Calc Good Bins
        total_good_prop = sum([b['prop'] for b in bin_stats if b["gap"] is not None and b["gap"] <= 0.05])
        pct_good = total_good_prop * 100

        if len(confs) > 0:
            color = 'blue' if 'Calibrated' in title else 'red'
            label = f"F1={f1:.4f}\nECE={ece:.4f}\nMCS={mcs:.4f}\nGood Bins: {pct_good:.1f}%"
            ax.plot(confs, accs, marker='o', color=color, linewidth=2, label=label)
            
            for b in bin_stats:
                if b["gap"] is not None and b["gap"] > 0.05:
                    prop_pct = b['prop'] * 100
                    txt = f"cnt:{b['count']}\n{prop_pct:.1f}%\ngap:{b['gap']:.3f}"
                    ax.text(b['conf'], b['acc'] + 0.03, txt, fontsize=8, ha='center', va='bottom', color='darkred',
                            bbox=dict(boxstyle='round,pad=0.3', fc='yellow', edgecolor='red', alpha=0.7))

        ax.set_title(f"{title} - Fold {fold_idx}", fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3); ax.legend(loc='upper left')

    plot_one(axes[0], bin_stats_orig, 'Original', ece_orig, mcs_orig, mce_orig, ace_orig, f1_orig)
    plot_one(axes[1], bin_stats_cal, 'Calibrated', ece_cal, mcs_cal, mce_cal, ace_cal, f1_cal)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path); plt.close(fig)
    else:
        plt.show()

# ============================================
# Segment Processing (FULL DETAILS RESTORED)
# ============================================
def process_segment(segment_name, y_true_file, logit_file_pattern, n_fold=10, output_dir="results", save_path=None, is_clade_h1n1_path=None, is_h5nx=False):
    seg_dir = os.path.join(output_dir, segment_name)
    os.makedirs(seg_dir, exist_ok=True)
    orig_stdout = sys.stdout
    sys.stdout = Logger(os.path.join(seg_dir, f"log_{segment_name}.txt"))
    
    stats_summary = {}
    class_wise_fold_metrics_orig = []
    class_wise_fold_metrics_cal = []
    bin_stats_all_folds = {}

    try:
        print(f"\n{'#'*80}\nPROCESSING SEGMENT: {segment_name}\n{'#'*80}")
        y_true = np.load(y_true_file)
        
        all_f1_orig, all_f1_cal, all_f1_orig_without_clade_h1h1 = [], [], []
        all_bal_acc_orig, all_bal_acc_cal, all_bal_acc_orig_without_clade_h1n1 = [], [], []
        all_mcc_orig, all_mcc_cal, all_mcc_orig_without_clade_h1n1 = [], [], []
        all_ece_orig, all_ece_cal = [], []
        all_mcs_orig, all_mcs_cal = [], []
        all_mce_orig, all_mce_cal = [], []
        all_ace_orig, all_ace_cal = [], []
        all_probs_uncal, all_probs_cal = [], []

        
        for i in range(n_fold):
            try:
                print(f"\n{'='*120}\nFOLD {i}\n{'='*120}")
                logits = np.load(logit_file_pattern.format(i))
                probs_uncal = softmax(logits)
                preds_uncal = np.argmax(probs_uncal, axis=1)

                ###### H5NX ###########
                if is_h5nx:
                    h5nx_logits = np.load(f'/home/hnguyen/Documents/PhD/Part2/Paper_02_Preparation/data/section1/{segment_name}/H5Nx_North_America_logits_fold_{i}.npy')
                    h5nx_probs_uncal = softmax(h5nx_logits)
                
                ece_orig, mce_orig, ace_orig, mcs_orig, stats_orig = compute_calibration_metrics(probs_uncal, y_true)
                f1_orig = f1_score(y_true, preds_uncal, average='macro')
                bal_acc_orig = balanced_accuracy_score(y_true, preds_uncal)
                mcc_orig = matthews_corrcoef(y_true, preds_uncal)
                
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                probs_cal_cv = np.zeros_like(probs_uncal)
                for tr, val in skf.split(probs_uncal, y_true):
                    iso = IsotonicCalibrator().fit(probs_uncal[tr], y_true[tr])
                    probs_cal_cv[val] = iso.predict_proba(probs_uncal[val])
                    
                    ###### H5NX ###########
                    if is_h5nx:
                        h5nx_probs_cal = iso.predict_proba(h5nx_probs_uncal)
                        h5nx_preds_cal = np.argmax(h5nx_probs_cal, axis=1)
                        class_counts = Counter(h5nx_preds_cal)
                        total = sum(class_counts.values())
                    
                        print("Predicted class counts (H5Nx subset):")
                        for cls, cnt in sorted(class_counts.items()):
                            pct = cnt / total * 100
                            print(f"Class {cls}: {cnt} ({pct:.1f}%)")
                
                print(f"{'Metric':<15} | {'F1 Orig':<9} | {'F1 Calib':<9} | {'ECE Orig':<9} | {'ECE Calib':<9} | {'MCS Orig':<9} | {'MCS Calib':<9} | {'MCE Orig':<9} | {'MCE Calib':<9} | {'ACE Orig':<9} | {'ACE Calib':<9}")
                print("-" * 120)
                
                probs_cal = probs_cal_cv
                preds_cal = np.argmax(probs_cal, axis=1)

                if save_path is not None:
                    np.save(f'{save_path}/{segment_name}_preds_cal_fold_{i}.npy', preds_cal)
                
                ece_cal, mce_cal, ace_cal, mcs_cal, stats_cal = compute_calibration_metrics(probs_cal, y_true)

                f1_cal = f1_score(y_true, preds_cal, average='macro')
                bal_acc_cal = balanced_accuracy_score(y_true, preds_cal)
                mcc_cal = matthews_corrcoef(y_true, preds_cal)

                ### Model performance without h1n1 clade ########
                if is_clade_h1n1_path is not None:
                    
                    is_clade_h1n1 = np.load(is_clade_h1n1_path)

                    y_true_without_clade_h1n1       = y_true[~is_clade_h1n1]
                    preds_uncal_without_clade_h1n1  = preds_uncal[~is_clade_h1n1]
                    
                    f1_cal_without_clade_h1n1       = f1_score(y_true_without_clade_h1n1, preds_uncal_without_clade_h1n1, average='macro')
                    bal_acc_cal_without_clade_h1n1  = balanced_accuracy_score(y_true_without_clade_h1n1, preds_uncal_without_clade_h1n1)
                    mcc_cal_without_clade_h1n1      = matthews_corrcoef(y_true_without_clade_h1n1, preds_uncal_without_clade_h1n1)

                    all_f1_orig_without_clade_h1h1.append(f1_cal_without_clade_h1n1)
                    all_bal_acc_orig_without_clade_h1n1.append(bal_acc_cal_without_clade_h1n1)
                    all_mcc_orig_without_clade_h1n1.append(mcc_cal_without_clade_h1n1)
                    
                    
                
                all_f1_orig.append(f1_orig); all_f1_cal.append(f1_cal)
                all_bal_acc_orig.append(bal_acc_orig); all_bal_acc_cal.append(bal_acc_cal)
                all_mcc_orig.append(mcc_orig); all_mcc_cal.append(mcc_cal)
                all_ece_orig.append(ece_orig); all_ece_cal.append(ece_cal)
                all_mcs_orig.append(mcs_orig); all_mcs_cal.append(mcs_cal)
                all_mce_orig.append(mce_orig); all_mce_cal.append(mce_cal)
                all_ace_orig.append(ace_orig); all_ace_cal.append(ace_cal)
                all_probs_uncal.append(probs_uncal); all_probs_cal.append(probs_cal)
                
                bin_stats_all_folds[i] = {'original': stats_orig, 'calibrated': stats_cal, 
                                          'metrics_orig': {'mce': mce_orig}, 'metrics_cal': {'mce': mce_cal}}

                print(f"{'Values':<15} | {f1_orig:<9.4f} | {f1_cal:<9.4f} | {ece_orig:<9.4f} | {ece_cal:<9.4f} | {mcs_orig:<9.4f} | {mcs_cal:<9.4f} | {mce_orig:<9.4f} | {mce_cal:<9.4f} | {ace_orig:<9.4f} | {ace_cal:<9.4f}")
                print("-" * 120)
                print(f"{'Additional':<15} | Bal Acc: {bal_acc_orig:.4f}->{bal_acc_cal:.4f} | MCC: {mcc_orig:.4f}->{mcc_cal:.4f}")
                print("-" * 120)
                
                print_bin_details(stats_orig, f"Fold {i} - Original Bins (Top 10 Worst)",top_n=10)
                print_bin_details(stats_cal, f"Fold {i} - Calibrated Bins (Top 10 Worst)", top_n=10)

                plot_path = os.path.join(seg_dir, f"{segment_name}_fold_{i}_reliability.png")
                plot_fold_reliability_and_scores(i, stats_orig, ece_orig, mcs_orig, mce_orig, ace_orig, f1_orig,
                                                 stats_cal, ece_cal, mcs_cal, mce_cal, ace_cal, f1_cal, save_path=plot_path)

                print(f"\n{'='*80}\nCLASSIFICATION REPORTS - FOLD {i}\n{'='*80}")
                report_orig = classification_report(y_true, preds_uncal, digits=4).splitlines()
                report_cal = classification_report(y_true, preds_cal, digits=4).splitlines()
                print(f"{'ORIGINAL':<55} | {'CALIBRATED':<55}")
                print("-" * 115)
                for line_o, line_c in zip(report_orig, report_cal):
                    print(f"{line_o:<55} | {line_c:<55}")
                print("="*115 + "\n")

                ########################################Class-Wise#############################################
                print ("=============================CLASS-WISE BEFORE CALIBRATION==========================")
                c_orig_metrics = compute_class_wise_metrics(probs_uncal, y_true, n_bins=10, n_classes=3)
                class_wise_fold_metrics_orig.append({'orig': c_orig_metrics})
                print_class_wise_results(c_orig_metrics, ['Human', 'Avian', 'Mammal'])
                print("=============================CLASS-WISE AFTER CALIBRATION==========================")
                c_cal_metrics = compute_class_wise_metrics(probs_cal, y_true, n_bins=10, n_classes=3)
                class_wise_fold_metrics_cal.append({'cal': c_cal_metrics})
                print_class_wise_results(c_cal_metrics, ['Human', 'Avian', 'Mammal'])
                ###########################################################################################

            except Exception as e: print(f"Fold {i} Error: {e}")

        # --- FINAL SUMMARY & ENSEMBLE STATS ---
        if all_f1_orig:
            print("\n" + "="*80 + "\n FINAL STATISTICS ACROSS ALL FOLDS (Mean ± Std)\n" + "="*80)
            def print_row(name, orig, cal):
                print(f"{name:<25} | {np.mean(orig):.4f} ± {np.std(orig):.4f} | {np.mean(cal):.4f} ± {np.std(cal):.4f}")
            print_row("F1 Score (Macro)", all_f1_orig, all_f1_cal)
            print_row("Balanced Accuracy", all_bal_acc_orig, all_bal_acc_cal)
            print_row("MCC", all_mcc_orig, all_mcc_cal)
            print("-" * 80)
            print_row("ECE", all_ece_orig, all_ece_cal)
            print_row("MCS", all_mcs_orig, all_mcs_cal)
            print_row("MCE", all_mce_orig, all_mce_cal)
            print_row("ACE", all_ace_orig, all_ace_cal)
            
            # --- CALCULATE ENSEMBLE FOR GRID PLOT ---
            avg_uncal = np.mean(all_probs_uncal, axis=0)
            avg_cal = np.mean(all_probs_cal, axis=0)
            assert avg_uncal.shape[0]==y_true.shape[0], "Not match n sequences"
            assert avg_cal.shape[0]==y_true.shape[0], "Not match n sequences"
            assert np.allclose(np.sum(avg_uncal, axis=-1), 1.0, atol=1e-6), "Probs don't sum to 1!"
            assert np.allclose(np.sum(avg_cal, axis=-1), 1.0, atol=1e-6), "Probs don't sum to 1!"
            _, _, _, _, ens_stats_orig = compute_calibration_metrics(avg_uncal, y_true)
            _, _, _, _, ens_stats_cal = compute_calibration_metrics(avg_cal, y_true)
            stats_summary = {
                "f1_orig": all_f1_orig, "f1_cal": all_f1_cal, "f1_orig_without_clade_h1n1": all_f1_orig_without_clade_h1h1,
                "bal_acc_orig": all_bal_acc_orig, "bal_acc_cal": all_bal_acc_cal, "bal_acc_orig_without_clade_h1n1": all_bal_acc_orig_without_clade_h1n1,
                "mcc_orig": all_mcc_orig, "mcc_cal": all_mcc_cal, "mcc_orig_without_clade_h1n1":all_mcc_orig_without_clade_h1n1,
                "ece_orig": all_ece_orig, "ece_cal": all_ece_cal,
                "mcs_orig": all_mcs_orig, "mcs_cal": all_mcs_cal,
                "mce_orig": all_mce_orig, "mce_cal": all_mce_cal,
                "ace_orig": all_ace_orig, "ace_cal": all_ace_cal,
                "ens_stats_orig": ens_stats_orig,
                "ens_stats_cal": ens_stats_cal,
                "bin_stats": bin_stats_all_folds
            }
            
    except Exception as e: print(f"Segment Error: {e}")
    finally: sys.stdout.close(); sys.stdout = orig_stdout; print(f"Finished: {segment_name}")
    return stats_summary, class_wise_fold_metrics_orig, class_wise_fold_metrics_cal


