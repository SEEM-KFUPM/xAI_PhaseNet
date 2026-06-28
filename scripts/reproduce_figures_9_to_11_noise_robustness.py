import argparse
import csv
import hashlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "xai-phasenet-matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from xai_phasenet.inference import shap_scalar_single
from xai_phasenet.model import PhaseNet
from xai_phasenet.noise_utils import add_harmonic_pump_noise, add_white_noise


DEFAULT_MODEL_PATHS = (
    "xai_phasenet/original.pt.v2",
    "../../../FNOclass/PhaseNet-main/original.pt.v2",
)
DEFAULT_NOISE_BANK_PATH = "datasets/x_harmonic_5ev_stand_210925.pt"
DEFAULT_REL_AMPS = tuple(round(value / 10.0, 1) for value in range(10, 21))
DEFAULT_EXTENDED_REL_AMPS = tuple(round(value / 10.0, 1) for value in range(0, 51))


def load_array(path):
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.numpy()
    return np.asarray(data)


def find_model_path(model_path):
    if model_path is not None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model weights not found: {path}")
        return path

    for candidate in DEFAULT_MODEL_PATHS:
        path = Path(candidate)
        if path.exists():
            return path

    searched = ", ".join(DEFAULT_MODEL_PATHS)
    raise FileNotFoundError(f"Model weights not found. Checked: {searched}")


def labels_to_signal_indices(labels):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return np.where(labels != 0)[0]
    return np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]


def ensure_time_channel(batch):
    if batch.ndim != 3:
        raise ValueError(f"Expected a 3D waveform batch, got shape {batch.shape}")
    if batch.shape[2] == 3:
        return batch.astype(np.float32, copy=False)
    if batch.shape[1] == 3:
        return np.transpose(batch, (0, 2, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Expected waveform batch shape (N,T,3) or (N,3,T), got {batch.shape}")


def ensure_channel_first(arr):
    if arr.shape[0] == 3:
        return arr.astype(np.float32, copy=False)
    if arr.shape[1] == 3:
        return arr.T.astype(np.float32, copy=False)
    raise ValueError(f"Expected waveform shape (3,T) or (T,3), got {arr.shape}")


def parse_float_list(value):
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got: {value}") from exc


def parse_float_grid(value):
    if ":" not in value:
        return parse_float_list(value)

    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Range grids must use start:stop:step")

    try:
        start, stop, step = map(float, parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected numeric range values, got: {value}") from exc
    if step <= 0:
        raise argparse.ArgumentTypeError("Range grid step must be positive")

    values = np.arange(start, stop + step / 2.0, step, dtype=float)
    return [round(float(item), 10) for item in values]


def parse_threshold_grid(value):
    if ":" not in value:
        return np.asarray(parse_float_list(value), dtype=float)

    parts = [part.strip() for part in value.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Range grids must use start:stop:step")

    start, stop, step = map(float, parts)
    if step <= 0:
        raise argparse.ArgumentTypeError("Threshold grid step must be positive")
    return np.arange(start, stop + step / 2.0, step, dtype=float)


def stratified_sample(sig_pool, noi_pool, n_sig, n_noi, rng):
    sig_sel = rng.choice(sig_pool, size=n_sig, replace=False)
    noi_sel = rng.choice(noi_pool, size=n_noi, replace=False)
    return np.concatenate([sig_sel, noi_sel])


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but PyTorch cannot see a CUDA device.")
    return torch.device(device_name)


def score_cache_name(args, noise_type, rel_amp):
    rel = str(rel_amp).replace(".", "p")
    trace = "full" if args.trace_length == 0 else str(args.trace_length)
    source = noise_source_id(args, noise_type)
    return (
        f"scores_{noise_type}_{source}_rel{rel}_n{args.num_samples}_"
        f"trace{trace}_win{args.win}_hop{args.hop}_noiseseed{args.noise_seed}.npz"
    )


def noise_source_id(args, noise_type):
    if noise_type == "white":
        return "srcwhite"
    if args.noise_bank_path is None:
        return "srcsynthetic"
    digest = hashlib.sha1(str(Path(args.noise_bank_path)).encode("utf-8")).hexdigest()[:8]
    return f"srcbank{digest}"


def noise_source_label(args, noise_type):
    if noise_type == "white":
        return "white"
    if args.noise_bank_path is None:
        return "synthetic_harmonic"
    return "real_harmonic_bank"


def normalize_optional_path(value):
    if value is None:
        return None
    value = str(value)
    if value.strip().lower() in {"", "none", "null", "synthetic"}:
        return None
    return value


def validate_noise_bank(args, noise_types):
    args.noise_bank_path = normalize_optional_path(args.noise_bank_path)
    if "harmonic" not in noise_types or args.noise_bank_path is None:
        return

    path = Path(args.noise_bank_path)
    if not path.exists():
        raise FileNotFoundError(
            "Harmonic noise bank not found: "
            f"{path}. The manuscript reproduction default expects this file. "
            "Pass --noise-bank-path none to use the synthetic harmonic fallback."
        )


def crop_trace(batch, trace_length):
    if trace_length == 0:
        return batch
    if batch.shape[1] < trace_length:
        raise ValueError(
            f"Requested --trace-length {trace_length}, but waveform length is only {batch.shape[1]}."
        )
    return batch[:, :trace_length, :]


def manuscript_prob_score_single(model, arr_enZ_3T, device="cpu"):
    """
    Manuscript robustness PROB score: max_t(1 - P(noise, t)).
    With PhaseNet's softmax this is max_t(P(P, t) + P(S, t)).
    """
    x = torch.from_numpy(arr_enZ_3T[None, :, :].astype(np.float32)).to(device)
    with torch.no_grad():
        prob = model(x)
        score = (1.0 - prob[:, 0, :]).max(dim=1).values
    return float(score.item())


def compute_all_scores(model, x_data_ct, device, win, hop, baseline):
    """
    Computes probability scores and SHAP scalar values for a dataset shaped (N, C, T).
    """
    n_samples = x_data_ct.shape[0]
    if x_data_ct.shape[2] < win:
        raise ValueError(f"SHAP window --win {win} exceeds evaluated trace length {x_data_ct.shape[2]}.")
    prob_scores = np.zeros(n_samples)
    shap_scores = np.zeros(n_samples)

    for i in tqdm(range(n_samples), desc="Scoring windows"):
        arr = x_data_ct[i]
        prob_scores[i] = manuscript_prob_score_single(model, arr, device=device)
        shap_scores[i] = shap_scalar_single(
            model, arr, win=win, hop=hop, baseline=baseline, device=device
        )

    return prob_scores, shap_scores


def select_balanced_dataset(args):
    print(f"Loading signal data from {args.data_path}")
    x_data = ensure_time_channel(load_array(args.data_path))
    y_data = load_array(args.labels_path)

    signal_source_indices = labels_to_signal_indices(y_data)
    if len(signal_source_indices) == 0:
        raise ValueError("No signal windows found in labels. Expected nonzero labels for signals.")

    x_signal_pool = x_data[signal_source_indices]

    print(f"Loading noise data from {args.noise_path}")
    x_noise_pool = ensure_time_channel(load_array(args.noise_path))

    signal_index_path = Path(args.signal_indices_path)
    noise_index_path = Path(args.noise_indices_path)
    indices_signal = np.load(signal_index_path) if signal_index_path.exists() else np.arange(len(x_signal_pool))
    indices_noise = np.load(noise_index_path) if noise_index_path.exists() else np.arange(len(x_noise_pool))

    n_signal = min(args.num_samples, len(indices_signal), len(x_signal_pool))
    n_noise = min(args.num_samples, len(indices_noise), len(x_noise_pool))
    if n_signal < 2 or n_noise < 2:
        raise ValueError(
            "Need at least two signal and two noise windows for train/test splits. "
            f"Selected signal={n_signal}, noise={n_noise}."
        )

    x_signal = x_signal_pool[indices_signal[:n_signal]]
    x_noise = x_noise_pool[indices_noise[:n_noise]]
    x_selected = np.concatenate([x_signal, x_noise], axis=0).astype(np.float32, copy=False)
    x_selected = crop_trace(x_selected, args.trace_length)
    y_selected = np.concatenate(
        [np.ones(n_signal, dtype=np.int64), np.zeros(n_noise, dtype=np.int64)], axis=0
    )

    print(f"Selected {n_signal} signal and {n_noise} noise windows with length {x_selected.shape[1]}.")
    return x_selected, y_selected


def inject_noise(args, noise_type, x_time_channel, rel_amp, noise_bank):
    if noise_type == "harmonic":
        return add_harmonic_pump_noise(
            x_time_channel,
            fs=args.fs,
            freqs=args.harmonic_freqs,
            use_snr_db=False,
            rel_amp=rel_amp,
            per_component=True,
            random_seed=args.noise_seed,
            noise_bank=noise_bank,
            noise_fs=args.noise_fs,
            center_and_detrend=True,
        )

    return add_white_noise(
        x_time_channel,
        fs=args.fs,
        use_snr_db=False,
        rel_amp=rel_amp,
        per_component=True,
        random_seed=args.noise_seed,
    )


def best_threshold(scores, labels, thresholds):
    best_f1, best_th = -1.0, None
    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(threshold)
    return best_f1, best_th


def evaluate_splits(args, noise_type, y, prob_scores, shap_scores, rel_amp, rng):
    sig_idx_all = np.where(y == 1)[0]
    noi_idx_all = np.where(y == 0)[0]
    n_train_sig = min(args.train_per_class, len(sig_idx_all) // 2)
    n_train_noi = min(args.train_per_class, len(noi_idx_all) // 2)

    if n_train_sig < 1 or n_train_noi < 1:
        raise ValueError("Not enough signal/noise windows for cross-validation.")

    rows = []
    for split in range(args.n_splits):
        train_idx = stratified_sample(sig_idx_all, noi_idx_all, n_train_sig, n_train_noi, rng)
        train_idx_set = set(train_idx.tolist())
        sig_rem = np.array([i for i in sig_idx_all if i not in train_idx_set])
        noi_rem = np.array([i for i in noi_idx_all if i not in train_idx_set])
        if args.test_per_class is None:
            test_idx = np.concatenate([sig_rem, noi_rem])
        else:
            if len(sig_rem) < args.test_per_class or len(noi_rem) < args.test_per_class:
                raise ValueError(
                    "--test-per-class exceeds the remaining signal or noise windows "
                    f"after training selection: signal={len(sig_rem)}, noise={len(noi_rem)}."
                )
            test_idx = stratified_sample(sig_rem, noi_rem, args.test_per_class, args.test_per_class, rng)

        if len(test_idx) == 0:
            raise ValueError("No test windows remain after selecting train windows.")

        y_train = y[train_idx]
        y_test = y[test_idx]

        p_train, s_train = prob_scores[train_idx], shap_scores[train_idx]
        p_test, s_test = prob_scores[test_idx], shap_scores[test_idx]

        _, best_prob_th = best_threshold(p_train, y_train, args.prob_thresholds)
        _, best_shap_th = best_threshold(s_train, y_train, args.shap_thresholds)

        preds_p_test = (p_test > best_prob_th).astype(int)
        preds_s_test = (s_test > best_shap_th).astype(int)

        row = {
            "noise_type": noise_type,
            "rel_amp": rel_amp,
            "split": split + 1,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "prob_threshold": best_prob_th,
            "shap_threshold": best_shap_th,
            "f1_prob": f1_score(y_test, preds_p_test, zero_division=0),
            "f1_shap": f1_score(y_test, preds_s_test, zero_division=0),
            "precision_prob": precision_score(y_test, preds_p_test, zero_division=0),
            "precision_shap": precision_score(y_test, preds_s_test, zero_division=0),
            "recall_prob": recall_score(y_test, preds_p_test, zero_division=0),
            "recall_shap": recall_score(y_test, preds_s_test, zero_division=0),
        }
        rows.append(row)

        print(
            "  Split {split}: F1_prob={f1_prob:.4f} (th={prob_threshold:.2f}), "
            "F1_shap={f1_shap:.4f} (th={shap_threshold:.2f})".format(**row)
        )

    return rows


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for rel_amp in sorted({row["rel_amp"] for row in rows}):
        for noise_type in sorted({row["noise_type"] for row in rows}):
            subset = [row for row in rows if row["rel_amp"] == rel_amp and row["noise_type"] == noise_type]
            if not subset:
                continue
            summary.append(
                {
                    "noise_type": noise_type,
                    "rel_amp": rel_amp,
                    "splits": len(subset),
                    "n_test": int(round(np.mean([row["n_test"] for row in subset]))),
                    "mean_f1_prob": float(np.mean([row["f1_prob"] for row in subset])),
                    "std_f1_prob": sample_std([row["f1_prob"] for row in subset]),
                    "mean_f1_shap": float(np.mean([row["f1_shap"] for row in subset])),
                    "std_f1_shap": sample_std([row["f1_shap"] for row in subset]),
                    "mean_precision_prob": float(np.mean([row["precision_prob"] for row in subset])),
                    "mean_precision_shap": float(np.mean([row["precision_shap"] for row in subset])),
                    "mean_recall_prob": float(np.mean([row["recall_prob"] for row in subset])),
                    "mean_recall_shap": float(np.mean([row["recall_shap"] for row in subset])),
                }
            )
    return summary


def sample_std(values):
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def metric_matrices(rows):
    grouped = {}
    for noise_type in sorted({row["noise_type"] for row in rows}):
        noise_rows = [row for row in rows if row["noise_type"] == noise_type]
        rel_amps = sorted({row["rel_amp"] for row in noise_rows})
        splits = sorted({row["split"] for row in noise_rows})
        shape = (len(rel_amps), len(splits))
        matrices = {
            "rel_amp_values": np.asarray(rel_amps, dtype=float),
            "f1_prob": np.full(shape, np.nan),
            "f1_shap": np.full(shape, np.nan),
            "pr_prob": np.full(shape, np.nan),
            "pr_shap": np.full(shape, np.nan),
            "re_prob": np.full(shape, np.nan),
            "re_shap": np.full(shape, np.nan),
        }
        rel_lookup = {rel_amp: idx for idx, rel_amp in enumerate(rel_amps)}
        split_lookup = {split: idx for idx, split in enumerate(splits)}
        for row in noise_rows:
            i = rel_lookup[row["rel_amp"]]
            j = split_lookup[row["split"]]
            matrices["f1_prob"][i, j] = row["f1_prob"]
            matrices["f1_shap"][i, j] = row["f1_shap"]
            matrices["pr_prob"][i, j] = row["precision_prob"]
            matrices["pr_shap"][i, j] = row["precision_shap"]
            matrices["re_prob"][i, j] = row["recall_prob"]
            matrices["re_shap"][i, j] = row["recall_shap"]
        grouped[noise_type] = matrices
    return grouped


def write_metric_archives(output_dir, rows, tag=None):
    suffix = {"white": "wn", "harmonic": "hn"}
    for noise_type, matrices in metric_matrices(rows).items():
        archive_tag = "" if tag is None else f"_{tag}"
        archive_path = output_dir / f"cv_all_metrics_{suffix.get(noise_type, noise_type)}{archive_tag}.npz"
        np.savez(archive_path, **matrices)


def mean_std(matrix):
    matrix = np.asarray(matrix, dtype=float)
    mean = np.nanmean(matrix, axis=1)
    if matrix.shape[1] > 1:
        std = np.nanstd(matrix, axis=1, ddof=1)
    else:
        std = np.zeros(matrix.shape[0], dtype=float)
    return mean, std


def plot_metric_with_band(ax, rel_amp, matrix, marker, label):
    mean, std = mean_std(matrix)
    ax.plot(rel_amp, mean, marker=marker, label=label)
    ax.fill_between(rel_amp, mean - std, mean + std, alpha=0.2)


def plot_summary(path, summary_rows, dpi=600):
    import matplotlib.pyplot as plt

    label_name = {"white": "random", "harmonic": "harmonic"}
    noise_order = [name for name in ("harmonic", "white") if name in {row["noise_type"] for row in summary_rows}]
    noise_order += sorted({row["noise_type"] for row in summary_rows} - set(noise_order))
    split_count = max(row["splits"] for row in summary_rows)
    test_count = max(row.get("n_test", 0) for row in summary_rows)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    for noise_type in noise_order:
        rows = sorted([row for row in summary_rows if row["noise_type"] == noise_type], key=lambda row: row["rel_amp"])
        rel_amp = np.asarray([row["rel_amp"] for row in rows], dtype=float)
        prob_mean = np.asarray([row["mean_f1_prob"] for row in rows], dtype=float)
        prob_std = np.asarray([row["std_f1_prob"] for row in rows], dtype=float)
        shap_mean = np.asarray([row["mean_f1_shap"] for row in rows], dtype=float)
        shap_std = np.asarray([row["std_f1_shap"] for row in rows], dtype=float)
        name = label_name.get(noise_type, noise_type)

        ax.plot(rel_amp, prob_mean, marker="o", label=f"PROB {name}")
        ax.fill_between(rel_amp, prob_mean - prob_std, prob_mean + prob_std, alpha=0.2)

        ax.plot(rel_amp, shap_mean, marker="s", label=f"mean6 {name}")
        ax.fill_between(rel_amp, shap_mean - shap_std, shap_mean + shap_std, alpha=0.2)

    ax.set_xlabel("Noise relative amplitude")
    ylabel = "F1"
    if test_count:
        ylabel += f" (test on {test_count:g} balanced samples)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"F1 Score vs. Noise Relative Amplitude (mean±std over {split_count} CV splits)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_precision_recall(path, rows, dpi=600):
    import matplotlib.pyplot as plt

    matrices_by_noise = metric_matrices(rows)
    label_name = {"white": "random", "harmonic": "harmonic"}
    noise_order = [name for name in ("harmonic", "white") if name in matrices_by_noise]
    noise_order += sorted(set(matrices_by_noise) - set(noise_order))
    split_count = max(matrices["pr_prob"].shape[1] for matrices in matrices_by_noise.values())
    test_count = max(row.get("n_test", 0) for row in rows)
    test_label = f"{test_count:g} balanced samples" if test_count else "balanced samples"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=dpi, sharey=True)
    for noise_type in noise_order:
        matrices = matrices_by_noise[noise_type]
        rel_amp = matrices["rel_amp_values"]
        name = label_name.get(noise_type, noise_type)
        plot_metric_with_band(ax1, rel_amp, matrices["pr_prob"], "o", f"PROB {name}")
        plot_metric_with_band(ax1, rel_amp, matrices["pr_shap"], "s", f"mean6 {name}")
        plot_metric_with_band(ax2, rel_amp, matrices["re_prob"], "o", f"PROB {name}")
        plot_metric_with_band(ax2, rel_amp, matrices["re_shap"], "s", f"mean6 {name}")

    ax1.set_xlabel("Noise relative amplitude")
    ax1.set_ylabel(f"Precision (test on {test_label})")
    ax1.set_title(f"Precision vs. Noise Relative Amplitude (mean±std over {split_count} CV splits)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel("Noise relative amplitude")
    ax2.set_ylabel(f"Recall (test on {test_label})")
    ax2.set_title(f"Recall vs. Noise Relative Amplitude (mean±std over {split_count} CV splits)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_extended_f1(path, rows, dpi=600):
    import matplotlib.pyplot as plt

    matrices_by_noise = metric_matrices(rows)
    if not matrices_by_noise:
        raise ValueError("Figure 11 expects extended noise metrics.")

    noise_type = "harmonic" if "harmonic" in matrices_by_noise else sorted(matrices_by_noise)[0]
    matrices = matrices_by_noise[noise_type]
    rel_amp = matrices["rel_amp_values"]
    prob_mean, prob_std = mean_std(matrices["f1_prob"])
    shap_mean, shap_std = mean_std(matrices["f1_shap"])
    split_count = matrices["f1_prob"].shape[1]
    test_count = max(row.get("n_test", 0) for row in rows)
    test_label = f"{test_count:g} balanced samples" if test_count else "balanced samples"

    fig, ax = plt.subplots(figsize=(7, 5), dpi=dpi)
    ax.plot(rel_amp, prob_mean, marker="o", label="Prob ONLY (mean F1)")
    ax.fill_between(rel_amp, prob_mean - prob_std, prob_mean + prob_std, alpha=0.2)
    ax.plot(rel_amp, shap_mean, marker="s", label="mean6 ONLY (mean F1)")
    ax.fill_between(rel_amp, shap_mean - shap_std, shap_mean + shap_std, alpha=0.2)
    ax.set_xlabel("Noise relative amplitude")
    ax.set_ylabel(f"F1 (test on {test_label})")
    ax.set_title(f"F1 Score vs. Noise Relative Amplitude (mean±std over {split_count} CV splits)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Reproduce Figures 9-11 PhaseNet noise-robustness evaluation.")
    parser.add_argument("--model-path", "--model_path", default=None)
    parser.add_argument("--data-path", "--data_path", default="datasets/x_test_5391ev_270425.pt")
    parser.add_argument("--labels-path", "--labels_path", default="datasets/y_test_5391ev_270425.pt")
    parser.add_argument("--noise-path", "--noise_path", default="datasets/x_noise_6480ev_270425.pt")
    parser.add_argument(
        "--signal-indices-path",
        "--signal_indices_path",
        default="datasets/indices_signal_140425.npy",
    )
    parser.add_argument(
        "--noise-indices-path",
        "--noise_indices_path",
        default="datasets/indices_noise_140425.npy",
    )
    parser.add_argument("--noise-type", "--noise_type", choices=("white", "harmonic", "both"), default="both")
    parser.add_argument("--noise-bank-path", "--noise_bank_path", default=DEFAULT_NOISE_BANK_PATH)
    parser.add_argument("--noise-fs", "--noise_fs", type=float, default=100.0)
    parser.add_argument("--harmonic-freqs", type=parse_float_list, default=(20.0, 40.0))
    parser.add_argument("--rel-amps", "--rel_amps", type=parse_float_grid, default=DEFAULT_REL_AMPS)
    parser.add_argument(
        "--extended-rel-amps",
        "--extended_rel_amps",
        type=parse_float_grid,
        default=DEFAULT_EXTENDED_REL_AMPS,
        help="Relative amplitudes for the extended Figure 11 sweep.",
    )
    parser.add_argument(
        "--extended-noise-type",
        "--extended_noise_type",
        choices=("white", "harmonic"),
        default="harmonic",
        help="Noise source for the extended Figure 11 sweep.",
    )
    parser.add_argument("--skip-extended", action="store_true", help="Only run the standard Figures 9 and 10 sweep.")
    parser.add_argument("--output-dir", "--output_dir", default="output/figures_9_to_11_noise_robustness")
    parser.add_argument("--cache-dir", "--cache_dir", default=None)
    parser.add_argument(
        "--num-samples",
        "--num_samples",
        type=int,
        default=5000,
        help="Number of signal windows and number of noise windows to evaluate.",
    )
    parser.add_argument("--n-splits", "--n_splits", type=int, default=5)
    parser.add_argument("--train-per-class", "--train_per_class", type=int, default=50)
    parser.add_argument("--test-per-class", "--test_per_class", type=int, default=4500)
    parser.add_argument("--prob-thresholds", type=parse_threshold_grid, default="0.01:0.99:0.01")
    parser.add_argument("--shap-thresholds", type=parse_threshold_grid, default="0.01:0.29:0.01")
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument(
        "--trace-length",
        "--trace_length",
        type=int,
        default=3001,
        help="Number of leading time samples to evaluate; use 0 for the full trace.",
    )
    parser.add_argument("--win", type=int, default=3001)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--baseline", choices=("zero", "mean"), default="zero")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--noise-seed",
        "--noise_seed",
        type=int,
        default=42,
        help="Seed for waveform noise injection. The CV split seed is --seed.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--force-recompute", action="store_true")
    return parser


def run_noise_sweep(args, model, x_selected, y_selected, noise_types, rel_amps, noise_bank, cache_dir, device):
    all_rows = []
    for noise_type in noise_types:
        if noise_type == "harmonic" and args.noise_bank_path is None:
            print(
                "WARNING: harmonic noise is using the synthetic fallback because "
                "--noise-bank-path was not provided. This will not reproduce the "
                "manuscript's real pump-noise harmonic curve."
            )
        rng_global = np.random.default_rng(args.seed)
        for rel_amp in rel_amps:
            print(f"\n=== Evaluating {noise_type} noise at relative amplitude: {rel_amp:.2f} ===")
            cache_file = cache_dir / score_cache_name(args, noise_type, rel_amp)

            if cache_file.exists() and not args.force_recompute:
                print(f"Loading cached scores from {cache_file}")
                cached = np.load(cache_file)
                prob_scores = cached["prob_scores"]
                shap_scores = cached["shap_scores"]
            else:
                x_aug, _ = inject_noise(args, noise_type, x_selected, rel_amp, noise_bank)
                x_aug_ct = np.transpose(x_aug, (0, 2, 1))
                print("Computing probability and SHAP scores...")
                prob_scores, shap_scores = compute_all_scores(
                    model, x_aug_ct, device, args.win, args.hop, args.baseline
                )
                np.savez(
                    cache_file,
                    prob_scores=prob_scores,
                    shap_scores=shap_scores,
                    labels=y_selected,
                    rel_amp=rel_amp,
                    noise_type=noise_type,
                    noise_source=noise_source_label(args, noise_type),
                    noise_bank_path="" if args.noise_bank_path is None else args.noise_bank_path,
                    trace_length=x_selected.shape[1],
                    noise_seed=args.noise_seed,
                )

            rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
            all_rows.extend(evaluate_splits(args, noise_type, y_selected, prob_scores, shap_scores, rel_amp, rng))
    return all_rows


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.num_samples < 2:
        raise ValueError("--num-samples must be at least 2")
    if args.n_splits < 1:
        raise ValueError("--n-splits must be at least 1")
    if args.train_per_class < 1:
        raise ValueError("--train-per-class must be at least 1")
    if args.trace_length < 0:
        raise ValueError("--trace-length must be non-negative")

    noise_types = ("white", "harmonic") if args.noise_type == "both" else (args.noise_type,)
    required_noise_types = (
        noise_types if args.skip_extended else tuple(set(noise_types + (args.extended_noise_type,)))
    )
    validate_noise_bank(args, required_noise_types)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else output_dir / "score_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_path = find_model_path(args.model_path)

    print(f"Loading weights from {model_path}")
    model = PhaseNet(component_order="ENZ")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    x_selected, y_selected = select_balanced_dataset(args)

    noise_bank = None
    if args.noise_bank_path is not None:
        print(f"Loading noise bank from {args.noise_bank_path}")
        noise_bank = ensure_time_channel(load_array(args.noise_bank_path))

    print("\n=== Standard robustness sweep for Figures 9 and 10 ===")
    all_rows = run_noise_sweep(args, model, x_selected, y_selected, noise_types, args.rel_amps, noise_bank, cache_dir, device)

    result_fields = [
        "noise_type",
        "rel_amp",
        "split",
        "n_train",
        "n_test",
        "prob_threshold",
        "shap_threshold",
        "f1_prob",
        "f1_shap",
        "precision_prob",
        "precision_shap",
        "recall_prob",
        "recall_shap",
    ]
    results_path = output_dir / "noise_robustness_results.csv"
    write_csv(results_path, all_rows, result_fields)

    summary_rows = summarize(all_rows)
    summary_fields = list(summary_rows[0].keys())
    summary_path = output_dir / "noise_robustness_summary.csv"
    write_csv(summary_path, summary_rows, summary_fields)
    write_metric_archives(output_dir, all_rows)

    plot_path = output_dir / "Figure9_Noise_Robustness.png"
    plot_summary(plot_path, summary_rows, args.dpi)
    precision_recall_path = output_dir / "Figure10_Precision_Recall_Noise_Robustness.png"
    plot_precision_recall(precision_recall_path, all_rows, args.dpi)

    print(f"\nSaved split results to {results_path}")
    print(f"Saved summary to {summary_path}")
    print(f"Saved Figure 9 to {plot_path}")
    print(f"Saved Figure 10 to {precision_recall_path}")

    if args.skip_extended:
        return

    print(f"\n=== Extended {args.extended_noise_type}-noise sweep for Figure 11 ===")
    extended_noise_bank = noise_bank if args.extended_noise_type == "harmonic" else None
    extended_rows = run_noise_sweep(
        args,
        model,
        x_selected,
        y_selected,
        (args.extended_noise_type,),
        args.extended_rel_amps,
        noise_bank=extended_noise_bank,
        cache_dir=cache_dir,
        device=device,
    )

    extended_results_path = output_dir / "noise_robustness_extended_results.csv"
    write_csv(extended_results_path, extended_rows, result_fields)
    extended_summary_rows = summarize(extended_rows)
    extended_summary_path = output_dir / "noise_robustness_extended_summary.csv"
    write_csv(extended_summary_path, extended_summary_rows, list(extended_summary_rows[0].keys()))
    write_metric_archives(output_dir, extended_rows, tag="extended")

    extended_plot_path = output_dir / "Figure11_Extended_F1_Noise_Robustness.png"
    plot_extended_f1(extended_plot_path, extended_rows, args.dpi)

    print(f"Saved extended split results to {extended_results_path}")
    print(f"Saved extended summary to {extended_summary_path}")
    print(f"Saved Figure 11 to {extended_plot_path}")


if __name__ == "__main__":
    main()
