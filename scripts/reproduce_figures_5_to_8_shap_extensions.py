import argparse
import csv
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

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from xai_phasenet.model import PhaseNet


DEFAULT_DATA_PATH = "datasets/x_test_5391ev_270425.pt"
DEFAULT_LABELS_PATH = "datasets/y_test_5391ev_270425.pt"
DEFAULT_NOISE_PATH = "datasets/x_noise_6480ev_270425.pt"
DEFAULT_SIGNAL_INDICES_PATH = "datasets/indices_signal_140425.npy"
DEFAULT_NOISE_INDICES_PATH = "datasets/indices_noise_140425.npy"
DEFAULT_OUTPUT_DIR = "output/figures_5_to_8_shape_extensions"
DEFAULT_FIGURE4_CACHE_DIR = "output/figure_4_shap_analysis/shap_data"
DEFAULT_N_TEST = 5000
DEFAULT_TRACE_LENGTH = 3001
DEFAULT_WIN = 3001
DEFAULT_HOP = 1
DEFAULT_DPI = 600
DEFAULT_BATCH_SIZE = 64
DEFAULT_MODEL_PATHS = (
    "xai_phasenet/original.pt.v2",
    "../../../FNOclass/PhaseNet-main/original.pt.v2",
)
COMPONENT_LABELS = ("E", "N", "Z")
PAIR_LABELS = ("E-N", "E-Z", "N-Z")
PAIR_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")
FIGURE7_CONFIGS = (
    "3C (E, N, Z)",
    "2C (E, N)",
    "2C (E, Z)",
    "2C (N, Z)",
    "1C (N)",
    "1C (E)",
    "1C (Z)",
)
FIGURE7_KEEP_MASKS = {
    "3C (E, N, Z)": (1, 1, 1),
    "2C (E, N)": (1, 1, 0),
    "2C (E, Z)": (1, 0, 1),
    "2C (N, Z)": (0, 1, 1),
    "1C (N)": (0, 1, 0),
    "1C (E)": (1, 0, 0),
    "1C (Z)": (0, 0, 1),
}
FIGURE7_NOTEBOOK_MEANS = np.array([0.9641, 0.9716, 0.9662, 0.9555, 0.9474, 0.9455, 0.8819])
FIGURE7_NOTEBOOK_STDS = np.array([0.0092, 0.0022, 0.0080, 0.0042, 0.0012, 0.0066, 0.0167])


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
    batch = np.asarray(batch)
    if batch.ndim != 3:
        raise ValueError(f"Expected waveform batch with 3 dimensions, got shape {batch.shape}")
    if batch.shape[-1] == 3:
        return batch.astype(np.float32, copy=False)
    if batch.shape[1] == 3:
        return np.transpose(batch, (0, 2, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Expected waveform batch shape (N,T,3) or (N,3,T), got {batch.shape}")


def ensure_channel_first(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected one waveform with 2 dimensions, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"Expected waveform shape (3,T) or (T,3), got {arr.shape}")


def crop_trace(batch, trace_length):
    if trace_length == 0:
        return batch.astype(np.float32, copy=False)
    if batch.shape[1] < trace_length:
        raise ValueError(f"Requested trace length {trace_length}, but waveform length is {batch.shape[1]}.")
    return batch[:, :trace_length, :].astype(np.float32, copy=False)


def cache_suffix(n_test, trace_length, win, hop):
    return f"{n_test}_trace{trace_length}_win{win}_hop{hop}"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce manuscript Figures 5-8 from the SHAP extension analysis: "
            "violin distributions, pairwise interactions, reduced-component performance, "
            "and SHAP dispersion versus SNR."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", "--data_path", default=DEFAULT_DATA_PATH, help="Test waveform tensor.")
    parser.add_argument("--labels-path", "--labels_path", default=DEFAULT_LABELS_PATH, help="Test labels tensor.")
    parser.add_argument("--noise-path", "--noise_path", default=DEFAULT_NOISE_PATH, help="Pure-noise waveform tensor.")
    parser.add_argument(
        "--signal-indices-path",
        "--signal_indices_path",
        default=DEFAULT_SIGNAL_INDICES_PATH,
        help="Manuscript signal ordering file.",
    )
    parser.add_argument(
        "--noise-indices-path",
        "--noise_indices_path",
        default=DEFAULT_NOISE_INDICES_PATH,
        help="Manuscript noise ordering file.",
    )
    parser.add_argument("--model-path", "--model_path", default=None, help="PhaseNet weights; auto-detected if omitted.")
    parser.add_argument("--output-dir", "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for figures and data.")
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        help="Directory for this script's SHAP caches; defaults to <output-dir>/shap_data.",
    )
    parser.add_argument(
        "--figure4-cache-dir",
        "--figure4_cache_dir",
        default=DEFAULT_FIGURE4_CACHE_DIR,
        help="Existing Figure 4 SHAP cache directory to reuse when compatible.",
    )
    parser.add_argument("--n-test", "--n_test", type=int, default=DEFAULT_N_TEST, help="Signal count and noise count.")
    parser.add_argument("--trace-length", "--trace_length", type=int, default=DEFAULT_TRACE_LENGTH, help="Trace crop length.")
    parser.add_argument("--win", type=int, default=DEFAULT_WIN, help="SHAP window length.")
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP, help="SHAP window hop.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure resolution.")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for Figure 7 recomputation.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cached SHAP/interactions.")
    parser.add_argument(
        "--recompute-figure7",
        action="store_true",
        help="Recompute the reduced-component ablation instead of using the included manuscript summary values.",
    )
    parser.add_argument("--n-splits", "--n_splits", type=int, default=5, help="Figure 7 recomputation split count.")
    parser.add_argument(
        "--train-per-class",
        "--train_per_class",
        type=int,
        default=50,
        help="Figure 7 recomputation training windows per class.",
    )
    parser.add_argument(
        "--test-per-class",
        "--test_per_class",
        type=int,
        default=4500,
        help="Figure 7 recomputation test windows per class.",
    )
    return parser


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a CUDA device.")
    return torch.device(device_name)


def select_manuscript_dataset(args):
    print("Loading data...")
    x_olddata = ensure_time_channel(load_array(args.data_path))
    y_olddata = load_array(args.labels_path)
    x_noise = ensure_time_channel(load_array(args.noise_path))

    signal_source_indices = labels_to_signal_indices(y_olddata)
    if len(signal_source_indices) == 0:
        raise ValueError("No signal windows found in labels. Expected nonzero labels for signals.")

    x_signal_pool = x_olddata[signal_source_indices]
    indices_signal = np.load(args.signal_indices_path)
    indices_noise = np.load(args.noise_indices_path)

    if len(indices_signal) < args.n_test or len(indices_noise) < args.n_test:
        raise ValueError("--n-test exceeds available signal or noise index count.")

    x_signal = crop_trace(x_signal_pool[indices_signal[: args.n_test]], args.trace_length)
    x_noise_selected = crop_trace(x_noise[indices_noise[: args.n_test]], args.trace_length)
    x_combined = np.concatenate([x_signal, x_noise_selected], axis=0).astype(np.float32, copy=False)
    y_combined = np.concatenate(
        [np.ones(len(x_signal), dtype=np.int64), np.zeros(len(x_noise_selected), dtype=np.int64)]
    )

    print(f"Selected {len(x_signal)} signal and {len(x_noise_selected)} noise windows.")
    return x_signal, x_noise_selected, x_combined, y_combined


def make_windows(arr_enZ, win=3001, hop=1):
    arr_enZ = ensure_channel_first(arr_enZ)
    if arr_enZ.shape[1] < win:
        raise ValueError(f"Window length {win} exceeds trace length {arr_enZ.shape[1]}.")
    starts = np.arange(0, arr_enZ.shape[1] - win + 1, hop, dtype=int)
    return np.stack([arr_enZ[:, start : start + win] for start in starts], axis=0).astype(np.float32)


def mask_channels(x, keep, baseline="zero"):
    if baseline != "zero":
        raise ValueError("Figures 5-8 use the manuscript zero baseline.")
    m = np.asarray(keep, dtype=x.dtype).reshape(1, 3, 1)
    return m * x


def score_p_s(model, batch_np, device):
    x = torch.from_numpy(batch_np.astype(np.float32, copy=False)).to(device)
    with torch.no_grad():
        prob = model(x)
        p_score = prob[:, 1, :].max(dim=1).values
        s_score = prob[:, 2, :].max(dim=1).values
    return p_score.detach().cpu().numpy(), s_score.detach().cpu().numpy()


def shapley_from_values(values):
    v000, v001, v010, v011, v100, v101, v110, v111 = [values[:, i] for i in range(8)]

    phi_e = (1 / 3) * (v100 - v000) + (1 / 6) * (v110 - v010) + (1 / 6) * (v101 - v001) + (1 / 3) * (v111 - v011)
    phi_n = (1 / 3) * (v010 - v000) + (1 / 6) * (v110 - v100) + (1 / 6) * (v011 - v001) + (1 / 3) * (v111 - v101)
    phi_z = (1 / 3) * (v001 - v000) + (1 / 6) * (v101 - v100) + (1 / 6) * (v011 - v010) + (1 / 3) * (v111 - v110)
    phi = np.stack([phi_e, phi_n, phi_z], axis=1)
    imp = np.abs(phi).mean(axis=0)

    i_en = 0.25 * ((v110 - v100 - v010 + v000) + (v111 - v101 - v011 + v001))
    i_ez = 0.25 * ((v101 - v100 - v001 + v000) + (v111 - v110 - v011 + v010))
    i_nz = 0.25 * ((v011 - v010 - v001 + v000) + (v111 - v110 - v101 + v100))
    interactions = np.stack([i_en, i_ez, i_nz], axis=1)
    mean_interactions = interactions.mean(axis=0)

    return imp, phi, mean_interactions, interactions


def channel_shapley_interactions(model, arr_enZ, win, hop, device):
    windows = make_windows(arr_enZ, win=win, hop=hop)
    masks = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )

    p_values, s_values = [], []
    for mask in masks:
        p_score, s_score = score_p_s(model, mask_channels(windows, mask), device)
        p_values.append(p_score)
        s_values.append(s_score)

    p_values = np.stack(p_values, axis=1)
    s_values = np.stack(s_values, axis=1)

    imp_p, _, mean_int_p, _ = shapley_from_values(p_values)
    imp_s, _, mean_int_s, _ = shapley_from_values(s_values)
    return imp_p, imp_s, mean_int_p, mean_int_s


def product_cache_paths(cache_dir, suffix):
    return {
        "impP_sig": cache_dir / f"impP_sig_{suffix}.npy",
        "impS_sig": cache_dir / f"impS_sig_{suffix}.npy",
        "impP_noi": cache_dir / f"impP_noi_{suffix}.npy",
        "impS_noi": cache_dir / f"impS_noi_{suffix}.npy",
        "intP_all": cache_dir / f"mean_intP_all_{suffix}.npy",
        "intS_all": cache_dir / f"mean_intS_all_{suffix}.npy",
    }


def load_importance_from_figure4(args, suffix):
    if args.figure4_cache_dir is None:
        return None
    fig4_dir = Path(args.figure4_cache_dir)
    paths = product_cache_paths(fig4_dir, suffix)
    keys = ("impP_sig", "impS_sig", "impP_noi", "impS_noi")
    if all(paths[key].exists() for key in keys):
        print(f"Reusing compatible Figure 4 SHAP cache from {fig4_dir}")
        return {key: np.load(paths[key]) for key in keys}
    return None


def compute_or_load_shap_products(args, model, x_signal, x_noise, x_combined, cache_dir, suffix, device):
    paths = product_cache_paths(cache_dir, suffix)
    required = tuple(paths)
    if not args.force_recompute and all(paths[key].exists() for key in required):
        print(f"Loading cached Figures 5-8 SHAP products from {cache_dir}")
        return {key: np.load(paths[key]) for key in required}

    products = {}
    figure4_importance = None if args.force_recompute else load_importance_from_figure4(args, suffix)
    if figure4_importance is not None:
        products.update(figure4_importance)

    needs_compute = args.force_recompute or "intP_all" not in products or not paths["intP_all"].exists() or not paths["intS_all"].exists()
    if not needs_compute and all(key in products or paths[key].exists() for key in required):
        for key in required:
            if key not in products:
                products[key] = np.load(paths[key])
        return products

    print(
        "Computing SHAP importances and pairwise interactions "
        f"for {len(x_combined)} windows (trace_length={args.trace_length}, win={args.win}, hop={args.hop})..."
    )

    imp_p_all, imp_s_all, int_p_all, int_s_all = [], [], [], []
    for waveform in tqdm(x_combined, desc="SHAP interactions"):
        imp_p, imp_s, mean_int_p, mean_int_s = channel_shapley_interactions(
            model, ensure_channel_first(waveform), win=args.win, hop=args.hop, device=device
        )
        imp_p_all.append(np.abs(imp_p))
        imp_s_all.append(np.abs(imp_s))
        int_p_all.append(mean_int_p)
        int_s_all.append(mean_int_s)

    imp_p_all = np.asarray(imp_p_all)
    imp_s_all = np.asarray(imp_s_all)
    int_p_all = np.asarray(int_p_all)
    int_s_all = np.asarray(int_s_all)
    n_signal = len(x_signal)

    products = {
        "impP_sig": imp_p_all[:n_signal],
        "impS_sig": imp_s_all[:n_signal],
        "impP_noi": imp_p_all[n_signal:],
        "impS_noi": imp_s_all[n_signal:],
        "intP_all": int_p_all,
        "intS_all": int_s_all,
    }

    for key, value in products.items():
        np.save(paths[key], value)

    return products


def add_panel_label(ax, label):
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")


def plot_violin_panel(ax, data, title, panel_label):
    parts = ax.violinplot([data[:, i] for i in range(3)], showmedians=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#9ec3e6")
        body.set_edgecolor("#9ec3e6")
        body.set_alpha(0.65)
    if "cmedians" in parts:
        parts["cmedians"].set_color("#1f77b4")
        parts["cmedians"].set_linewidth(1.2)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(COMPONENT_LABELS)
    ax.set_ylabel("|Shapley|")
    ax.set_title(title, fontsize=10)
    add_panel_label(ax, panel_label)


def plot_figure5(products, output_dir, dpi):
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.5), dpi=dpi)
    panels = (
        (products["impP_sig"], "P-class |Shapley| distribution (signal)", "a)"),
        (products["impP_noi"], "P-class |Shapley| distribution (noise)", "b)"),
        (products["impS_sig"], "S-class |Shapley| distribution (signal)", "c)"),
        (products["impS_noi"], "S-class |Shapley| distribution (noise)", "d)"),
    )
    for ax, (data, title, label) in zip(axes.ravel(), panels):
        plot_violin_panel(ax, data, title, label)
    fig.tight_layout(w_pad=2.0, h_pad=2.0)
    out_path = output_dir / "Figure5_SHAP_violins.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_figure6(products, output_dir, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True, dpi=dpi)
    for ax, data, title in (
        (axes[0], products["intP_all"], "(a) P-class Pairwise Interactions"),
        (axes[1], products["intS_all"], "(b) S-class Pairwise Interactions"),
    ):
        for idx, (label, color) in enumerate(zip(PAIR_LABELS, PAIR_COLORS)):
            ax.hist(data[:, idx], bins=50, alpha=0.6, density=True, label=label, color=color, histtype="stepfilled", edgecolor="none")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, zorder=3)
        ax.set_title(title, fontsize=11, pad=8, loc="left", fontweight="bold")
        ax.set_xlabel("Shapley Interaction Index per sample")
        ax.grid(True, linestyle=":", alpha=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
        ax.legend(frameon=True, edgecolor="black", fancybox=False)
    axes[0].set_ylabel("Density")
    fig.tight_layout()
    out_path = output_dir / "Figure6_SHAP_pairwise_interactions.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def stratified_sample(sig_pool, noi_pool, n_sig, n_noi, rng):
    sig_sel = rng.choice(sig_pool, size=n_sig, replace=False)
    noi_sel = rng.choice(noi_pool, size=n_noi, replace=False)
    return np.concatenate([sig_sel, noi_sel])


def manuscript_prob_scores_batch(model, x_time_channel, device, batch_size):
    scores = []
    x_time_channel = x_time_channel.astype(np.float32, copy=False)
    for start in tqdm(range(0, len(x_time_channel), batch_size), desc="PROB scores"):
        batch = x_time_channel[start : start + batch_size]
        tensor = torch.from_numpy(np.transpose(batch, (0, 2, 1))).to(device)
        with torch.no_grad():
            prob = model(tensor)
            batch_scores = (1.0 - prob[:, 0, :]).max(dim=1).values
        scores.append(batch_scores.detach().cpu().numpy())
    return np.concatenate(scores)


def best_threshold(scores, labels, thresholds):
    best_f1, best_th = -1.0, None
    for threshold in thresholds:
        preds = (scores > threshold).astype(int)
        score = f1_score(labels, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(threshold)
    return best_f1, best_th


def recompute_figure7_ablation(args, model, x_combined, y_combined, device):
    sig_idx_all = np.where(y_combined == 1)[0]
    noi_idx_all = np.where(y_combined == 0)[0]
    if len(sig_idx_all) < args.train_per_class + args.test_per_class:
        raise ValueError("Not enough signal windows for Figure 7 recomputation.")
    if len(noi_idx_all) < args.train_per_class + args.test_per_class:
        raise ValueError("Not enough noise windows for Figure 7 recomputation.")

    thresholds = np.arange(1, 100) / 100.0
    rng_global = np.random.default_rng(12345)
    f1_mat = np.zeros((len(FIGURE7_CONFIGS), args.n_splits), dtype=float)
    precision_mat = np.zeros_like(f1_mat)
    recall_mat = np.zeros_like(f1_mat)

    print("Recomputing Figure 7 reduced-component ablation...")
    for idx, config in enumerate(FIGURE7_CONFIGS):
        mask = np.asarray(FIGURE7_KEEP_MASKS[config], dtype=x_combined.dtype).reshape(1, 1, 3)
        x_masked = x_combined * mask
        scores = manuscript_prob_scores_batch(model, x_masked, device, args.batch_size)
        rng = np.random.default_rng(rng_global.integers(0, 2**32 - 1))
        for split in range(args.n_splits):
            train_idx = stratified_sample(sig_idx_all, noi_idx_all, args.train_per_class, args.train_per_class, rng)
            train_idx_set = set(train_idx.tolist())
            sig_rem = np.array([i for i in sig_idx_all if i not in train_idx_set])
            noi_rem = np.array([i for i in noi_idx_all if i not in train_idx_set])
            test_idx = stratified_sample(sig_rem, noi_rem, args.test_per_class, args.test_per_class, rng)

            _, threshold = best_threshold(scores[train_idx], y_combined[train_idx], thresholds)
            preds = (scores[test_idx] > threshold).astype(int)
            y_test = y_combined[test_idx]
            f1_mat[idx, split] = f1_score(y_test, preds, zero_division=0)
            precision_mat[idx, split] = precision_score(y_test, preds, zero_division=0)
            recall_mat[idx, split] = recall_score(y_test, preds, zero_division=0)

    means = f1_mat.mean(axis=1)
    stds = f1_mat.std(axis=1, ddof=1) if args.n_splits > 1 else np.zeros(len(FIGURE7_CONFIGS))
    return means, stds, f1_mat, precision_mat, recall_mat


def write_figure7_summary(output_dir, means, stds, source):
    path = output_dir / "figure7_reduced_component_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["configuration", "mean_f1", "std_f1", "source"])
        for config, mean, std in zip(FIGURE7_CONFIGS, means, stds):
            writer.writerow([config, f"{mean:.6f}", f"{std:.6f}", source])
    print(f"Saved {path}")


def plot_figure7(output_dir, dpi, means, stds, source):
    colors = ["#2ca02c"] + ["#1f77b4"] * 3 + ["#ff7f0e"] * 3
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    bars = ax.bar(
        FIGURE7_CONFIGS,
        means,
        yerr=stds,
        capsize=5,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.2,
        error_kw=dict(lw=1.2, capthick=1.2),
    )
    ax.set_ylim(0.85, 1.0)
    ax.set_ylabel("Mean F1 Score", fontsize=14, fontweight="bold")
    ax.set_title("Model Performance Across Reduced Component Configurations", fontsize=14, pad=15, fontweight="bold")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    ax.tick_params(axis="both", which="major", labelsize=14)
    for bar, mean_val, std_val in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std_val + 0.002, f"{mean_val:.3f}", ha="center", va="bottom", fontsize=14)

    legend_elements = [
        Patch(facecolor="#2ca02c", edgecolor="black", linewidth=1.2, alpha=0.8, label="3-Component (Baseline)"),
        Patch(facecolor="#1f77b4", edgecolor="black", linewidth=1.2, alpha=0.8, label="2-Component Systems"),
        Patch(facecolor="#ff7f0e", edgecolor="black", linewidth=1.2, alpha=0.8, label="1-Component Systems"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=14, frameon=True, edgecolor="black", fancybox=False)
    plt.xticks(rotation=15, ha="center")
    fig.tight_layout()
    out_path = output_dir / "Figure7_Reduced_Component_Performance.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path} ({source})")


def compute_snr(arr_enZ, arrival_idx=None, noise_len=200, signal_len=200):
    arr_enZ = ensure_channel_first(arr_enZ)
    trace_length = arr_enZ.shape[1]
    if arrival_idx is not None:
        noise_start = max(0, arrival_idx - noise_len)
        noise_end = arrival_idx
        signal_start = arrival_idx
        signal_end = min(trace_length, arrival_idx + signal_len)
    else:
        energy = np.sum(arr_enZ**2, axis=0)
        peak_energy_idx = int(np.argmax(energy))
        signal_start = min(peak_energy_idx, max(0, trace_length - signal_len))
        signal_end = signal_start + signal_len
        noise_start = 0
        noise_end = min(noise_len, trace_length)

    if noise_start == noise_end or signal_start == signal_end:
        return 0.0

    noise_window = arr_enZ[:, noise_start:noise_end]
    signal_window = arr_enZ[:, signal_start:signal_end]
    noise_rms = np.sqrt(np.mean(noise_window**2, axis=1)) + 1e-10
    signal_rms = np.sqrt(np.mean(signal_window**2, axis=1))
    snr_linear = signal_rms / noise_rms
    snr_db_components = 20 * np.log10(snr_linear + 1e-10)
    return float(np.mean(snr_db_components))


def write_figure5_summary(products, output_dir):
    path = output_dir / "figure5_shap_violin_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "component", "mean", "median", "p05", "p95"])
        for panel, key in (
            ("P_signal", "impP_sig"),
            ("P_noise", "impP_noi"),
            ("S_signal", "impS_sig"),
            ("S_noise", "impS_noi"),
        ):
            data = products[key]
            for idx, component in enumerate(COMPONENT_LABELS):
                values = data[:, idx]
                writer.writerow(
                    [
                        panel,
                        component,
                        f"{np.mean(values):.6f}",
                        f"{np.median(values):.6f}",
                        f"{np.percentile(values, 5):.6f}",
                        f"{np.percentile(values, 95):.6f}",
                    ]
                )
    print(f"Saved {path}")


def write_figure6_summary(products, output_dir):
    path = output_dir / "figure6_pairwise_interactions_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["phase_class", "pair", "mean", "pct_synergy_gt_zero", "pct_redundancy_lt_zero"])
        for phase_class, key in (("P", "intP_all"), ("S", "intS_all")):
            data = products[key]
            for idx, pair in enumerate(PAIR_LABELS):
                values = data[:, idx]
                writer.writerow(
                    [
                        phase_class,
                        pair,
                        f"{np.mean(values):.6f}",
                        f"{100 * np.mean(values > 0):.3f}",
                        f"{100 * np.mean(values < 0):.3f}",
                    ]
                )
    print(f"Saved {path}")


def plot_figure8(products, x_combined, output_dir, dpi):
    imp_p_all = np.vstack([products["impP_sig"], products["impP_noi"]])
    imp_s_all = np.vstack([products["impS_sig"], products["impS_noi"]])
    combined_imp = np.hstack([imp_p_all, imp_s_all])
    d_shap = np.std(combined_imp, axis=1)
    snr = np.asarray([compute_snr(ensure_channel_first(waveform), arrival_idx=None) for waveform in tqdm(x_combined, desc="SNR proxy")])

    n_signal = len(products["impP_sig"])
    event_snr = snr[:n_signal]
    event_dispersion = d_shap[:n_signal]
    noise_snr = snr[n_signal:]
    noise_dispersion = d_shap[n_signal:]
    valid_noise_mask = noise_snr > -100
    noise_snr_clean = noise_snr[valid_noise_mask]
    noise_dispersion_clean = noise_dispersion[valid_noise_mask]

    csv_path = output_dir / "figure8_dispersion_vs_snr.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_type", "snr_db", "d_shap"])
        for value_snr, value_dispersion in zip(event_snr, event_dispersion):
            writer.writerow(["event", f"{value_snr:.6f}", f"{value_dispersion:.6f}"])
        for value_snr, value_dispersion in zip(noise_snr_clean, noise_dispersion_clean):
            writer.writerow(["pure_noise", f"{value_snr:.6f}", f"{value_dispersion:.6f}"])
    print(f"Saved {csv_path}")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=dpi)
    ax.scatter(noise_snr_clean, noise_dispersion_clean, color="red", marker="x", alpha=0.35, s=20, linewidths=0.6, label="Pure Noise")
    ax.scatter(event_snr, event_dispersion, color="blue", marker="o", alpha=0.4, s=20, edgecolors="none", label="Seismic Events")
    ax.set_xlabel("Signal-to-Noise Ratio (dB)")
    ax.set_ylabel(r"SHAP Dispersion ($D_{SHAP}$)")
    ax.grid(True, linestyle="--", alpha=0.3, linewidth=0.5)
    ax.legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right")
    out_path = output_dir / "Figure8_SHAP_dispersion_vs_snr.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {out_path}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_test < 1:
        raise ValueError("--n-test must be at least 1")
    if args.trace_length < 1:
        raise ValueError("--trace-length must be at least 1")
    if args.win < 1:
        raise ValueError("--win must be at least 1")
    if args.hop < 1:
        raise ValueError("--hop must be at least 1")
    if args.win > args.trace_length:
        raise ValueError("--win cannot be longer than --trace-length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir is not None else output_dir / "shap_data"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_path = find_model_path(args.model_path)
    print(f"Loading weights from {model_path}")
    model = PhaseNet(component_order="ENZ")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    x_signal, x_noise, x_combined, y_combined = select_manuscript_dataset(args)
    suffix = cache_suffix(args.n_test, args.trace_length, args.win, args.hop)
    products = compute_or_load_shap_products(args, model, x_signal, x_noise, x_combined, cache_dir, suffix, device)

    write_figure5_summary(products, output_dir)
    write_figure6_summary(products, output_dir)
    plot_figure5(products, output_dir, args.dpi)
    plot_figure6(products, output_dir, args.dpi)

    if args.recompute_figure7:
        means, stds, f1_mat, precision_mat, recall_mat = recompute_figure7_ablation(
            args, model, x_combined, y_combined, device
        )
        np.savez(
            output_dir / "figure7_reduced_component_recomputed_metrics.npz",
            configs=np.asarray(FIGURE7_CONFIGS),
            f1=f1_mat,
            precision=precision_mat,
            recall=recall_mat,
        )
        figure7_source = "recomputed"
    else:
        means = FIGURE7_NOTEBOOK_MEANS.copy()
        stds = FIGURE7_NOTEBOOK_STDS.copy()
        figure7_source = "manuscript_summary"

    write_figure7_summary(output_dir, means, stds, figure7_source)
    plot_figure7(output_dir, args.dpi, means, stds, figure7_source)
    plot_figure8(products, x_combined, output_dir, args.dpi)


if __name__ == "__main__":
    main()
