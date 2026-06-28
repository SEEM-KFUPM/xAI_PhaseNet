import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "xai-phasenet-matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from xai_phasenet.model import PhaseNet
from xai_phasenet.shap_utils import channel_shapley


DEFAULT_DATA_PATH = "datasets/x_test_5391ev_270425.pt"
DEFAULT_LABELS_PATH = "datasets/y_test_5391ev_270425.pt"
DEFAULT_NOISE_PATH = "datasets/x_noise_6480ev_270425.pt"
DEFAULT_SIGNAL_INDICES_PATH = "datasets/indices_signal_140425.npy"
DEFAULT_NOISE_INDICES_PATH = "datasets/indices_noise_140425.npy"
DEFAULT_OUTPUT_DIR = "output/figure_4_shap_analysis"
DEFAULT_N_TEST = 5000
DEFAULT_TRACE_LENGTH = 3001
DEFAULT_WIN = 3001
DEFAULT_HOP = 1
DEFAULT_NBINS = 40
DEFAULT_BIN_QMAX = 99.5
DEFAULT_DPI = 600
COMPONENT_STYLES = (
    (0, "E component", "blue"),
    (1, "N component", "green"),
    (2, "Z component", "red"),
)
FIGURE4_PANEL_TITLES = (
    "a) P-class SHAP (Signal windows)",
    "b) P-class SHAP (Noise windows)",
    "c) S-class SHAP (Signal windows)",
    "d) S-class SHAP (Noise windows)",
)

DEFAULT_MODEL_PATHS = (
    "xai_phasenet/original.pt.v2",
    "../../../FNOclass/PhaseNet-main/original.pt.v2",
)


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


def crop_trace(batch, trace_length):
    batch = np.asarray(batch)
    if batch.ndim != 3:
        raise ValueError(f"Expected waveform batch with 3 dimensions, got shape {batch.shape}")

    if batch.shape[-1] == 3:
        if batch.shape[1] < trace_length:
            raise ValueError(f"Trace length {batch.shape[1]} is shorter than requested {trace_length}")
        return batch[:, :trace_length, :].astype(np.float32, copy=False)

    if batch.shape[1] == 3:
        if batch.shape[2] < trace_length:
            raise ValueError(f"Trace length {batch.shape[2]} is shorter than requested {trace_length}")
        return np.transpose(batch[:, :, :trace_length], (0, 2, 1)).astype(np.float32, copy=False)

    raise ValueError(f"Expected one waveform axis to contain 3 components, got shape {batch.shape}")


def ensure_channel_first(arr):
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected one waveform with 2 dimensions, got shape {arr.shape}")
    if arr.shape[0] == 3:
        return arr
    if arr.shape[1] == 3:
        return arr.T
    raise ValueError(f"Expected one waveform axis to contain 3 components, got shape {arr.shape}")


def cache_suffix(n_test, trace_length, win, hop):
    return f"{n_test}_trace{trace_length}_win{win}_hop{hop}"


def default_cache_dir(output_dir):
    return Path(output_dir) / "shap_data"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce manuscript Figure 4: component-wise SHAP histograms for "
            "5,000 signal and 5,000 noise traces."
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
    parser.add_argument("--output-dir", "--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the combined PNG.")
    parser.add_argument(
        "--cache-dir",
        "--cache_dir",
        default=None,
        help="Directory for SHAP .npy caches; defaults to <output-dir>/shap_data.",
    )
    parser.add_argument("--n-test", "--n_test", type=int, default=DEFAULT_N_TEST, help="Signal count and noise count.")
    parser.add_argument(
        "--trace-length",
        "--trace_length",
        type=int,
        default=DEFAULT_TRACE_LENGTH,
        help="Manuscript crop length before SHAP.",
    )
    parser.add_argument("--win", type=int, default=DEFAULT_WIN, help="SHAP window length.")
    parser.add_argument("--hop", type=int, default=DEFAULT_HOP, help="SHAP window hop.")
    parser.add_argument("--nbins", type=int, default=DEFAULT_NBINS, help="Histogram bin count.")
    parser.add_argument(
        "--bin-qmax",
        "--bin_qmax",
        type=float,
        default=DEFAULT_BIN_QMAX,
        help="Percentile upper limit for manuscript-style histogram bins.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Inference device.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure resolution.")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore cached SHAP arrays.")
    return parser


def make_bins(*arrays, qmax=99.5, nbins=40):
    data = np.concatenate([np.asarray(array).reshape(-1) for array in arrays])
    hi = np.percentile(np.abs(data), qmax) if data.size else 1.0
    hi = max(float(hi), 1e-6)
    return np.linspace(0.0, hi, nbins)


def plot_hist(ax, data, title, nbins=40, qmax=99.5):
    bins = make_bins(data, qmax=qmax, nbins=nbins)
    for component_idx, label, color in COMPONENT_STYLES:
        ax.hist(data[:, component_idx], bins=bins, alpha=0.5, density=True, label=label, color=color)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(r"Absolute SHAP value ($|\phi|$)")
    ax.set_ylabel("Density")
    ax.legend()


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

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    model_path = find_model_path(args.model_path)
    print(f"Loading weights from {model_path}")
    model = PhaseNet(component_order="ENZ")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("Loading data...")
    x_olddata = load_array(args.data_path)
    y_olddata = load_array(args.labels_path)
    x_noise = load_array(args.noise_path)

    signal_indices = labels_to_signal_indices(y_olddata)
    x_signal = x_olddata[signal_indices]

    indices_signal = np.load(args.signal_indices_path)
    indices_noise = np.load(args.noise_indices_path)

    if len(indices_signal) < args.n_test or len(indices_noise) < args.n_test:
        raise ValueError("--n-test exceeds available signal or noise index count")

    x_signal_test = crop_trace(x_signal[indices_signal[: args.n_test]], args.trace_length)
    x_noise_test = crop_trace(x_noise[indices_noise[: args.n_test]], args.trace_length)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.cache_dir) if args.cache_dir is not None else default_cache_dir(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = cache_suffix(args.n_test, args.trace_length, args.win, args.hop)
    f_p_sig = out_dir / f"impP_sig_{suffix}.npy"
    f_s_sig = out_dir / f"impS_sig_{suffix}.npy"
    f_p_noi = out_dir / f"impP_noi_{suffix}.npy"
    f_s_noi = out_dir / f"impS_noi_{suffix}.npy"

    cache_files = (f_p_sig, f_s_sig, f_p_noi, f_s_noi)
    if not args.force_recompute and all(path.exists() for path in cache_files):
        print(f"Loading precomputed SHAP values for {args.n_test} samples...")
        impP_sig = np.load(f_p_sig)
        impS_sig = np.load(f_s_sig)
        impP_noi = np.load(f_p_noi)
        impS_noi = np.load(f_s_noi)
    else:
        print(
            f"Computing SHAP values for {args.n_test} signal and {args.n_test} noise samples "
            f"(trace_length={args.trace_length}, win={args.win}, hop={args.hop})..."
        )
        impP_sig, impS_sig = [], []
        impP_noi, impS_noi = [], []

        for i in tqdm(range(args.n_test), desc="Signal Windows"):
            arr = ensure_channel_first(x_signal_test[i])
            p, _ = channel_shapley(
                model, arr, win=args.win, hop=args.hop, cls="P", agg="max", baseline="zero", device=device
            )
            s, _ = channel_shapley(
                model, arr, win=args.win, hop=args.hop, cls="S", agg="max", baseline="zero", device=device
            )
            impP_sig.append(np.abs(p))
            impS_sig.append(np.abs(s))

        for i in tqdm(range(args.n_test), desc="Noise Windows"):
            arr = ensure_channel_first(x_noise_test[i])
            p, _ = channel_shapley(
                model, arr, win=args.win, hop=args.hop, cls="P", agg="max", baseline="zero", device=device
            )
            s, _ = channel_shapley(
                model, arr, win=args.win, hop=args.hop, cls="S", agg="max", baseline="zero", device=device
            )
            impP_noi.append(np.abs(p))
            impS_noi.append(np.abs(s))

        impP_sig = np.array(impP_sig)
        impS_sig = np.array(impS_sig)
        impP_noi = np.array(impP_noi)
        impS_noi = np.array(impS_noi)

        np.save(f_p_sig, impP_sig)
        np.save(f_s_sig, impS_sig)
        np.save(f_p_noi, impP_noi)
        np.save(f_s_noi, impS_noi)

    print("Plotting histograms...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=args.dpi)

    panel_data = (impP_sig, impP_noi, impS_sig, impS_noi)
    for ax, data, title in zip(axes.ravel(), panel_data, FIGURE4_PANEL_TITLES):
        plot_hist(ax, data, title, args.nbins, args.bin_qmax)

    plt.tight_layout()
    out_img = output_dir / "Figure4_SHAP_histograms.png"
    plt.savefig(out_img, dpi=args.dpi)
    plt.close()
    print(f"Saved {out_img}")


if __name__ == "__main__":
    main()
