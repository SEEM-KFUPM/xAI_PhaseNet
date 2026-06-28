import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from xai_phasenet.gradcam import compute_gradcam
from xai_phasenet.model import PhaseNet


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


def ensure_channel_first(signal):
    if signal.shape[1] == 3:
        return signal.T
    return signal


def parse_x_range(value):
    start, end = value.split(",", maxsplit=1)
    return int(start), int(end)


def plot_heatmap(
    signal,
    localization_map,
    title,
    out_file,
    ppick=None,
    spick=None,
    xlim=None,
    dpi=600,
):
    """
    signal: (3, T) array
    localization_map: (T,) array
    """
    comp_names = ["E component", "N component", "Z component"]

    if xlim is not None:
        fig, axes = plt.subplots(
            3, 1, figsize=(10, 7), sharex=True, constrained_layout=True, dpi=dpi
        )
    else:
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 7), sharex=True, constrained_layout=True, dpi=dpi
        )

    localization_map = 1.0 - localization_map

    if xlim is not None:
        xmin, xmax = xlim
        xmin = max(0, xmin)
        xmax = min(signal.shape[1], xmax)
        if xmin >= xmax:
            raise ValueError(f"Invalid x-range {xlim} for signal length {signal.shape[1]}")
        heatmap_slice = np.uint8(255 * localization_map[xmin:xmax]).reshape(1, -1)
    else:
        xmin = 0
        xmax = signal.shape[1]
        heatmap_slice = np.uint8(255 * localization_map).reshape(1, -1)

    global_min = math.floor(signal[:, xmin:xmax].min())
    global_max = math.ceil(signal[:, xmin:xmax].max())

    if global_min == global_max:
        global_min -= 1
        global_max += 1

    for i, ax in enumerate(axes):
        sig_slice = signal[i, xmin:xmax]
        ymin = global_min
        ymax = global_max

        if ymin == ymax:
            ymin -= 1
            ymax += 1

        pcm2 = ax.imshow(
            heatmap_slice,
            cmap="Reds",
            aspect="auto",
            extent=[xmin, xmax, ymin, ymax],
            alpha=1.0,
            vmin=0,
            vmax=255,
        )

        ax.plot(np.arange(xmin, xmax), sig_slice, "k")

        ymin_ax, ymax_ax = ax.get_ylim()
        if ppick is not None:
            ax.vlines(ppick * 100, ymin_ax, ymax_ax, color="b", linewidth=1, label="P-arrival")
        if spick is not None:
            ax.vlines(spick * 100, ymin_ax, ymax_ax, color="g", linewidth=1, label="S-arrival")

        ax.grid(which="major", color="black", linestyle="-", linewidth=0.5)
        ax.grid(which="minor", color="gray", linestyle="--", linewidth=0.5)

        ax.set_title(comp_names[i])
        ax.set_ylabel("Amplitude")

    axes[-1].set_xlabel("Time steps")

    if ppick is not None or spick is not None:
        axes[0].legend(loc="upper right")

    cbar = fig.colorbar(pcm2, ax=axes, orientation="vertical", location="right", shrink=0.6)
    cbar.set_label("Prediction Intensity")

    fig.suptitle(title, fontweight="bold")

    plt.savefig(out_file, dpi=dpi)
    plt.close()
    print(f"Saved {out_file}")


def build_parser():
    parser = argparse.ArgumentParser(description="Reproduce PhaseNet Grad-CAM figures 2 and 3.")
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
    parser.add_argument("--model-path", "--model_path", default=None)
    parser.add_argument("--output-dir", "--output_dir", default="output/figures_2_and_3_gradcam")
    parser.add_argument("--n-test", "--n_test", type=int, default=50)
    parser.add_argument("--high-snr-index", type=int, default=0)
    parser.add_argument("--noise-index", type=int, default=0)
    parser.add_argument("--low-snr-source-index", type=int, default=3596)
    parser.add_argument("--high-p-pick", type=float, default=22.2320)
    parser.add_argument("--high-s-pick", type=float, default=23.0080)
    parser.add_argument("--low-p-pick", type=float, default=20.5275)
    parser.add_argument("--low-s-pick", type=float, default=21.1731)
    parser.add_argument("--high-zoom", type=parse_x_range, default=(2000, 2600))
    parser.add_argument("--low-zoom", type=parse_x_range, default=(1700, 2700))
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

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

    print(f"Loading data from {args.data_path} ...")
    x_olddata = load_array(args.data_path)
    y_olddata = load_array(args.labels_path)
    x_noise = load_array(args.noise_path)

    signal_indices = labels_to_signal_indices(y_olddata)
    x_signal = x_olddata[signal_indices]

    indices_signal = np.load(args.signal_indices_path)
    indices_noise = np.load(args.noise_indices_path)

    if args.n_test < 1:
        raise ValueError("--n-test must be at least 1")
    if len(indices_signal) < args.n_test or len(indices_noise) < args.n_test:
        raise ValueError("--n-test exceeds available signal or noise index count")

    x_signal_test = x_signal[indices_signal[: args.n_test]]
    x_noise_test = x_noise[indices_noise[: args.n_test]]

    if args.high_snr_index >= len(x_signal_test):
        raise IndexError("--high-snr-index is outside the selected signal set")
    if args.noise_index >= len(x_noise_test):
        raise IndexError("--noise-index is outside the selected noise set")
    if args.low_snr_source_index >= len(x_olddata):
        raise IndexError("--low-snr-source-index is outside the source data")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    noise_display_idx = args.n_test + args.noise_index
    samples_to_plot = [
        (
            "Figure2_High_SNR",
            x_signal_test[args.high_snr_index],
            args.high_snr_index,
            args.high_p_pick,
            args.high_s_pick,
            None,
        ),
        (
            "Figure2_Noise",
            x_noise_test[args.noise_index],
            noise_display_idx,
            None,
            None,
            None,
        ),
        (
            "Figure2_Low_SNR",
            x_olddata[args.low_snr_source_index],
            args.low_snr_source_index,
            args.low_p_pick,
            args.low_s_pick,
            None,
        ),
        (
            "Figure3_High_SNR_zoom",
            x_signal_test[args.high_snr_index],
            args.high_snr_index,
            args.high_p_pick,
            args.high_s_pick,
            args.high_zoom,
        ),
        (
            "Figure3_Low_SNR_zoom",
            x_olddata[args.low_snr_source_index],
            args.low_snr_source_index,
            args.low_p_pick,
            args.low_s_pick,
            args.low_zoom,
        ),
    ]

    for prefix, signal, idx, ppick, spick, xlim in samples_to_plot:
        print(f"Processing {prefix} (index {idx})...")
        signal = ensure_channel_first(signal)
        x_batch = torch.from_numpy(signal).float().unsqueeze(0).to(device)

        localization_map, _, _ = compute_gradcam(model, x_batch, target_class_idx=args.target_class)

        out_file = output_dir / f"{prefix}_idx{idx}.png"
        title = "Grad-CAM Heatmap PhaseNet (zoomed)" if xlim is not None else "Grad-CAM Heatmap PhaseNet"
        plot_heatmap(signal, localization_map, title, out_file, ppick=ppick, spick=spick, xlim=xlim, dpi=args.dpi)


if __name__ == "__main__":
    main()
