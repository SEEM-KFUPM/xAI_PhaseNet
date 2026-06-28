import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "xai-phasenet-matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from xai_phasenet.model import PhaseNet
from xai_phasenet.noise_utils import add_harmonic_pump_noise, add_white_noise


DEFAULT_NOISE_BANK_PATH = "datasets/x_harmonic_5ev_stand_210925.pt"
DEFAULT_MODEL_PATHS = (
    "xai_phasenet/original.pt.v2",
    "../../../FNOclass/PhaseNet-main/original.pt.v2",
)
COMPONENTS = ("E", "N", "Z")
COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c")


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


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested with --device cuda, but PyTorch cannot see a CUDA device.")
    return torch.device(device_name)


def ensure_time_channel(batch):
    batch = np.asarray(batch)
    if batch.ndim != 3:
        raise ValueError(f"Expected a 3D waveform batch, got shape {batch.shape}")
    if batch.shape[2] == 3:
        return batch.astype(np.float32, copy=False)
    if batch.shape[1] == 3:
        return np.transpose(batch, (0, 2, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Expected waveform batch shape (N,T,3) or (N,3,T), got {batch.shape}")


def labels_to_signal_indices(labels):
    labels = np.asarray(labels)
    if labels.ndim == 1:
        return np.where(labels != 0)[0]
    return np.where(labels.reshape(labels.shape[0], -1).any(axis=1))[0]


def normalize_optional_path(value):
    if value is None:
        return None
    value = str(value)
    if value.strip().lower() in {"", "none", "null", "synthetic"}:
        return None
    return value


def crop_trace(batch, trace_length):
    if trace_length == 0:
        return batch
    if batch.shape[1] < trace_length:
        raise ValueError(f"Requested --trace-length {trace_length}, but waveform length is {batch.shape[1]}.")
    return batch[:, :trace_length, :]


def label_summary(labels, source_index):
    label = np.asarray(labels[source_index])
    if label.ndim == 0:
        value = int(label.item())
        if value == 0:
            return "noise window"
        if value == 1:
            return "event window"
        return f"class label={value}"
    nonzero = int(np.count_nonzero(label))
    return f"label mask, nonzero={nonzero}"


def select_examples(args):
    print(f"Loading signal data from {args.data_path}")
    x_data = ensure_time_channel(load_array(args.data_path))
    labels = load_array(args.labels_path)

    signal_source_indices = labels_to_signal_indices(labels)
    if len(signal_source_indices) == 0:
        raise ValueError("No signal windows found in labels. Expected nonzero labels for signals.")

    ordered_signal = np.load(args.signal_indices_path)
    if len(ordered_signal) < args.n_examples:
        raise ValueError("--n-examples exceeds available signal index count")

    signal_pool = x_data[signal_source_indices]
    signal_pool_indices = ordered_signal[: args.n_examples]
    signal_source = signal_source_indices[signal_pool_indices]
    signal_examples = signal_pool[signal_pool_indices]

    signal_examples = crop_trace(signal_examples, args.trace_length)
    return signal_examples, signal_source, labels


def build_noisy_examples(args, signal_examples):
    noise_bank_path = normalize_optional_path(args.noise_bank_path)
    noise_bank = None
    if noise_bank_path is not None:
        print(f"Loading harmonic noise bank from {noise_bank_path}")
        noise_bank = ensure_time_channel(load_array(noise_bank_path))
    else:
        print("Using synthetic harmonic fallback because --noise-bank-path is none.")

    white_by_amp = {}
    harmonic_by_amp = {}
    for rel_amp in args.rel_amps:
        white_noisy, _ = add_white_noise(
            signal_examples,
            fs=args.fs,
            use_snr_db=False,
            rel_amp=rel_amp,
            per_component=True,
            random_seed=args.noise_seed,
        )
        harmonic_noisy, _ = add_harmonic_pump_noise(
            signal_examples,
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
        white_by_amp[rel_amp] = white_noisy.astype(np.float32, copy=False)
        harmonic_by_amp[rel_amp] = harmonic_noisy.astype(np.float32, copy=False)

    return white_by_amp, harmonic_by_amp


def predict_arrivals(args, signal_examples):
    if args.arrival_source == "none":
        return None

    device = resolve_device(args.device)
    model_path = find_model_path(args.model_path)
    print(f"Predicting P/S arrivals with {model_path}")
    model = PhaseNet(component_order="ENZ")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    x_batch = torch.from_numpy(np.transpose(signal_examples, (0, 2, 1)).astype(np.float32)).to(device)
    with torch.no_grad():
        prob = model(x_batch).detach().cpu().numpy()

    p_times = np.argmax(prob[:, 1, :], axis=1) / args.fs
    s_times = np.argmax(prob[:, 2, :], axis=1) / args.fs
    return p_times, s_times


def plot_offset_traces(ax, waveform, fs, title, arrivals=None, show_arrival_labels=False):
    t = np.arange(waveform.shape[0], dtype=float) / fs
    offsets = np.array([2.4, 0.0, -2.4])
    for component_idx, (name, color, offset) in enumerate(zip(COMPONENTS, COLORS, offsets)):
        trace = waveform[:, component_idx]
        scale = np.max(np.abs(trace))
        if scale <= 0:
            scale = 1.0
        ax.plot(t, trace / scale + offset, color=color, linewidth=0.8, label=name)

    if arrivals is not None:
        p_time, s_time = arrivals
        ax.axvline(
            p_time,
            color="blue",
            linewidth=1.0,
            linestyle="--",
            label="Pred. P-arrival" if show_arrival_labels else None,
        )
        ax.axvline(
            s_time,
            color="green",
            linewidth=1.0,
            linestyle="--",
            label="Pred. S-arrival" if show_arrival_labels else None,
        )

    ax.set_title(title, fontsize=10)
    ax.set_yticks(offsets)
    ax.set_yticklabels(COMPONENTS)
    ax.set_ylim(offsets[-1] - 1.25, offsets[0] + 1.25)
    ax.set_xlim(t[0], t[-1])
    ax.margins(x=0)
    ax.grid(True, alpha=0.25, linewidth=0.5)


def plot_examples(args, signal_examples, signal_source, white_by_amp, harmonic_by_amp, labels, arrivals):
    n_rows = signal_examples.shape[0]
    panels = [("Clean signal", signal_examples)]
    for rel_amp in args.rel_amps:
        panels.append((f"Signal + white noise\nrel_amp={rel_amp:g}", white_by_amp[rel_amp]))
    for rel_amp in args.rel_amps:
        panels.append((f"Signal + harmonic noise\nrel_amp={rel_amp:g}", harmonic_by_amp[rel_amp]))

    n_cols = len(panels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.25 * n_cols, 3.8 * n_rows),
        sharex=True,
        squeeze=False,
        dpi=args.dpi,
    )

    for row in range(n_rows):
        signal_meta = f"source idx={signal_source[row]}, {label_summary(labels, signal_source[row])}"
        for col, (panel_title, panel_data) in enumerate(panels):
            row_arrivals = None if arrivals is None else (arrivals[0][row], arrivals[1][row])
            plot_offset_traces(
                axes[row, col],
                panel_data[row],
                args.fs,
                f"{panel_title}\n{signal_meta}",
                arrivals=row_arrivals,
                show_arrival_labels=(row == 0 and col == 0),
            )

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (s)")
    axes[0, 0].legend(loc="upper left", fontsize=8)

    fig.suptitle("Input waveform examples", fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "input_data_examples.png"
    fig.savefig(out_file, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved {out_file}")


def parse_float_list(value):
    try:
        return tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected comma-separated floats, got: {value}") from exc


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize clean, white-noisy, and harmonic-noisy signal traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-path", "--data_path", default="datasets/x_test_5391ev_270425.pt")
    parser.add_argument("--labels-path", "--labels_path", default="datasets/y_test_5391ev_270425.pt")
    parser.add_argument("--signal-indices-path", "--signal_indices_path", default="datasets/indices_signal_140425.npy")
    parser.add_argument("--model-path", "--model_path", default=None)
    parser.add_argument("--noise-bank-path", "--noise_bank_path", default=DEFAULT_NOISE_BANK_PATH)
    parser.add_argument("--output-dir", "--output_dir", default="output/data")
    parser.add_argument("--n-examples", "--n_examples", type=int, default=2)
    parser.add_argument("--trace-length", "--trace_length", type=int, default=3001)
    parser.add_argument("--fs", type=float, default=100.0)
    parser.add_argument("--noise-fs", "--noise_fs", type=float, default=100.0)
    parser.add_argument("--rel-amps", "--rel_amps", type=parse_float_list, default=(1.0, 2.0))
    parser.add_argument(
        "--rel-amp",
        "--rel_amp",
        type=float,
        default=None,
        help="Optional single relative amplitude override. Prefer --rel-amps for multiple columns.",
    )
    parser.add_argument("--harmonic-freqs", type=parse_float_list, default=(20.0, 40.0))
    parser.add_argument("--arrival-source", choices=("model", "none"), default="model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--noise-seed", "--noise_seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_examples < 1:
        raise ValueError("--n-examples must be at least 1")
    if args.trace_length < 0:
        raise ValueError("--trace-length must be non-negative")
    if args.rel_amp is not None:
        args.rel_amps = (args.rel_amp,)
    if not args.rel_amps:
        raise ValueError("--rel-amps must contain at least one value")

    signal_examples, signal_source, labels = select_examples(args)
    white_by_amp, harmonic_by_amp = build_noisy_examples(args, signal_examples)
    arrivals = predict_arrivals(args, signal_examples)
    plot_examples(args, signal_examples, signal_source, white_by_amp, harmonic_by_amp, labels, arrivals)


if __name__ == "__main__":
    main()
