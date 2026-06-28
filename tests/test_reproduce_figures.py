import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "xai-phasenet-matplotlib"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

from xai_phasenet.model import PhaseNet


REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def make_waveforms(count, length):
    t = np.linspace(0.0, 1.0, length, dtype=np.float32)
    waveforms = np.zeros((count, length, 3), dtype=np.float32)
    for i in range(count):
        waveforms[i, :, 0] = np.sin(2 * np.pi * (i + 1) * t)
        waveforms[i, :, 1] = np.cos(2 * np.pi * (i + 2) * t)
        waveforms[i, :, 2] = np.sin(2 * np.pi * (i + 3) * t + 0.25)
    return waveforms


class ReproduceFiguresTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work_dir = Path(self.temp_dir.name)
        self.data_dir = self.work_dir / "data"
        self.output_dir = self.work_dir / "output"
        self.data_dir.mkdir()
        self.output_dir.mkdir()

        length = 3001
        x_data = make_waveforms(4, length)
        y_data = np.array([1, 0, 1, 1], dtype=np.int64)

        rng = np.random.default_rng(20240622)
        x_noise = rng.normal(0.0, 0.05, size=(2, length, 3)).astype(np.float32)

        torch.save(torch.from_numpy(x_data), self.data_dir / "x_test.pt")
        torch.save(torch.from_numpy(y_data), self.data_dir / "y_test.pt")
        torch.save(torch.from_numpy(x_noise), self.data_dir / "x_noise.pt")
        np.save(self.data_dir / "indices_signal.npy", np.array([0, 1], dtype=np.int64))
        np.save(self.data_dir / "indices_noise.npy", np.array([0, 1], dtype=np.int64))

        torch.manual_seed(7)
        model = PhaseNet(component_order="ENZ")
        self.model_path = self.data_dir / "phasenet.pt"
        torch.save(model.state_dict(), self.model_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_script(self, *args, timeout=240):
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            self.fail(
                "Command failed:\n"
                f"{' '.join([sys.executable, *args])}\n\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )
        return result

    def assert_png(self, path):
        self.assertTrue(path.exists(), f"Missing expected figure: {path}")
        self.assertGreater(path.stat().st_size, 0, f"Figure is empty: {path}")
        with path.open("rb") as handle:
            self.assertEqual(handle.read(len(PNG_SIGNATURE)), PNG_SIGNATURE)

    def test_reproduce_figure_scripts_with_synthetic_fixture(self):
        fig2_dir = self.output_dir / "figure2"
        fig4_dir = self.output_dir / "figure4"

        self.run_script(
            "scripts/reproduce_figures_2_and_3_gradcam.py",
            "--model-path",
            str(self.model_path),
            "--data-path",
            str(self.data_dir / "x_test.pt"),
            "--labels-path",
            str(self.data_dir / "y_test.pt"),
            "--noise-path",
            str(self.data_dir / "x_noise.pt"),
            "--signal-indices-path",
            str(self.data_dir / "indices_signal.npy"),
            "--noise-indices-path",
            str(self.data_dir / "indices_noise.npy"),
            "--output-dir",
            str(fig2_dir),
            "--n-test",
            "1",
            "--low-snr-source-index",
            "2",
            "--device",
            "cpu",
            "--dpi",
            "40",
        )

        for filename in (
            "Figure2_High_SNR_idx0.png",
            "Figure2_Noise_idx1.png",
            "Figure2_Low_SNR_idx2.png",
            "Figure3_High_SNR_zoom_idx0.png",
            "Figure3_Low_SNR_zoom_idx2.png",
        ):
            self.assert_png(fig2_dir / filename)

        self.run_script(
            "scripts/reproduce_figure_4_shap_analysis.py",
            "--model-path",
            str(self.model_path),
            "--data-path",
            str(self.data_dir / "x_test.pt"),
            "--labels-path",
            str(self.data_dir / "y_test.pt"),
            "--noise-path",
            str(self.data_dir / "x_noise.pt"),
            "--signal-indices-path",
            str(self.data_dir / "indices_signal.npy"),
            "--noise-indices-path",
            str(self.data_dir / "indices_noise.npy"),
            "--output-dir",
            str(fig4_dir),
            "--n-test",
            "1",
            "--hop",
            "1",
            "--device",
            "cpu",
            "--dpi",
            "40",
        )

        self.assert_png(fig4_dir / "Figure4_SHAP_histograms.png")
        for filename in (
            "impP_sig_1_trace3001_win3001_hop1.npy",
            "impS_sig_1_trace3001_win3001_hop1.npy",
            "impP_noi_1_trace3001_win3001_hop1.npy",
            "impS_noi_1_trace3001_win3001_hop1.npy",
        ):
            path = fig4_dir / "shap_data" / filename
            self.assertTrue(path.exists(), f"Missing expected SHAP cache: {path}")
            self.assertEqual(np.load(path).shape, (1, 3))

    def test_reproduce_figures_5_to_8_script_with_synthetic_fixture(self):
        fig_dir = self.output_dir / "figures_5_to_8"

        self.run_script(
            "scripts/reproduce_figures_5_to_8_shap_extensions.py",
            "--model-path",
            str(self.model_path),
            "--data-path",
            str(self.data_dir / "x_test.pt"),
            "--labels-path",
            str(self.data_dir / "y_test.pt"),
            "--noise-path",
            str(self.data_dir / "x_noise.pt"),
            "--signal-indices-path",
            str(self.data_dir / "indices_signal.npy"),
            "--noise-indices-path",
            str(self.data_dir / "indices_noise.npy"),
            "--output-dir",
            str(fig_dir),
            "--figure4-cache-dir",
            str(fig_dir / "missing_figure4_cache"),
            "--n-test",
            "2",
            "--hop",
            "3001",
            "--device",
            "cpu",
            "--dpi",
            "40",
        )

        for filename in (
            "Figure5_SHAP_violins.png",
            "Figure6_SHAP_pairwise_interactions.png",
            "Figure7_Reduced_Component_Performance.png",
            "Figure8_SHAP_dispersion_vs_snr.png",
        ):
            self.assert_png(fig_dir / filename)

        for filename in (
            "impP_sig_2_trace3001_win3001_hop3001.npy",
            "impS_sig_2_trace3001_win3001_hop3001.npy",
            "impP_noi_2_trace3001_win3001_hop3001.npy",
            "impS_noi_2_trace3001_win3001_hop3001.npy",
            "mean_intP_all_2_trace3001_win3001_hop3001.npy",
            "mean_intS_all_2_trace3001_win3001_hop3001.npy",
        ):
            path = fig_dir / "shap_data" / filename
            self.assertTrue(path.exists(), f"Missing expected cache: {path}")

        self.assertEqual(np.load(fig_dir / "shap_data" / "impP_sig_2_trace3001_win3001_hop3001.npy").shape, (2, 3))
        self.assertEqual(np.load(fig_dir / "shap_data" / "mean_intP_all_2_trace3001_win3001_hop3001.npy").shape, (4, 3))

    def test_visualize_input_data_script_with_synthetic_fixture(self):
        data_vis_dir = self.output_dir / "data_vis"

        self.run_script(
            "scripts/visualize_input_data.py",
            "--data-path",
            str(self.data_dir / "x_test.pt"),
            "--labels-path",
            str(self.data_dir / "y_test.pt"),
            "--signal-indices-path",
            str(self.data_dir / "indices_signal.npy"),
            "--noise-bank-path",
            "none",
            "--arrival-source",
            "none",
            "--output-dir",
            str(data_vis_dir),
            "--n-examples",
            "1",
            "--dpi",
            "40",
        )

        self.assert_png(data_vis_dir / "input_data_examples.png")

    def test_reproduce_figure_4_defaults_match_manuscript_run(self):
        from scripts.reproduce_figure_4_shap_analysis import COMPONENT_STYLES, FIGURE4_PANEL_TITLES, build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.output_dir, "output/figure4")
        self.assertEqual(args.n_test, 5000)
        self.assertEqual(args.trace_length, 3001)
        self.assertEqual(args.win, 3001)
        self.assertEqual(args.hop, 1)
        self.assertEqual(args.nbins, 40)
        self.assertEqual(args.bin_qmax, 99.5)
        self.assertEqual(args.dpi, 600)
        self.assertEqual(
            FIGURE4_PANEL_TITLES,
            (
                "a) P-class SHAP (Signal windows)",
                "b) P-class SHAP (Noise windows)",
                "c) S-class SHAP (Signal windows)",
                "d) S-class SHAP (Noise windows)",
            ),
        )
        self.assertEqual(
            COMPONENT_STYLES,
            (
                (0, "E component", "blue"),
                (1, "N component", "green"),
                (2, "Z component", "red"),
            ),
        )

    def test_reproduce_figures_5_to_8_defaults_match_manuscript_run(self):
        from scripts.reproduce_figures_5_to_8_shap_extensions import (
            FIGURE7_CONFIGS,
            FIGURE7_NOTEBOOK_MEANS,
            FIGURE7_NOTEBOOK_STDS,
            build_parser,
        )

        args = build_parser().parse_args([])
        self.assertEqual(args.output_dir, "output/figures_5_to_8")
        self.assertEqual(args.figure4_cache_dir, "output/figure4/shap_data")
        self.assertEqual(args.n_test, 5000)
        self.assertEqual(args.trace_length, 3001)
        self.assertEqual(args.win, 3001)
        self.assertEqual(args.hop, 1)
        self.assertEqual(args.dpi, 600)
        self.assertEqual(
            FIGURE7_CONFIGS,
            (
                "3C (E, N, Z)",
                "2C (E, N)",
                "2C (E, Z)",
                "2C (N, Z)",
                "1C (N)",
                "1C (E)",
                "1C (Z)",
            ),
        )
        np.testing.assert_allclose(
            FIGURE7_NOTEBOOK_MEANS,
            np.array([0.9641, 0.9716, 0.9662, 0.9555, 0.9474, 0.9455, 0.8819]),
        )
        np.testing.assert_allclose(
            FIGURE7_NOTEBOOK_STDS,
            np.array([0.0092, 0.0022, 0.0080, 0.0042, 0.0012, 0.0066, 0.0167]),
        )

    def test_noise_robustness_script_with_synthetic_fixture(self):
        noise_dir = self.output_dir / "noise_robustness"

        self.run_script(
            "scripts/reproduce_figures_9_to_11_noise_robustness.py",
            "--model-path",
            str(self.model_path),
            "--data-path",
            str(self.data_dir / "x_test.pt"),
            "--labels-path",
            str(self.data_dir / "y_test.pt"),
            "--noise-path",
            str(self.data_dir / "x_noise.pt"),
            "--signal-indices-path",
            str(self.data_dir / "indices_signal.npy"),
            "--noise-indices-path",
            str(self.data_dir / "indices_noise.npy"),
            "--output-dir",
            str(noise_dir),
            "--noise-type",
            "both",
            "--noise-bank-path",
            "none",
            "--num-samples",
            "2",
            "--rel-amps",
            "1.0",
            "--extended-rel-amps",
            "1.0",
            "--n-splits",
            "1",
            "--train-per-class",
            "1",
            "--test-per-class",
            "1",
            "--hop",
            "3001",
            "--device",
            "cpu",
            "--dpi",
            "40",
        )

        for filename in (
            "noise_robustness_results.csv",
            "noise_robustness_summary.csv",
            "noise_robustness_extended_results.csv",
            "noise_robustness_extended_summary.csv",
            "Figure9_Noise_Robustness.png",
            "Figure10_Precision_Recall_Noise_Robustness.png",
            "Figure11_Extended_F1_Noise_Robustness.png",
        ):
            self.assertTrue((noise_dir / filename).exists(), f"Missing expected output: {filename}")
        self.assert_png(noise_dir / "Figure9_Noise_Robustness.png")
        self.assert_png(noise_dir / "Figure10_Precision_Recall_Noise_Robustness.png")
        self.assert_png(noise_dir / "Figure11_Extended_F1_Noise_Robustness.png")

        cache_files = sorted((noise_dir / "score_cache").glob("scores_*_rel1p0*.npz"))
        self.assertEqual(len(cache_files), 2)
        for cache_file in cache_files:
            cached = np.load(cache_file)
            self.assertEqual(cached["prob_scores"].shape, (4,))
            self.assertEqual(cached["shap_scores"].shape, (4,))
            self.assertEqual(int(cached["trace_length"]), 3001)
            self.assertEqual(int(cached["noise_seed"]), 42)
            if "harmonic" in cache_file.name:
                self.assertEqual(str(cached["noise_source"]), "synthetic_harmonic")
            else:
                self.assertEqual(str(cached["noise_source"]), "white")

        for filename in ("cv_all_metrics_wn.npz", "cv_all_metrics_hn.npz"):
            archive = noise_dir / filename
            self.assertTrue(archive.exists(), f"Missing manuscript-style metrics archive: {filename}")
            metrics = np.load(archive)
            self.assertEqual(metrics["f1_prob"].shape, (1, 1))
            self.assertEqual(metrics["f1_shap"].shape, (1, 1))

        extended_archive = noise_dir / "cv_all_metrics_hn_extended.npz"
        self.assertTrue(extended_archive.exists(), "Missing extended metrics archive")
        extended_metrics = np.load(extended_archive)
        self.assertEqual(extended_metrics["f1_prob"].shape, (1, 1))
        self.assertEqual(extended_metrics["f1_shap"].shape, (1, 1))

    def test_noise_robustness_defaults_match_manuscript_run(self):
        from scripts.reproduce_figures_9_to_11_noise_robustness import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.noise_type, "both")
        self.assertEqual(args.noise_bank_path, "datasets/x_harmonic_5ev_stand_210925.pt")
        self.assertEqual(args.noise_fs, 100.0)
        self.assertEqual(args.output_dir, "output/figures_9_to_11_noise_robustness")
        self.assertEqual(args.num_samples, 5000)
        self.assertEqual(list(args.rel_amps), [round(value / 10.0, 1) for value in range(10, 21)])
        self.assertEqual(list(args.extended_rel_amps), [round(value / 10.0, 1) for value in range(0, 51)])
        self.assertEqual(args.extended_noise_type, "harmonic")
        self.assertEqual(args.n_splits, 5)
        self.assertEqual(args.train_per_class, 50)
        self.assertEqual(args.test_per_class, 4500)
        self.assertEqual(args.trace_length, 3001)
        self.assertEqual(args.win, 3001)
        self.assertEqual(args.hop, 1)
        self.assertEqual(args.dpi, 600)

        ranged = build_parser().parse_args(["--rel-amps", "0.0:0.2:0.1"])
        self.assertEqual(list(ranged.rel_amps), [0.0, 0.1, 0.2])

        extended_ranged = build_parser().parse_args(["--extended-rel-amps", "0.0:0.2:0.1"])
        self.assertEqual(list(extended_ranged.extended_rel_amps), [0.0, 0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
