import numpy as np

def add_harmonic_pump_noise(
    X,
    fs=None,                         # sampling rate of X (Hz); if None, inferred as n_samples/60
    freqs=(20.0, 40.0),              # used only for synthetic fallback
    use_snr_db=True,                 # True -> use SNR (dB); False -> use rel_amp
    snr_db=10.0,                     # scalar SNR OR (low, high) tuple for per-sample random SNR
    rel_amp=0.2,                     # scalar relative amplitude OR (low, high) tuple for per-sample random
    per_component=True,              # scale noise level per component (N/E/Z) vs one scale for 3C vector
    random_seed=None,                # reproducibility
    return_noise=True,               # also return the noise array that was added
    noise_bank=None,                 # np.ndarray (M, T_bank, 3) of real noise examples
    noise_fs=None,                   # sampling rate (Hz) of noise_bank; if None and fs is known, assume same
    allow_wrap=True,                 # if True, circular-wrap noise segments shorter than needed
    center_and_detrend=True,         # remove mean from each noise component before scaling
    bank_permute_components=False    # if True, randomly permute components of chosen noise (rarely needed)
):
    assert X.ndim == 3 and X.shape[2] == 3, "Expected shape (N, T, 3)."
    N, T, C = X.shape

    if fs is None:
        fs = T / 60.0
        if not np.isclose(fs, round(fs)):
            raise ValueError(f"Cannot infer integer fs from T={T}. Please pass fs explicitly.")
        fs = int(round(fs))

    rng = np.random.default_rng(random_seed)

    def _resample_1d(x, src_fs, dst_fs):
        if np.isclose(src_fs, dst_fs):
            return x.copy()
        src_t = np.arange(len(x)) / float(src_fs)
        dst_t = np.arange(int(round(len(x) * (dst_fs / float(src_fs))))) / float(dst_fs)
        return np.interp(dst_t, src_t, x)

    def _resample_3c(noise_tc, src_fs, dst_fs):
        if np.isclose(src_fs, dst_fs):
            return noise_tc.copy()
        out = np.empty((int(round(noise_tc.shape[0] * (dst_fs / float(src_fs)))), 3), dtype=noise_tc.dtype)
        for j in range(3):
            out[:, j] = _resample_1d(noise_tc[:, j], src_fs, dst_fs)
        return out

    def _select_noise_segment(noise_bank, noise_fs, target_len):
        M, T_bank, _ = noise_bank.shape
        idx = rng.integers(0, M)
        seg = noise_bank[idx]  # (T_bank, 3)
        src_fs = noise_fs if (noise_fs is not None) else fs
        seg = _resample_3c(seg, src_fs, fs)

        if seg.shape[0] == target_len:
            return seg
        if seg.shape[0] > target_len:
            start = rng.integers(0, seg.shape[0] - target_len + 1)
            return seg[start:start+target_len]

        if allow_wrap:
            reps = int(np.ceil(target_len / seg.shape[0]))
            tiled = np.tile(seg, (reps, 1))[:target_len]
            return tiled
        else:
            out = np.zeros((target_len, 3), dtype=seg.dtype)
            out[:seg.shape[0]] = seg
            return out

    def _maybe_center(seg):
        if not center_and_detrend:
            return seg
        return seg - seg.mean(axis=0, keepdims=True)

    eps = 1e-12
    if per_component:
        sig_rms = np.sqrt(np.mean(X**2, axis=1, keepdims=True)) + eps  # (N,1,3)
    else:
        s = np.sqrt(np.mean(X**2, axis=(1,2), keepdims=True)) + eps     # (N,1,1)
        sig_rms = np.repeat(s, C, axis=2)                               # (N,1,3)

    if use_snr_db:
        if np.iterable(snr_db) and len(snr_db) == 2:
            snr_vals = rng.uniform(snr_db[0], snr_db[1], size=(N,1,1))
        else:
            snr_vals = np.full((N,1,1), float(snr_db))
        noise_rms_target = sig_rms / (10.0**(snr_vals/20.0))            # (N,1,3)
    else:
        if np.iterable(rel_amp) and len(rel_amp) == 2:
            rel_vals = rng.uniform(rel_amp[0], rel_amp[1], size=(N,1,1))
        else:
            rel_vals = np.full((N,1,1), float(rel_amp))
        noise_rms_target = sig_rms * rel_vals                           # (N,1,3)

    if noise_bank is not None:
        assert noise_bank.ndim == 3 and noise_bank.shape[2] == 3, "noise_bank must be (M, T_bank, 3)."
        M, T_bank, _ = noise_bank.shape
        if M < 1:
            raise ValueError("noise_bank is empty.")

        noise = np.empty_like(X, dtype=float)  # (N, T, 3)
        for n in range(N):
            seg = _select_noise_segment(noise_bank, noise_fs, T)  # (T,3)
            if bank_permute_components:
                perm = rng.permutation(3)
                seg = seg[:, perm]
            seg = _maybe_center(seg)
            seg_rms = np.sqrt(np.mean(seg**2, axis=0, keepdims=True)) + eps  # (1,3)
            scale = (noise_rms_target[n,0,:] / seg_rms).reshape(1,3)         # (1,3)
            seg_scaled = seg * scale                                         # (T,3)
            noise[n] = seg_scaled

        X_noisy = X + noise
        return (X_noisy, noise) if return_noise else X_noisy

    nyq = fs / 2.0
    freqs_arr = np.array(freqs, dtype=float)
    if np.any(freqs_arr <= 0) or np.any(freqs_arr >= nyq):
        raise ValueError(f"All freqs must be in (0, {nyq}) for fs={fs}. Got {freqs_arr}.")

    t = np.arange(T, dtype=float) / fs  # (T,)
    F = len(freqs_arr)
    phases = rng.uniform(0, 2*np.pi, size=(N, C, F))
    sines_per_freq = np.sin(2*np.pi*freqs_arr.reshape(1,1,F,1)*t.reshape(1,1,1,T) + phases.reshape(N,C,F,1))
    harmonic_sum = sines_per_freq.sum(axis=2)  # (N, C, T)

    noise_rms_nat = np.sqrt(np.mean(harmonic_sum**2, axis=2, keepdims=True)) + eps  # (N,C,1)
    harmonic_unit = harmonic_sum / noise_rms_nat  # (N,C,T)
    noise = harmonic_unit.reshape(N,1,C,T) * noise_rms_target.reshape(N,1,C,1)
    noise = noise.reshape(N,C,T).transpose(0,2,1)  # (N,T,3)

    X_noisy = X + noise
    return (X_noisy, noise) if return_noise else X_noisy

def add_white_noise(
    X,
    fs=None,                      
    use_snr_db=True,              
    snr_db=10.0,                  
    rel_amp=0.2,                  
    per_component=True,           
    random_seed=None,             
    return_noise=True             
):
    assert X.ndim == 3 and X.shape[2] == 3, "Expected shape (N, T, 3)."
    N, T, C = X.shape
    rng = np.random.default_rng(random_seed)
    eps = 1e-12

    if per_component:
      sig_rms = np.sqrt(np.mean(X**2, axis=1, keepdims=True)) + eps
    else:
      s = np.sqrt(np.mean(X**2, axis=(1,2), keepdims=True)) + eps
      sig_rms = np.repeat(s, C, axis=2)

    if use_snr_db:
        if np.iterable(snr_db) and len(snr_db) == 2:
            snr_vals = rng.uniform(snr_db[0], snr_db[1], size=(N,1,1))
        else:
            snr_vals = np.full((N,1,1), float(snr_db))
        noise_rms_target = sig_rms / (10.0**(snr_vals/20.0))  # (N,1,3)
    else:
        if np.iterable(rel_amp) and len(rel_amp) == 2:
            rel_vals = rng.uniform(rel_amp[0], rel_amp[1], size=(N,1,1))
        else:
            rel_vals = np.full((N,1,1), float(rel_amp))
        noise_rms_target = sig_rms * rel_vals                   # (N,1,3)

    white = rng.standard_normal(size=(N, T, C))
    cur_rms = np.sqrt(np.mean(white**2, axis=1, keepdims=True)) + eps  # (N,1,3)
    white_scaled = white * (noise_rms_target / cur_rms)                # (N,T,3)

    X_noisy = X + white_scaled
    return (X_noisy, white_scaled) if return_noise else X_noisy
