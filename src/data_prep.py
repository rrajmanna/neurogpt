import csv
import numpy as np
import mne

TARGET_SFREQ = 250          
WINDOW_SEC = 0.2 
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SFREQ)

MOVEMENT_HALFWIDTH_SEC = 0.2  

def clean_channel_name(raw_name):

    name = raw_name.replace("POL ", "").strip()
    if "-" in name:
        name = name.split("-")[0]
    return name


def load_good_channels(channels_tsv_path):
    good = set()
    with open(channels_tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["status"] == "good":
                good.add(row["name"])
    return good


def load_subject_session(edf_path, channels_tsv_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    name_map = {clean_channel_name(ch): ch for ch in raw.ch_names}
    good_names = load_good_channels(channels_tsv_path)

  
    keep = [name_map[n] for n in good_names if n in name_map]

    if len(keep) == 0:
        raise ValueError(f"No good channels found in {edf_path}")

    raw.pick(keep)
    raw.notch_filter(60, verbose=False)
    raw.filter(1., 100., verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)

    return raw


def get_movement_mask(raw):
   
    n_samples = len(raw.times)
    mask = np.zeros(n_samples, dtype=bool)
    sfreq = raw.info["sfreq"]

    for ann in raw.annotations:
        if "101" in str(ann["description"]):
            center = int(ann["onset"] * sfreq)
            half = int(MOVEMENT_HALFWIDTH_SEC * sfreq)
            start = max(0, center - half)
            end = min(n_samples, center + half)
            mask[start:end] = True

    return mask


def windowize(raw, movement_mask):
    
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

    n_channels, n_samples = data.shape
    n_windows = n_samples // WINDOW_SAMPLES

    windows = np.zeros((n_windows, n_channels, WINDOW_SAMPLES), dtype=np.float32)
    labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        s, e = i * WINDOW_SAMPLES, (i + 1) * WINDOW_SAMPLES
        windows[i] = data[:, s:e]
        labels[i] = int(movement_mask[s:e].mean() > 0.5)

    return windows, labels


def process_subject_session(edf_path, channels_tsv_path):
   
    raw = load_subject_session(edf_path, channels_tsv_path)
    mask = get_movement_mask(raw)
    windows, labels = windowize(raw, mask)
    return windows, labels

def get_common_good_channels(session_paths):
    """
    session_paths: list of (edf_path, channels_tsv_path) tuples.
    Returns the set of channel names marked 'good' in EVERY session.
    """
    common = None
    for edf_path, channels_tsv_path in session_paths:
        good = load_good_channels(channels_tsv_path)
        # also restrict to channels that actually exist in this EDF
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        name_map = {clean_channel_name(ch): ch for ch in raw.ch_names}
        good = {n for n in good if n in name_map}

        common = good if common is None else (common & good)
    return common


def process_subject_session_fixed(edf_path, channels_tsv_path, fixed_channels):
    """Like process_subject_session, but restricted to a pre-determined channel set."""
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    name_map = {clean_channel_name(ch): ch for ch in raw.ch_names}
    keep = [name_map[n] for n in fixed_channels if n in name_map]

    raw.pick(keep)
    raw.notch_filter(60, verbose=False)
    raw.filter(1., 100., verbose=False)
    raw.resample(TARGET_SFREQ, verbose=False)

    mask = get_movement_mask(raw)
    windows, labels = windowize(raw, mask)
    return windows, labels



from scipy.signal import welch

BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (70, 100)}
MOVEMENT_PRE_SEC = 0.1
MOVEMENT_POST_SEC = 0.35

def compute_band_power_features(window, sfreq):
    freqs, psd = welch(window, fs=sfreq, axis=-1, nperseg=window.shape[-1])
    features = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs <= high)
        features.append(psd[:, mask].mean(axis=-1))
    return np.stack(features, axis=-1)

def get_movement_mask_v2(raw):
    n_samples = len(raw.times)
    mask = np.zeros(n_samples, dtype=bool)
    sfreq = raw.info["sfreq"]
    for ann in raw.annotations:
        if "101" in str(ann["description"]):
            center = int(ann["onset"] * sfreq)
            pre = int(MOVEMENT_PRE_SEC * sfreq)
            post = int(MOVEMENT_POST_SEC * sfreq)
            start = max(0, center - pre)
            end = min(n_samples, center + post)
            mask[start:end] = True
    return mask

def windowize_with_features(raw, movement_mask):
    data = raw.get_data()
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-6)

    n_channels, n_samples = data.shape
    n_windows = n_samples // WINDOW_SAMPLES
    sfreq = raw.info["sfreq"]

    windows = np.zeros((n_windows, n_channels, WINDOW_SAMPLES), dtype=np.float32)
    band_features = np.zeros((n_windows, n_channels, len(BANDS)), dtype=np.float32)
    labels = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        s, e = i * WINDOW_SAMPLES, (i + 1) * WINDOW_SAMPLES
        windows[i] = data[:, s:e]
        band_features[i] = compute_band_power_features(data[:, s:e], sfreq)
        labels[i] = int(movement_mask[s:e].mean() > 0.5)

    return windows, band_features, labels