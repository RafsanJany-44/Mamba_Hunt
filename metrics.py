"""Signal post-processing and official-compatible RhythmMamba metrics."""

import numpy as np
import scipy
from scipy.signal import butter, welch
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve


def get_hr(signal, sampling_rate=30, minimum_bpm=45, maximum_bpm=150):
    frequencies, power = welch(
        signal,
        sampling_rate,
        nfft=1e5 / sampling_rate,
        nperseg=np.min((len(signal) - 1, 256)),
    )
    valid = (frequencies > minimum_bpm / 60) & (
        frequencies < maximum_bpm / 60
    )
    return frequencies[valid][np.argmax(power[valid])] * 60


def _detrend(signal, regularization=100):
    """Smoothness-prior detrending without a dense matrix inverse.

    This computes exactly the same expression as the official implementation:
    ``(I - inv(I + lambda^2 D.T D)) @ signal``.  Solving the sparse
    pentadiagonal system avoids constructing and inverting an NxN dense matrix.
    """
    length = signal.shape[0]
    diagonals = np.array(
        [np.ones(length), -2 * np.ones(length), np.ones(length)]
    )
    difference = spdiags(
        diagonals, np.array([0, 1, 2]), length - 2, length
    ).tocsc()
    system = eye(length, format="csc") + (regularization**2) * (
        difference.T @ difference
    )
    trend = spsolve(system, np.asarray(signal, dtype=np.float64))
    return np.asarray(signal, dtype=np.float64) - trend


def calculate_hr(prediction, label, fs=30, diff_flag=False):
    """Return prediction HR and label HR, matching the official loss helper."""
    prediction = np.asarray(prediction)
    label = np.asarray(label)
    if diff_flag:
        prediction = _detrend(np.cumsum(prediction))
        label = _detrend(np.cumsum(label))
    else:
        prediction = _detrend(prediction)
        label = _detrend(label)

    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype="bandpass")
    prediction = scipy.signal.filtfilt(b, a, np.double(prediction))
    label = scipy.signal.filtfilt(b, a, np.double(label))
    return get_hr(prediction, sampling_rate=fs), get_hr(label, sampling_rate=fs)


def _next_power_of_two(value):
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _snr(prediction, label_hr, fs=30):
    first_harmonic = label_hr / 60
    second_harmonic = 2 * first_harmonic
    deviation = 6 / 60

    prediction = np.expand_dims(prediction, 0)
    nfft = _next_power_of_two(prediction.shape[1])
    frequencies, power = scipy.signal.periodogram(
        prediction, fs=fs, nfft=nfft, detrend=False
    )
    power = np.squeeze(power)

    harmonic_1 = (frequencies >= first_harmonic - deviation) & (
        frequencies <= first_harmonic + deviation
    )
    harmonic_2 = (frequencies >= second_harmonic - deviation) & (
        frequencies <= second_harmonic + deviation
    )
    remainder = (
        (frequencies >= 0.75)
        & (frequencies <= 2.5)
        & ~harmonic_1
        & ~harmonic_2
    )

    signal_power = power[harmonic_1].sum() + power[harmonic_2].sum()
    noise_power = power[remainder].sum()
    return 0 if noise_power == 0 else 20 * np.log10(signal_power / noise_power)


def calculate_video_metrics(prediction, label, fs=30, diff_flag=False):
    if diff_flag:
        prediction = _detrend(np.cumsum(prediction))
        label = _detrend(np.cumsum(label))
    else:
        prediction = _detrend(prediction)
        label = _detrend(label)

    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype="bandpass")
    prediction = scipy.signal.filtfilt(b, a, np.double(prediction))
    label = scipy.signal.filtfilt(b, a, np.double(label))

    predicted_hr = get_hr(prediction, sampling_rate=fs)
    label_hr = get_hr(label, sampling_rate=fs)
    return label_hr, predicted_hr, _snr(prediction, label_hr, fs)


def _join_chunks(chunks):
    ordered = [tensor for _, tensor in sorted(chunks.items(), key=lambda item: item[0])]
    import torch

    return torch.cat(ordered, dim=0).detach().cpu().numpy().reshape(-1)


def calculate_metrics(predictions, labels, fs=30):
    """Calculate and print metrics exactly over each reconstructed recording."""
    predicted_hr = []
    label_hr = []
    snr = []

    for recording_id in predictions:
        prediction = _join_chunks(predictions[recording_id])
        label = _join_chunks(labels[recording_id])
        gt_value, pred_value, snr_value = calculate_video_metrics(
            prediction, label, fs=fs, diff_flag=False
        )
        label_hr.append(gt_value)
        predicted_hr.append(pred_value)
        snr.append(snr_value)

    predicted_hr = np.asarray(predicted_hr)
    label_hr = np.asarray(label_hr)
    snr = np.asarray(snr)
    count = len(predicted_hr)

    absolute_error = np.abs(predicted_hr - label_hr)
    squared_error = np.square(predicted_hr - label_hr)
    percentage_error = np.abs((predicted_hr - label_hr) / label_hr) * 100
    correlation = np.corrcoef(predicted_hr, label_hr)[0, 1]

    result = {
        "MAE": float(np.mean(absolute_error)),
        "RMSE": float(np.sqrt(np.mean(squared_error))),
        "MAPE": float(np.mean(percentage_error)),
        "Pearson": float(correlation),
        "SNR": float(np.mean(snr)),
    }

    # These uncertainty calculations intentionally match the official output.
    print(f"FFT MAE (FFT Label): {result['MAE']} +/- {np.std(absolute_error) / np.sqrt(count)}")
    print(f"FFT RMSE (FFT Label): {result['RMSE']} +/- {np.std(squared_error) / np.sqrt(count)}")
    print(f"FFT MAPE (FFT Label): {result['MAPE']} +/- {np.std(percentage_error) / np.sqrt(count)}")
    pearson_se = np.sqrt((1 - correlation**2) / (count - 2))
    print(f"FFT Pearson (FFT Label): {result['Pearson']} +/- {pearson_se}")
    print(f"FFT SNR (FFT Label): {result['SNR']} +/- {np.std(snr) / np.sqrt(count)} (dB)")
    return result
