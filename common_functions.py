import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
import pywt


##################################

def mwa (signal, window):
    dataframe = pd.DataFrame(signal)
    signal_mwaved = dataframe.rolling(window, center=True, closed='both').mean()
    return signal_mwaved


def wavelet_pywt(signal):
    wavelet_mwa = mwa(signal, 20)
    weveleted, a2, a3 = pywt.wavedec(signal, 'sym5', mode='smooth', level=2, axis=-1)
    return weveleted


##################################



def deriv(signal, win_len: int = 120):
    sav_gol_deriv = savgol_filter(x=signal,
                                  window_length=win_len,  # 300
                                  polyorder=2,
                                  deriv=1,
                                  mode='nearest')
    return sav_gol_deriv


def filtered(signal, win_len: int = 120):
    sav_gol_filtered = savgol_filter(x=signal,
                                     window_length=win_len,  # 300
                                     polyorder=2,
                                     mode='nearest')
    return sav_gol_filtered


def signal_peaks(signal):
    peaks, properties = find_peaks(signal, prominence=1, width=20)
    return peaks, properties


def mean_compensate(signal, poly_order: int = 3):
    poly_coefficients = np.polyfit(range(len(signal)), signal, poly_order)
    poly_baseline = np.poly1d(poly_coefficients)
    signal_detrended = signal - poly_baseline(range(len(signal)))
    return signal_detrended


def cycles_avg(signal: np.ndarray, cycles_idxs: list = None) -> np.ndarray:

    win_len = int(len(signal) // 13)
    signal = mean_compensate(signal)
    lvl = 0.35
    # waveleted = wavelet_pywt(signal)
    sav_gol_deriv = deriv(signal, win_len=win_len)

    try:
           # waveleted /= max(waveleted)
            sav_gol_deriv /= max(sav_gol_deriv)
    except ZeroDivisionError:
        pass

    # fronts = np.where(waveleted > lvl * waveleted.max(), 1, -1)
    fronts = np.where(sav_gol_deriv > lvl * sav_gol_deriv.max(), 1, -1)
    
    fronts[-win_len:] = -1
    fronts_edges = (fronts - np.roll(fronts, shift=1)) * 0.5

    starts = np.where(fronts_edges >= 0.2)
    peaks_ind = np.where(-fronts_edges >= 0.2)
    sav_gol_filtered = filtered(signal, win_len=win_len)
    starts = starts[0]

    try:
        length = np.min(np.diff(starts))
        # print(length)
    except:
        return None

    cycles = []

    sl_start = len(starts) - 2

    if sl_start < 0:
        sl_start = 0

    sl_end = sl_start + 1
    cycles_idxs = range(sl_start, sl_end) if cycles_idxs is None else cycles_idxs

    for i in cycles_idxs:
        try:
            start = starts[i]
            end = starts[i] + length
            peak = signal_peaks(signal)[0][i]
            cycle = waveleted[start: end]
            baseline = np.linspace(cycle[0], cycle[-1], len(cycle))
            cycle_shifted = cycle - baseline
            # cycle_norm =
            cycle_scaled = cycle_shifted / np.max(cycle_shifted)
            cycles.append(cycle_scaled)
        except:
            pass

    cycles = np.array(cycles).T

    return cycles


def data_to_period(data_df: pd.DataFrame) -> pd.DataFrame:
    """Combines all periods of single patient into one ensemble averaged period.

    Parameters
    ----------
    data_df : pd.Dataframe
        single patient Dataframe


    Returns
    -------
    pd.DataFrame
        Combined periods of single patient
    """
    # For debugging:
    # print(data_df)

    periods = []
    for signal_name, signal in data_df.items():
        try:
            periods_part = cycles_avg(signal=signal.values)
            # plt.plot(periods_part)
            # plt.show()
        except ValueError:
            pass

        if periods_part is not None:
            periods.append(periods_part)

    try:
        lengths = [max(x.shape) for x in periods]
        min_len = min(lengths)
        # print('Lengths: ', lengths)
    except ValueError:
        return None

    periods = [x[:min_len] for x in periods]

    periods_df = pd.DataFrame(periods).T
    periods_df.columns = data_df.columns
    periods_df.dropna(inplace=True)
    # print('periods_df.shape:', periods_df.shape)
    # print('periods_df:', periods_df)

    # plt.plot(periods_df)
    # plt.show()

    period_avg_df = pd.DataFrame(periods_df.mean(axis=1).dropna())
    col_name = '_'.join(data_df.columns[0].split('_')[:2] + ['wav'])
    # print('col_name:', col_name)
    # print('period_avg_df.shape:', period_avg_df.shape)
    period_avg_df.columns = [col_name]
    # print('period_avg_df:\n', period_avg_df)
    return period_avg_df
    # input : df 3 signals
    # return: df period (усредненный)
