import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths
from scipy.integrate import simpson


def period_to_features(period_df: pd.DataFrame) -> dict:
    """Convert averaged period into a dictionary of features.

    Parameters
    ----------
    period_df : pd.DataFrame
        Ensemble averaged period.

    Returns
    -------
    dict
        A dictionary of features, where the keys represent the feature names and the values represent the feature values.
    """

    features = {}
    features['cycle_width_50'] = cycle_width(period_df, at_level=0.5)
    features['cycle_width_25'] = cycle_width(period_df, at_level=0.25)
    features['cycle_width_75'] = cycle_width(period_df, at_level=0.75)

    # features['sys_start'] = sys_starts(period_df)
    
    features['dia_start'] = dia_starts(period_df)
    features['area_under'] = AUC(period_df)
    features['peak_coordinates'] = peak_coord(period_df)
    #features['sys_duration'] = sys_duration(period_df)
    #features['sys_dia_ratio'] = sys_to_dia_ratio(period_df)
    #features['avg_pulse_rate'] = avg_pulse_rate(period_df)
    #features['second_deriv'] = second_derivative(period_df)
    #features['diastolic_peak'] = dia_peak(period_df)
    #features['dicrotic_notch'] = dic_notch(period_df)
    

    return features

"""
def moving_average_window(ppg_signal, window_size=20):
    smoothed_signal = np.zeros(len(ppg_signal))

    for i in range(len(ppg_signal)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(ppg_signal), i + window_size // 2 + 1)
        smoothed_signal[i] = np.mean(ppg_signal[start_index:end_index])

    return smoothed_signal

"""

def cycle_width(period: pd.DataFrame,
                at_level: float = 0.5,
                ) -> int:
    """Extracts width at defined level of height of averaged period

    Parameters
    ----------
    takes in: Series of averaged period.

    Returns
    -------
    width parameter of waveform at defined level.
    """
    at_level = max(0, min(1, at_level))
    # print("at_level", at_level)
    at_level = 1 - at_level
    # print("at_level", at_level)

    period_squeezed = period.squeeze()
    peaks, _ = find_peaks(period_squeezed, width=len(period_squeezed) // 5)
    results_half = peak_widths(period_squeezed, peaks, rel_height=at_level)
    # print("results_half", results_half)

    # plot_peaks_and_widths(period_squeezed, peaks, results_half)

    width = np.max(results_half[0])
    return int(width)



"""
def sys_duration(period: pd.DataFrame, fs = 1000):
    
    # ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    peaks, _ = find_peaks(period_squeezed, height=np.mean(period_squeezed))
    systolic_phase_duration = len(peaks) / fs  # Convert number of samples to time duration
    
    return int(systolic_phase_duration)


def sys_to_dia_ratio(period: pd.DataFrame, fs = 1000):
    
    # ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    peaks, _ = find_peaks(ppg, height=np.mean(period_squeezed))
    systolic_phase_duration = len(peaks) / fs
    diastolic_phase_duration = len(ppg) / fs - systolic_phase_duration
    ratio = systolic_phase_duration / diastolic_phase_duration
    
    return int(ratio)


def avg_pulse_rate(period: pd.DataFrame, fs = 1000):
    
    # ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    peaks, _ = find_peaks(ppg, height=np.mean(period_squeezed), prominence=1, width=20)
    ibis = np.diff(peaks) / fs # Calculate time intervals between consecutive peaks
    pulse_rates = 60 / ibis  
    avg_pulse_rate = np.mean(pulse_rates)
    
    return int(avg_pulse_rate)


def second_derivative(period: pd.DataFrame):
    
    # ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    averaged_cycle = (moving_average_window(period_squeezed, window_size=100))
    first_derivative = np.diff(averaged_cycle )
    second_derivative = np.diff(first_derivative)
    
    return int(second_derivative)


def dia_peak (period: pd.DataFrame):
    
    
    # ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    peaks_deriv, prop_deriv = find_peaks(moving_average_window(second_derivative(period_squeezed), window_size=40), 
                                         height=np.mean(sec_deriv)
                                        )
    notch = peaks_deriv[3]
    end = peaks_deriv[4]
    peak = (notch + end) / 2
    #peak = np.mean(end, notch)
    
    return int(peak)
    


def dic_notch (period: pd.DataFrame):
    
    #ppg = np.asarray(period).ravel()
    period_squeezed = period.squeeze()
    sec_deriv = moving_average_window(second_derivative(period_squeezed), window_size=100)
    peaks_deriv, prop_deriv = find_peaks(moving_average_window(sec_deriv, window_size=40), 
                                         height=np.mean(sec_deriv)
                                        )
    notch = peaks_deriv[3]
    
    return int(notch)
"""                          

# def sys_starts(period: pd.DataFrame) -> int:
#    """ extracts start of systole at left_bases parameter of find_peaks properties"""
#    period_squeezed = period.squeeze()
#    peaks, properties = find_peaks(period_squeezed, width = 20)
#    sys_start_coord   = properties['left_bases']

#    return int(*sys_start_coord)


def dia_starts(period: pd.DataFrame) -> int:
    """ extracts start of diastole at right_ips parameter of find_peaks properties"""
    period_squeezed = period.squeeze()
    peaks, properties = find_peaks(period_squeezed, width=20)
    dia_start_coord = properties['right_ips']
    dia_start_coord = dia_start_coord[0]
    
    return int(dia_start_coord)


def AUC(period: pd.DataFrame) -> int:
    """ area under curve returns integrated area under the given waveform cycle"""
    period_squeezed = period.squeeze()
    auc = simpson(period_squeezed, dx=5)

    return int(auc)


def peak_coord(period: pd.DataFrame) -> int:
    """ returns X coordinate of the cycle peak"""
    # period_squeezed =
    # peak = np.argmax(period.squeeze())
    # peaks, properties = find_peaks(period_squeezed, prominence=1, width=20)

    return np.argmax(period.squeeze())


# def sys_width(period: pd.DataFrame, level):

    # period_squeezed = period.squeeze()
    # curve_at_lvl = np.argwhere(period_squeezed > level)
    # width_sys = np.argmax(period_squeezed[curve_at_lvl]) + curve_at_lvl[0,0]

    # peaks, _ = find_peaks(period_squeezed)
    # horiz = level * period[peaks]
    # idx = np.argwhere(np.diff(np.sign(period - horiz)) != 0).reshape(-1)
    # width_sys = abs(peaks[0] - idx[0])

    # return width_sys


def sys_width(period: pd.DataFrame, level):

    period = period.apply(lambda col: pd.Series(col.unique()))
    period = period.fillna(0)
    cycle_squeezed = period.squeeze()
    
    # period_np = period.values.squeeze()
    # cycle_lvl = np.argwhere(period_np > level).squeeze()
    # sys_max_lvl = np.argmax(period_np[cycle_lvl]) + cycle_lvl[0]
    # sys_width = sys_max_lvl - cycle_lvl[0]

    cycle_lvl = np.argwhere(cycle_squeezed > level)
    sys_max_lvl = np.argmax(cycle_squeezed[cycle_lvl]) + cycle_lvl[0,0]
    sys_width = sys_max_lvl - cycle_lvl[0,0]

    return sys_width



# def sys_width(period: pd.DataFrame, level):
#    period = period.fillna(0)
#    cycle_squeezed = period.squeeze()
#    cycle_lvl = np.argwhere(cycle_squeezed > level)
#    sys_max_lvl = np.argmax(cycle_squeezed[cycle_lvl]) + cycle_lvl[0,0]
#    sys_width = sys_max_lvl - cycle_lvl[0,0]
#    sys_width_array = np.full(len(period), sys_width)

#    return sys_width_array


def plot_peaks_and_widths(x, peaks, results_at_lev):
    import matplotlib.pyplot as plt
    plt.plot(x, color="blue")
    plt.plot(peaks, x[peaks], "x")
    plt.hlines(*results_at_lev[1:], color="red")
    plt.show()


# ====================================================================
# This part is for testing the functions from this file only:
#
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    x = np.linspace(0, 6 * np.pi, 1000)
    x = np.sin(x) + 0.6 * np.sin(2.6 * x)
    # Find all peaks and calculate their widths at the relative height of 0.5 (contour line at half the prominence height) and 1 (at the lowest contour line at full prominence height).

    peaks, _ = find_peaks(x)
    results_half = peak_widths(x, peaks, rel_height=0.1)
    print(results_half[0])  # widths
    # array([ 64.25172825,  41.29465463,  35.46943289, 104.71586081,
    #         35.46729324,  41.30429622, 181.93835853,  45.37078546])

    results_full = peak_widths(x, peaks, rel_height=1)

    print(results_full[0])  # widths
    # array([181.9396084 ,  72.99284945,  61.28657872, 373.84622694,
    #     61.78404617,  72.48822812, 253.09161876,  79.36860878])
    # Plot signal, peaks and contour lines at which the widths where calculated

    plt.plot(x)
    plt.plot(peaks, x[peaks], "x")
    plt.hlines(*results_half[1:], color="red")
    plt.hlines(*results_full[1:], color="green")
    plt.show()
