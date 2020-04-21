#!/usr/bin/env python3

# Libraries
import argparse
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scikits.samplerate import resample
from scipy import signal
from scipy.fft import fft, fftshift
import numpy as np
import os
import re
import scipy.io.wavfile
import sys
import warnings
import wave

# Constants
male_periodic = "./audio/am1.wav"
female_periodic = "./audio/af2.wav"
sample_rate_16k = 16000.0

def open_wav_file(wav_path, channel_id=0, t_limA_ms=None, t_limB_ms=None, sample_rate = sample_rate_16k):
    """
    Open a WAV audio file and provides a 1-D array with the audio amplitudes
    """
    print("Opening: " + wav_path)
    signal_wave = wave.open(wav_path, "r")
    signal = np.frombuffer(signal_wave.readframes(sample_rate * 100), dtype=np.int16)
    if (t_limA_ms != None) and (t_limB_ms != None):
        signal = signal[t_limA_ms:t_limB_ms]        # Portion of signal
    left, right = signal[0::2], signal[1::2]
    if (channel_id == 0):
        channel = left
    else:
        channel = right
    max_val = np.amax(channel)
    norm_channel = (np.array(channel) * (1.0 / max_val))
    print("Samples: " + str(len(norm_channel)))
    return norm_channel


def plot_wave_time(signal, title, xlim_a=None, xlim_b=None, sample_rate = sample_rate_16k):
    """
    Plot a time domain signal from an audio file
    """
    plt.figure(1)
    plt.title(title)
    N_samples = len(signal)
    dt = 1.0 / sample_rate
    resize_factor = dt * 1000.0
    sig_time = np.arange(N_samples) * resize_factor
    if xlim_a and xlim_b:
        xlim_a *= resize_factor
        xlim_b *= resize_factor
    plt.plot(sig_time, signal)
    plt.xlim(xlim_a, xlim_b)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def get_sub_signal(signal, i_lim, f_lim, sample_rate=sample_rate_16k):
    """
    Provides a sub-signal located between i_lim (ms) and f_lim (ms)
    """
    i_lim_samples = i_lim * sample_rate / 1000
    f_lim_samples = f_lim * sample_rate / 1000
    N_samples = int(f_lim_samples) - int(i_lim_samples)
    sub_signal = np.zeros(N_samples)
    for t_val in range(N_samples):
        sub_signal[t_val] = signal[int(i_lim_samples) + t_val]
    return sub_signal


def plot_time_signal(signal, time_ms, sample_rate=sample_rate_16k, title='', xlims=None):
    """
    Plot a time-domain signal
    """
    t = np.linspace(0, time_ms/1000.0, (time_ms/1000.0) * sample_rate)
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, which ='both')
    plt.axhline(y=0, color='k')
    if xlims:
        plt.xlim(xlims[0], xlims[1])
    plt.ylim(-1.25, 1.25)
    plt.show()


def plot_fft(signal, time, title, xlims):
    """
    Compute the Fast Fourier Transform of the 1D array and plot the obtained spectrum
    """
    fft = np.fft.fft(signal)
    T = time[1] - time[0]
    N = signal.size
    freq = 1 / T
    f = np.linspace(0, freq, N)
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.xlim(xlims)
    abs_fft = np.abs(fft)
    plt.stem(f, abs_fft / np.max(abs_fft))
    plt.grid(True)
    plt.show()
    return (f, (abs_fft / np.max(abs_fft)))


def get_sawtooth(time_ms, frequency, sample_rate=sample_rate_16k):
    """
    Provides a saw-tooth signal
    """
    T = 1.0 / frequency
    t = np.linspace(0, time_ms/1000.0, (time_ms/1000.0) * sample_rate)
    sawtooth = signal.sawtooth(2 * np.pi * frequency * (t - T/2.0))
    return sawtooth


def get_window(win_width_ms, sample_rate, title, win_type="rectangular"):
    """
    Plot a Rectangular, Hann or Hamming window returning it as a 1-D arrary
    """
    n_samples = int(win_width_ms*sample_rate/1000)
    if win_type == "rectangular":
        window = signal.boxcar(n_samples)
        plt.title("Rectangular window")
    elif win_type == "hann":
        window = signal.hann(n_samples)
        plt.title("Hann window")
    elif win_type == "hamming":
        window = signal.hamming(n_samples)
        plt.title("Hamming window")
    t = np.linspace(0, win_width_ms, n_samples)
    plt.plot(t, window)
    plt.ylabel("Amplitude")
    plt.xlabel("Time (ms)")
    plt.title(title)
    plt.grid(True)
    plt.show()
    return window


def plot_window_spectrum(window, f_range, title, sample_rate):
    """
    Plot the normalized spectrum of a window
    """
    plt.figure()
    samples = int(sample_rate + (pow(2, 14) - sample_rate))
    A = fft(window, samples) / (len(window)/2.0)
    freq = np.linspace(f_range[0], f_range[1], len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    plt.plot(freq, response)
    plt.axis([f_range[0], f_range[1], -120, 0])
    plt.title(title)
    plt.ylabel("Normalized magnitude [dB]")
    plt.xlabel("Normalized frequency [cycles/second]")
    plt.grid(True)
    plt.show()


def get_harmonics_magnitude_error(spectrum, freq_array, fund_freq, win_type):
    """
    Provides the relative error in percentage of the harmonics 2f_0, 3f_0 and 4f_0 
    """
    print("==========================================================================")
    print(win_type + " window: Harmonics magnitude error for f_0 = " + str(fund_freq) + " Hz:")
    for k in range (2, 5):
        expected = 1.0 / k
        harmonic = k * fund_freq
        threshold = 5.0 
        comp_low_harmonic = np.abs(harmonic - threshold)
        comp_high_harmonic = np.abs(harmonic + threshold)
        f_index = np.where((freq_array <= comp_high_harmonic) & (freq_array >= comp_low_harmonic))
        real_f = freq_array[f_index]
        magnitude = spectrum[f_index]
        relative_err = np.abs(expected - magnitude) / expected * 100
        print("mag_" + str(k) + "f_0(" + str(harmonic) + " -> " + str(real_f) + ") = " + str(magnitude) + ". Error: " + str(relative_err) + " %")


def main():

    # ITEM 1.A
    time_ms = 100
    male_freq = 105.03
    title = 'Saw-tooth wave of ' + str(male_freq) + ' Hz'
    sawtooth_M = get_sawtooth(time_ms, male_freq)
    plot_time_signal(sawtooth_M, time_ms, title=title)

    female_freq = 204.07
    title = 'Saw-tooth wave of ' + str(female_freq) + ' Hz'
    sawtooth_F = get_sawtooth(time_ms, female_freq)
    plot_time_signal(sawtooth_F, time_ms, title=title)

    # ITEM 1.C
    xlims = (0, 10 * male_freq)
    time = np.linspace(0, time_ms/1000.0, (time_ms/1000.0) * sample_rate_16k)
    title = 'FFT. Spectrum of saw-toot signal of ' + str(male_freq) + ' Hz'
    ## plot_fft(sawtooth_M, time, title, xlims)

    # ITEM 1.D
    win_width_M32 = 32
    title = "Rectangular Window"
    rect_window = get_window(win_width_M32, sample_rate_16k, title, "rectangular")
    title = "Rectangular Window Spectrum"
    xlims = (0, 100)
    time = np.linspace(0, win_width_M32, sample_rate_16k)
    plot_window_spectrum(rect_window, (-0.5, 0.5), title, sample_rate_16k)

    win_width_M64 = 64
    title = "Hann Window"
    hann_window = get_window(win_width_M64, sample_rate_16k, title, "hann")
    title = "Hann Window Spectrum"
    xlims = (0, 100)
    time = np.linspace(0, win_width_M64, sample_rate_16k)
    plot_window_spectrum(hann_window, (-0.5, 0.5), title, sample_rate_16k)

    title = "Hamming Window"
    hamming_window = get_window(win_width_M64, sample_rate_16k, title, "hamming")
    title = "Hamming Window Spectrum"
    xlims = (0, 100)
    time = np.linspace(0, win_width_M64, sample_rate_16k)
    plot_window_spectrum(hamming_window, (-0.5, 0.5), title, sample_rate_16k)


    # Rectangular
    win_len = len(rect_window)
    sawtooth_len = len(sawtooth_M)
    zero_signal = np.zeros(sawtooth_len - win_len)
    complete_rect = np.concatenate((rect_window, zero_signal))
    rect_result = np.multiply(sawtooth_M, complete_rect)
    title = "Result saw-tooth signal after applying the rectangular window"
    plot_time_signal(rect_result, time_ms, title=title, xlims=(0, win_width_M32 / 1000))

    title = "Spectrum of the saw-tooth signal after applying the rectangular window"
    xlims = (0, 10 * male_freq)
    time = np.linspace(0, time_ms/1000.0, (time_ms/1000.0) * sample_rate_16k)
    (freqs, mags) = plot_fft(rect_result, time, title, xlims)
    get_harmonics_magnitude_error(mags, freqs, male_freq, "Rectangular")


    # Hann
    win_len = len(hann_window)
    sawtooth_len = len(sawtooth_M)
    zero_signal = np.zeros(sawtooth_len - win_len)
    complete_hann = np.concatenate((hann_window, zero_signal))
    hann_result = np.multiply(sawtooth_M, complete_hann)
    title = "Result saw-tooth signal after applying the Hann window"
    plot_time_signal(hann_result, time_ms, title=title, xlims=(0, win_width_M64 / 1000))

    title = "Spectrum of the saw-tooth signal after applying the Hann window"
    (freqs, mags) = plot_fft(hann_result, time, title, xlims)
    get_harmonics_magnitude_error(mags, freqs, male_freq, "Hann")

    # Hamming
    win_len = len(hamming_window)
    sawtooth_len = len(sawtooth_M)
    zero_signal = np.zeros(sawtooth_len - win_len)
    complete_rect = np.concatenate((hamming_window, zero_signal))
    hamming_result = np.multiply(sawtooth_M, complete_rect)
    title = "Result saw-tooth signal after applying the Hamming window"
    plot_time_signal(hamming_result, time_ms, title=title, xlims=(0, win_width_M64 / 1000))

    title = "Spectrum of the saw-tooth signal after applying the Hamming window"
    (freqs, mags) = plot_fft(hamming_result, time, title, xlims)
    get_harmonics_magnitude_error(mags, freqs, male_freq, "Hamming")


    # ITEM 2.C
    T_ms = 10.2
    male_freq = 1.0 / (T_ms / 1000)
    N_periods = 15
    a_male_signal = open_wav_file(male_periodic)
    i_lim_ms = 600.0
    f_lim_ms = i_lim_ms + (N_periods * T_ms)
    a_sub_signal = get_sub_signal(a_male_signal, i_lim_ms, f_lim_ms)
    plot_wave_time(a_sub_signal, "A - Masculine: " + str(N_periods) + " Periods")

    title = "Hann Window"
    hann_window = get_window(win_width_M64, sample_rate_16k, title, "hann")
    win_len = len(hann_window)
    male_signal_len = len(a_sub_signal)
    time_ms = male_signal_len * 1000 / sample_rate_16k
    zero_signal = np.zeros(male_signal_len - win_len)
    complete_hann = np.concatenate((hann_window, zero_signal))
    hann_result = np.multiply(a_sub_signal, complete_hann)
    title = "Result of a periodic male voiced signal after applying the Hann window"
    plot_wave_time(hann_result, title, 0, win_width_M64)

    title = "Spectrum of the periodic male voiced signal after applying the Hann window"
    xlims = (0, 5 * male_freq)
    time = np.linspace(0, win_width_M64 / 1000, win_width_M64 * sample_rate_16k / 1000)
    (freqs, mags) = plot_fft(hann_result, time, title, xlims)


    # ITEM 2.D
    T_ms = 4.94
    female_freq = 1.0 / (T_ms / 1000)
    N_periods = 10.0
    a_female_signal = open_wav_file(female_periodic)
    i_lim = 400.0
    f_lim = i_lim + (N_periods * T_ms)
    a_sub_signal = get_sub_signal(a_female_signal, i_lim, f_lim)
    plot_wave_time(a_sub_signal, "A - Femenine: " + str(int(N_periods)) + " Periods")

    title = "Hann Window"
    hann_window = get_window(win_width_M32, sample_rate_16k, title, "hann")
    win_len = len(hann_window) 
    male_signal_len = len(a_sub_signal)
    time_ms = male_signal_len * 1000 / sample_rate_16k
    zero_signal = np.zeros(male_signal_len - win_len)
    complete_hann = np.concatenate((hann_window, zero_signal))
    hann_result = np.multiply(a_sub_signal, complete_hann)
    title = "Result of a periodic female voiced signal after applying the Hann window"
    plot_wave_time(hann_result, title, 0, win_width_M32)

    title = "Spectrum of the periodic female voiced signal after applying the Hann window"
    time = np.linspace(0, win_width_M32 / 1000, win_width_M32 * sample_rate_16k / 1000)
    xlims = (0, 10 * female_freq)
    (freqs, mags) = plot_fft(hann_result, time, title, xlims)

if __name__== "__main__":
  warnings.simplefilter("ignore")
  main()
