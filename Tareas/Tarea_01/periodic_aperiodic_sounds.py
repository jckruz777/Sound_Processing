#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
import numpy as np
import os
import re
import sys
import warnings
import wave

m_folder = "M/"
f_folder = "F/"

periodic = ["a", "e", "i", "o", "u", "m"]
aperiodic = ["f", "s"]

# Obtained by seeing the Audacity signal plots

#m1_periods_ms = [156, 148, 148, 150, 163, 159]
#f2_periods_ms = [76, 80, 79, 79, 82, 77]
m1_periods_ms = [163, 159, 148, 150, 148, 156]
m2_periods_ms = [155, 149, 154, 159, 150, 150]
f1_periods_ms = [72, 76, 75, 73, 80, 73]
f2_periods_ms = [79, 82, 79, 80, 76, 77]

m1_periodic_list = []
m1_aperiodic_list = []
m2_periodic_list = []
m2_aperiodic_list = []
f1_periodic_list = []
f1_aperiodic_list = []
f2_periodic_list = []
f2_aperiodic_list = []

sample_rate = 16000

def get_paths_from_pattern(main_dir, pattern):
    for root, dirs, files in os.walk(main_dir):
        for file in files:
            base_path = os.path.join(root,file)
            abs_path = os.path.abspath(base_path)
            filename = re.search(pattern, abs_path)
            if(filename):
                sound = filename.group(1)[0]
                gender = filename.group(1)[1]
                n_sample = filename.group(1)[2]
                if gender == 'm':
                    if n_sample == '1':
                        if sound in periodic:
                            m1_periodic_list.append(abs_path)
                        elif sound in aperiodic:
                            m1_aperiodic_list.append(abs_path)
                    else:
                        if sound in periodic:
                            m2_periodic_list.append(abs_path)
                        elif sound in aperiodic:
                            m2_aperiodic_list.append(abs_path)
                else:
                    if n_sample == '1':
                        if sound in periodic:
                            f1_periodic_list.append(abs_path)
                        elif sound in aperiodic:
                            f1_aperiodic_list.append(abs_path)
                    else:
                        if sound in periodic:
                            f2_periodic_list.append(abs_path)
                        elif sound in aperiodic:
                            f2_aperiodic_list.append(abs_path)

def get_wav_paths(main_dir):
    m_pattern = 'M/(.*).wav'
    f_pattern = 'F/(.*).wav'
    get_paths_from_pattern(main_dir, m_pattern)
    get_paths_from_pattern(main_dir, f_pattern)

def print_list(input_list, list_name="List"):
    print("=======================")
    print(list_name + " Content:")
    print("=======================")
    for item in input_list:
        print(item)

def open_wav_file(wav_path, channel_id=0, t_limA_ms=None, t_limB_ms=None):
    signal_wave = wave.open(wav_path, "r")
    signal = np.frombuffer(signal_wave.readframes(sample_rate), dtype=np.int16)
    if (t_limA_ms == None) and (t_limB_ms == None):
        signal = signal[:]                          # Whole signal
    else:
        signal = signal[t_limA_ms:t_limB_ms]        # Portion of signal
    left, right = signal[0::2], signal[1::2]
    if (channel_id == 0):
        channel = left
    else:
        channel = right
    max_val = np.amax(channel)
    norm_channel = (np.array(channel) * (1.0 / max_val))
    return norm_channel

def plot_wave_time(signal, title, xlim_a=None, xlim_b=None, period=None):
    plt.figure(1)
    plt.title(title)
    N_samples = len(signal)
    dt = 1.0 / sample_rate
    resize_factor = dt * 1000.0
    sig_time = np.arange(N_samples) * resize_factor
    if xlim_a and xlim_b:
        if period:
            period_signal = [0] * (int(xlim_a + period))
            for t_val in range(int(xlim_a), int(xlim_a + period)):
                period_signal[t_val] = signal[t_val]
            NT_samples = len(period_signal)
            per_time = np.arange(NT_samples) * resize_factor
            plt.plot(sig_time, signal, "g", per_time, period_signal, "b")
            xlim_a *= resize_factor
            xlim_b *= resize_factor
        else:
            xlim_a *= resize_factor
            xlim_b *= resize_factor
            plt.plot(sig_time, signal)
    else:
            xlim_a *= resize_factor
            xlim_b *= resize_factor
            plt.plot(sig_time, signal)
    plt.xlim(xlim_a, xlim_b)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def print_fund_frequencies(input_periods, name):
    print("\n*****************")
    print("*      " + name + "       *")
    print("*****************")
    for i in range (len(input_periods)):
        period = input_periods[i] / sample_rate * 1000.0                       # ms
        print("T_" + periodic[i] + " = " + str(period) + " ms")
        fund_freq = 1.0 / period * 1000.0                                      # Hz
        print("F_" + periodic[i] + " = " + str(fund_freq) + " Hz\n")

def calc_fmpd(N, k, s):
    top_lim = N - k
    n = 1
    sum_dk = 0
    while (n <= (top_lim)):
        s_n = s[n - 1]
        s_nk = s[n - 1 + k]
        sum_dk += abs(s_n + s_nk)
        n += 1
    return (1.0/(top_lim)) * sum_dk

def plot_fmpd(signal, period, title):
    N = len(signal)
    fmpd_result = np.zeros(N)
    for k in range(N):
        fmpd_val = calc_fmpd(N, k, signal)
        fmpd_result[k] = fmpd_val
    plt.figure(1)
    plt.title(title)
    dt = 1.0 / sample_rate
    resize_factor = dt * 1000.0
    sub_time = np.arange(N) * resize_factor
    plt.text(period, np.max(fmpd_result), r' $T$_' + title[-1])
    plt.plot(sub_time, fmpd_result, "b", [period, period], [0, np.max(fmpd_result)], "r-h")
    plt.xlabel('Time (ms)')
    plt.ylabel('FMPD')
    plt.grid(True)
    plt.show()

def get_sub_signal(signal, i_lim, f_lim):
    N_samples = int(f_lim) - int(i_lim)
    period_signal = np.zeros(N_samples)
    for t_val in range(N_samples):
        period_signal[t_val] = signal[int(i_lim) + t_val]
    return period_signal

def plot_frequency_domain(signal, f_0, title):
    N = len(signal)
    fm = sample_rate
    (S, f ,tt)  = specgram (signal, N, fm, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, mode="magnitude")
    S = np.reshape(S, len(S))   # 1D-Array
    S = S/(N/4);
    plt.figure(1)
    plt.title(title)
    if(f_0 > 0):
        plt.text(f_0, np.max(S), r' $f$_' + title[-1])
        plt.text(2.0 * f_0, np.max(S), r' $2f$_' + title[-1])
        plt.text(3.0 * f_0, np.max(S), r' $3f$_' + title[-1])
        plt.plot(f, S, "b", [f_0, f_0], [0, np.max(S)], "r-h", [2.0 * f_0, 2.0 * f_0], [0, np.max(S)], "r-h", [3.0 * f_0, 3.0 * f_0], [0, np.max(S)], "r-h")
        plt.xlim(0, 10.0 * f_0)
    else:
        plt.plot(f, S, "b")
    plt.xlabel("Frequency (Hz)")
    plt.grid(True)
    plt.show()
    return f, S

def plot_sonority(f, S):
    fm = sample_rate
    fmax = fm /2; # Nyquist Frequency
    x = np.linspace (0, 5.7 * np.log2 (1 + fmax / 230), 1000)
    fc = ( 2 ** ( x / 5.7) - 1 ) * 230
    S = np.sqrt(np.interp(fc, f, S))
    plt.figure(1)
    plt.plot(x, S / max(S))
    plt.ylabel("Relative specific sonority (sones)")
    plt.xlabel("Distance from the base of the cochlea (mm)")
    plt.grid(True)
    plt.show()

def main():
    # WAV PATHS
    get_wav_paths("./")

    #print_list(m1_periodic_list, "M1 Periodic")
    #print_list(m1_aperiodic_list, "M1 Aperiodic")
    #print_list(m2_periodic_list, "M2 Periodic")
    #print_list(m2_aperiodic_list, "M2 Aperiodic")
    #print_list(f1_periodic_list, "F1 Periodic")
    #print_list(f1_aperiodic_list, "F1 Aperiodic")
    #print_list(f2_periodic_list, "F2 Periodic")
    #print_list(f2_aperiodic_list, "F2 Aperiodic")

    sorted_m1_periodic_list = []
    sorted_m1_periodic_list.append(m1_periodic_list[4]) #A
    sorted_m1_periodic_list.append(m1_periodic_list[5]) #E
    sorted_m1_periodic_list.append(m1_periodic_list[2]) #I
    sorted_m1_periodic_list.append(m1_periodic_list[3]) #O
    sorted_m1_periodic_list.append(m1_periodic_list[1]) #U
    sorted_m1_periodic_list.append(m1_periodic_list[0]) #M
    #print_list(sorted_m1_periodic_list, "M1 Periodic")

    sorted_f2_periodic_list = []
    sorted_f2_periodic_list.append(f2_periodic_list[2]) #A
    sorted_f2_periodic_list.append(f2_periodic_list[4]) #E
    sorted_f2_periodic_list.append(f2_periodic_list[3]) #I
    sorted_f2_periodic_list.append(f2_periodic_list[1]) #O
    sorted_f2_periodic_list.append(f2_periodic_list[0]) #U
    sorted_f2_periodic_list.append(f2_periodic_list[5]) #M
    #print_list(sorted_f2_periodic_list, "F2 Periodic")


    i_lim = 3200.0

    #---------
    # ITEM 1.
    # POINT A
    #---------
    # M1: A
    f_lim = i_lim + (8.0 * m1_periods_ms[0])
    channel = open_wav_file(sorted_m1_periodic_list[0], 0)
    plot_wave_time(channel, "A - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[0])
    
    # M1: E
    f_lim = i_lim + (8.0 * m1_periods_ms[1])
    channel = open_wav_file(sorted_m1_periodic_list[1], 0)
    plot_wave_time(channel, "E - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[1])

    # M1: I
    f_lim = i_lim + (8.0 * m1_periods_ms[2])
    channel = open_wav_file(sorted_m1_periodic_list[2], 0)
    plot_wave_time(channel, "I - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[2])

    # M1: O
    f_lim = i_lim + (8.0 * m1_periods_ms[3])
    channel = open_wav_file(sorted_m1_periodic_list[3], 0)
    plot_wave_time(channel, "O - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[3])

    # M1: U
    f_lim = i_lim + (8.0 * m1_periods_ms[4])
    channel = open_wav_file(sorted_m1_periodic_list[4], 0)
    plot_wave_time(channel, "U - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[4])

    # M1: M
    f_lim = i_lim + (8.0 * m1_periods_ms[5])
    channel = open_wav_file(sorted_m1_periodic_list[5], 0)
    plot_wave_time(channel, "M - Masculine #1: 8 Periods", i_lim, f_lim, m1_periods_ms[5])

    #--------
    # ITEM 1.
    # POINT B
    #--------
    print("******************************************")

    # M1
    print_fund_frequencies(m1_periods_ms, "M1")

    # M2: 
    print_fund_frequencies(m2_periods_ms, "M2")

    # F1:
    print_fund_frequencies(f1_periods_ms, "F1")

    # F2:
    print_fund_frequencies(f2_periods_ms, "F2")

    #--------
    # ITEM 1.
    # POINT C
    #--------
    print("******************************************")

    #fmpd_periods = [9.6, 9.1, 9.2, 9.4, 10, 9.7]
    fmpd_periods = [10, 9.7, 9.2, 9.4, 9.1, 9.6]

    # M1: A
    f_lim = i_lim + (8.0 * m1_periods_ms[0])
    signal = open_wav_file(sorted_m1_periodic_list[0], 0)
    sub_signal_a = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - A"
    T_a = fmpd_periods[0]
    plot_fmpd(sub_signal_a, T_a, title)
    f_a = 1.0 / fmpd_periods[0] * 1000.0
    print("Period for " + title + " = " + str(T_a) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_a) + " Hz\n")

    # M1: E
    f_lim = i_lim + (8.0 * m1_periods_ms[1])
    signal = open_wav_file(sorted_m1_periodic_list[1], 0)
    sub_signal_e = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - E"
    T_e = fmpd_periods[1]
    plot_fmpd(sub_signal_e, T_e, title)
    f_e = 1.0 / fmpd_periods[1] * 1000.0
    print("Period for " + title + " = " + str(T_e) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_e) + " Hz\n")

    # M1: I
    f_lim = i_lim + (8.0 * m1_periods_ms[2])
    signal = open_wav_file(sorted_m1_periodic_list[2], 0)
    sub_signal_i = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - I"
    T_i = fmpd_periods[2]
    plot_fmpd(sub_signal_i, T_i, title)
    f_i = 1.0 / fmpd_periods[2] * 1000.0
    print("Period for " + title + " = " + str(T_i) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_i) + " Hz\n")

    # M1: O
    f_lim = i_lim + (8.0 * m1_periods_ms[3])
    signal = open_wav_file(sorted_m1_periodic_list[3], 0)
    sub_signal_o = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - O"
    T_o = fmpd_periods[3]
    plot_fmpd(sub_signal_o, T_o, title)
    f_o = 1.0 / fmpd_periods[3] * 1000.0
    print("Period for " + title + " = " + str(T_o) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_o) + " Hz\n")

    # M1: U
    f_lim = i_lim + (8.0 * m1_periods_ms[4])
    signal = open_wav_file(sorted_m1_periodic_list[4], 0)
    sub_signal_u = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - U"
    T_u = fmpd_periods[4]
    plot_fmpd(sub_signal_u, T_u, title)
    f_u = 1.0 / fmpd_periods[4] * 1000.0
    print("Period for " + title + " = " + str(T_u) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_u) + " Hz\n")

    # M1: M
    f_lim = i_lim + (8.0 * m1_periods_ms[5])
    signal = open_wav_file(sorted_m1_periodic_list[5], 0)
    sub_signal_m = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Masculine #1 - M"
    T_m = fmpd_periods[5]
    plot_fmpd(sub_signal_m, T_m, title)
    f_m = 1.0 / fmpd_periods[5] * 1000.0
    print("Period for " + title + " = " + str(T_m) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_m) + " Hz\n")

    #--------
    # ITEM 1.
    # POINT D
    #--------
    print("******************************************")

    title = "Spectrum of Masculine #1 - A"
    f_A, S_A = plot_frequency_domain(sub_signal_a, f_a, title)
    print("\nHarmonies: " + title)
    print("f_A = " + str(f_a))
    print("2f_A = " + str(2.0 * f_a))
    print("3f_A = " + str(3.0 * f_a))

    title = "Spectrum of Masculine #1 - E"
    f_E, S_E = plot_frequency_domain(sub_signal_e, f_e, title)
    print("\nHarmonies: " + title)
    print("f_E = " + str(f_e))
    print("2f_E = " + str(2.0 * f_e))
    print("3f_E = " + str(3.0 * f_e))

    title = "Spectrum of Masculine #1 - I"
    f_I, S_I = plot_frequency_domain(sub_signal_i, f_i, title)
    print("\nHarmonies: " + title)
    print("f_I = " + str(f_i))
    print("2f_I = " + str(2.0 * f_i))
    print("3f_I = " + str(3.0 * f_i))

    title = "Spectrum of Masculine #1 - O"
    f_O, S_O = plot_frequency_domain(sub_signal_o, f_o, title)
    print("\nHarmonies: " + title)
    print("f_O = " + str(f_o))
    print("2f_O = " + str(2.0 * f_o))
    print("3f_O = " + str(3.0 * f_o))

    title = "Spectrum of Masculine #1 - U"
    f_U, S_U = plot_frequency_domain(sub_signal_u, f_u, title)
    print("\nHarmonies: " + title)
    print("f_U = " + str(f_u))
    print("2f_U = " + str(2.0 * f_u))
    print("3f_U = " + str(3.0 * f_u))

    title = "Spectrum of Masculine #1 - M"
    f_M, S_M = plot_frequency_domain(sub_signal_m, f_m, title)
    print("\nHarmonies: " + title)
    print("f_M = " + str(f_m))
    print("2f_M = " + str(2.0 * f_m))
    print("3f_M = " + str(3.0 * f_m))

    #--------
    # ITEM 1.
    # POINT E
    #--------
    title = "Relative Specific Sonority of Masculine #1 - A"
    plot_sonority(f_A, S_A)

    title = "Relative Specific Sonority of Masculine #1 - E"
    plot_sonority(f_E, S_E)

    title = "Relative Specific Sonority of Masculine #1 - I"
    plot_sonority(f_I, S_I)

    title = "Relative Specific Sonority of Masculine #1 - O"
    plot_sonority(f_O, S_O)

    title = "Relative Specific Sonority of Masculine #1 - U"
    plot_sonority(f_U, S_U)

    title = "Relative Specific Sonority of Masculine #1 - M"
    plot_sonority(f_M, S_M)

    #--------
    # ITEM 2.
    #--------

    # F2: A
    f_lim = i_lim + (8.0 * f2_periods_ms[0])
    channel = open_wav_file(sorted_f2_periodic_list[0], 0)
    plot_wave_time(channel, "A - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[0])
    
    # F2: E
    f_lim = i_lim + (8.0 * f2_periods_ms[1])
    channel = open_wav_file(sorted_f2_periodic_list[1], 0)
    plot_wave_time(channel, "E - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[1])

    # F2: I
    f_lim = i_lim + (8.0 * f2_periods_ms[2])
    channel = open_wav_file(sorted_f2_periodic_list[2], 0)
    plot_wave_time(channel, "I - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[2])

    # F2: O
    f_lim = i_lim + (8.0 * f2_periods_ms[3])
    channel = open_wav_file(sorted_f2_periodic_list[3], 0)
    plot_wave_time(channel, "O - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[3])

    # F2: U
    f_lim = i_lim + (8.0 * f2_periods_ms[4])
    channel = open_wav_file(sorted_f2_periodic_list[4], 0)
    plot_wave_time(channel, "U - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[4])

    # F2: M
    f_lim = i_lim + (8.0 * f2_periods_ms[5])
    channel = open_wav_file(sorted_f2_periodic_list[5], 0)
    plot_wave_time(channel, "M - Femenine #2: 8 Periods", i_lim, f_lim, f2_periods_ms[5])


    #--------

    fmpd_fperiods = [77.8, 80.9, 78.4, 79.5, 76, 76.5]

    # F2: A
    f_lim = i_lim + (8.0 * f2_periods_ms[0])
    signal = open_wav_file(sorted_f2_periodic_list[0], 0)
    sub_signal_a = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - A"
    resize_factor = 1.0/16.0
    T_a = fmpd_fperiods[0] * resize_factor
    plot_fmpd(sub_signal_a, T_a, title)
    f_a = 1.0 / T_a * 1000.0
    print("Period for " + title + " = " + str(T_a) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_a) + " Hz\n")

    # F2: E
    f_lim = i_lim + (8.0 * f2_periods_ms[1])
    signal = open_wav_file(sorted_f2_periodic_list[1], 0)
    sub_signal_e = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - E"
    T_e = fmpd_fperiods[1] * resize_factor
    plot_fmpd(sub_signal_e, T_e, title)
    f_e = 1.0 / T_e * 1000.0
    print("Period for " + title + " = " + str(T_e) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_e) + " Hz\n")

    # F2: I
    f_lim = i_lim + (8.0 * f2_periods_ms[2])
    signal = open_wav_file(sorted_f2_periodic_list[2], 0)
    sub_signal_i = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - I"
    T_i = fmpd_fperiods[2]  * resize_factor
    plot_fmpd(sub_signal_i, T_i, title)
    f_i = 1.0 / T_i * 1000.0
    print("Period for " + title + " = " + str(T_i) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_i) + " Hz\n")

    # F2: O
    f_lim = i_lim + (8.0 * f2_periods_ms[3])
    signal = open_wav_file(sorted_f2_periodic_list[3], 0)
    sub_signal_o = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - O"
    T_o = fmpd_fperiods[3]  * resize_factor
    plot_fmpd(sub_signal_o, T_o, title)
    f_o = 1.0 / T_o * 1000.0
    print("Period for " + title + " = " + str(T_o) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_o) + " Hz\n")

    # F2: U
    f_lim = i_lim + (8.0 * f2_periods_ms[4])
    signal = open_wav_file(sorted_f2_periodic_list[4], 0)
    sub_signal_u = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - U"
    T_u = fmpd_fperiods[4] * resize_factor
    plot_fmpd(sub_signal_u, T_u, title)
    f_u = 1.0 / T_u * 1000.0
    print("Period for " + title + " = " + str(T_u) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_u) + " Hz\n")

    # F2: M
    f_lim = i_lim + (8.0 * f2_periods_ms[5])
    signal = open_wav_file(sorted_f2_periodic_list[5], 0)
    sub_signal_m = get_sub_signal(signal, i_lim, f_lim)
    title = "FMPD Femenine #2 - M"
    T_m = fmpd_fperiods[5] * resize_factor
    plot_fmpd(sub_signal_m, T_m, title)
    f_m = 1.0 / T_m * 1000.0
    print("Period for " + title + " = " + str(T_m) + " ms")
    print("Fund. Frequency for " + title + " = " + str(f_m) + " Hz\n")
   

    #---------
    # ITEM NP.
    # POINT 1
    #---------

    i_lim = 1000.0
    f_lim = 3000.0
    channel_f = open_wav_file(m1_aperiodic_list[0], 0)
    plot_wave_time(channel_f, "F - Masculine #1: Signal Segment", i_lim, f_lim)
    channel_s = open_wav_file(m1_aperiodic_list[1], 0)
    plot_wave_time(channel_s, "S - Masculine #1: Signal Segment", i_lim, f_lim)
    
    #---------
    # ITEM NP.
    # POINT 2
    #---------

    sub_signal_f = get_sub_signal(channel_f, i_lim, f_lim)
    title = "Spectrum of Masculine #1 - F"
    f_F, S_F = plot_frequency_domain(channel_s, 0, title)

    sub_signal_s = get_sub_signal(channel_s, i_lim, f_lim)
    title = "Spectrum of Masculine #1 - S"
    f_S, S_S = plot_frequency_domain(channel_s, 0, title)

    #---------
    # ITEM NP.
    # POINT 3
    #---------

    title = "Relative Specific Sonority of Masculine #1 - F"
    plot_sonority(f_F, S_F)

    title = "Relative Specific Sonority of Masculine #1 - S"
    plot_sonority(f_S, S_S)

if __name__== "__main__":
  warnings.simplefilter("ignore")
  main()