#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
from matplotlib.mlab import specgram
from scikits.samplerate import resample
from scipy import signal
import numpy as np
import os
import re
import scipy.io.wavfile
import sys
import warnings
import wave

impulse_path = "./audio/impulse.wav"
voice_sample = "./audio/8m1.wav"
voice_number = "./audio/4m1_44k.wav"
voice_poem = "./audio/pirata.wav"
radio_path = "./audio/AM870_890_910.wav"

sample_rate_44k = 44100.0
sample_rate_2M = 2000000.0
sample_rate_4M = 4000000.0
sample_rate_870k = 870000.0

def print_list(input_list, list_name="List"):
    print("=======================")
    print(list_name + " Content:")
    print("=======================")
    for item in input_list:
        print(item)

def open_wav_file_mono(filename):
    (fm,s) = scipy.io.wavfile.read(filename)
    s = s / np.iinfo(s.dtype).max
    return (fm, s)

def open_wav_file(wav_path, channel_id=0, t_limA_ms=None, t_limB_ms=None, sample_rate = sample_rate_44k):
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


def plot_wave_time(signal, title, xlim_a=None, xlim_b=None, sample_rate = sample_rate_44k):
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


def plot_two_signals(signal_a, title_1, signal_b, title_2, xlimits_a = None, xlimits_b = None, sample_rate = sample_rate_44k):
    N_samples_a = len(signal_a)
    N_samples_b = len(signal_b)

    dt = 1.0 / sample_rate
    resize_factor = dt * 1000.0

    sig_time_a = np.arange(N_samples_a) * resize_factor
    sig_time_b = np.arange(N_samples_b) * resize_factor

    ax1 = plt.subplot(211)

    if xlimits_a:
        ax1.set_xlim([xlimits_a[0], xlimits_a[1]])

    ax1.set_title(title_1, y = -0.225)
    ax1.plot(sig_time_a, signal_a)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    ax2 = plt.subplot(212)

    if xlimits_b:
        ax2.set_xlim([xlimits_b[0], xlimits_b[1]])

    ax2.set_title(title_2, y = -0.225)
    ax2.plot(sig_time_b, signal_b)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def get_simulation(h, x):
    y, zf = signal.lfilter (h, 1, x, zi = np.zeros(len(h) - 1))
    y = np.append(y, zf)
    return y


def get_sub_signal(signal, i_lim, f_lim):
    N_samples = int(f_lim) - int(i_lim)
    sub_signal = np.zeros(N_samples)
    for t_val in range(N_samples):
        sub_signal[t_val] = signal[int(i_lim) + t_val]
    return sub_signal


def low_pass_filter(input_signal, fs, fc, gain = 1.0):
    order = 5
    w = fc / (fs / 2.0)                          # Normalize the frequency
    b, a = signal.butter(order, w, 'low', analog = False)
    output = signal.filtfilt(b, a, input_signal)
    return gain * output


def main():

    # ITEM 1.A
    impulse = open_wav_file(impulse_path)
    title = "Impulse Response of the Bathroom"
    plot_wave_time(impulse, title)


    # ITEM 1.B
    voice_eight = open_wav_file(voice_sample)
    title_1 = "(a) Voice Sample: Eight"
    sim_signal = get_simulation(impulse, voice_eight)
    title_2 = "(b) Simulation of voice transmission by a bathroom based on the impulse response of the bathroom"

    plot_two_signals(voice_eight, title_1, sim_signal, title_2, xlimits_b=(0,702))
    print("Saving: result_eight.wav...")
    scipy.io.wavfile.write("./results/result_eight.wav", int(sample_rate_44k), sim_signal)

    voice_pirat = open_wav_file(voice_poem)
    title_1 = "(a) Voice Sample: The Pirat's Song"
    sim_signal = get_simulation(impulse, voice_pirat)
    title_2 = "(b) Simulation of voice transmission by a bathroom based on the impulse response of the bathroom"

    plot_two_signals(voice_pirat, title_1, sim_signal, title_2)
    print("Saving: result_pirat.wav...")
    scipy.io.wavfile.write("./results/result_pirat.wav", int(sample_rate_44k), sim_signal)


    # ITEM 2.A
    voice_four = open_wav_file(voice_number)
    timeA = 186.5
    timeB = 188.5
    limA = timeA * sample_rate_44k / 1000.0
    limB = timeB * sample_rate_44k / 1000.0
    #zoomed_signal = get_sub_signal(voice_four, limA, limB)
    title_1 = "(a) Original Voice Sample: Four"
    title_2 = "(b) Two milliseconds zoomed signal at the interval: [" + str(timeA) + " ms - " + str(timeB) + " ms]"
    plot_two_signals(voice_four, title_1, voice_four, title_2, xlimits_b = (timeA, timeB))

    

    # ITEM 2.B
    ratio = sample_rate_4M / sample_rate_44k
    converter = "sinc_best"
    N_oversampled = int(len(voice_four) * ratio)
    #oversampled_signal = resample(voice_four, ratio, converter).astype(voice_four.dtype)
    oversampled_signal = signal.resample(voice_four, N_oversampled)
    limA = timeA * sample_rate_4M / 1000.0
    limB = timeB * sample_rate_4M / 1000.0
    t = np.linspace(0, N_oversampled, N_oversampled, endpoint = False)
    carrier_f = sample_rate_870k / sample_rate_4M
    carrier = np.cos(2.0 * np.pi * carrier_f * t)
    module_signal = np.multiply(oversampled_signal, carrier)
    title_1 = "(a) Modulated Voice Signal: Four"
    title_2 = "(b) Two milliseconds zoomed modulated-signal at the interval: [" + str(timeA) + " ms - " + str(timeB) + " ms]"
    plot_two_signals(module_signal, title_1, module_signal, title_2, xlimits_b = (timeA, timeB), sample_rate = sample_rate_4M)


    # ITEM 2.C
    audible_f_lim = 20000
    demodule_signal = np.multiply(module_signal, carrier)
    filtered_signal = low_pass_filter(demodule_signal, sample_rate_4M, audible_f_lim)
    title_1 = "(a) Demodulated Voice Signal: Four"
    title_2 = "(b) Filtered Voice Signal: Four (audible range)"
    plot_two_signals(demodule_signal, title_1, filtered_signal, title_2, sample_rate = sample_rate_4M)
    title_1 = "(a) Two milliseconds of demodulated voice signal: Four"
    title_2 = "(b) Two milliseconds of filtered voice signal: Four (audible range)"
    plot_two_signals(demodule_signal, title_1, filtered_signal, title_2, xlimits_a = (timeA, timeB), xlimits_b = (timeA, timeB), sample_rate = sample_rate_4M)
    scipy.io.wavfile.write("./results/result_filtered_four.wav", int(sample_rate_4M), filtered_signal)


    # ITEM 2.D
    record_ucr_f = 870000.0
    record_her_f = 890000.0
    record_bbn_f = 910000.0
    audible_f_lim = 20000.0

    fs_wav, radio_record = open_wav_file_mono(radio_path)
    N_samples = len(radio_record)
    t = np.linspace(0, N_samples, N_samples, endpoint = False)

    lims = (1000, 1001)

    # UCR
    ucr_f = record_ucr_f / fs_wav
    ucr_carrier = np.cos(2.0 * np.pi * ucr_f * t)
    ucr_demoduled = np.multiply(radio_record, ucr_carrier)
    filtered_ucr = low_pass_filter(ucr_demoduled, fs_wav, audible_f_lim)
    plot_two_signals(ucr_demoduled, "One millisecond of demodulated UCR signal", filtered_ucr, "One millisecond of filtered UCR signal", lims, lims, fs_wav)
    scipy.io.wavfile.write("./results/result_UCR_870k.wav", int(fs_wav), filtered_ucr)

    lims = (4000, 4001)

    # Heredia
    her_f = record_her_f / fs_wav
    her_carrier = np.cos(2.0 * np.pi * her_f * t)
    her_demoduled = np.multiply(radio_record, her_carrier)
    filtered_her = low_pass_filter(her_demoduled, fs_wav, audible_f_lim)
    plot_two_signals(her_demoduled, "One millisecond of demodulated Heredia Signal", filtered_her, "One millisecond of filtered  Heredia signal", lims, lims, fs_wav)
    scipy.io.wavfile.write("./results/result_HER_890k.wav", int(fs_wav), filtered_her)

    lims = (2000, 2001)

    # BBN
    bbn_f = record_bbn_f / fs_wav
    bbn_carrier = np.cos(2.0 * np.pi * bbn_f * t)
    bbn_demoduled = np.multiply(radio_record, bbn_carrier)
    filtered_bbn = low_pass_filter(bbn_demoduled, fs_wav, audible_f_lim, gain = 10.0)
    plot_two_signals(bbn_demoduled, "One millisecond of demodulated BBN Signal", filtered_bbn, "One millisecond of filtered BBN signal", lims, lims, fs_wav)
    scipy.io.wavfile.write("./results/result_BBN_910k.wav", int(fs_wav), filtered_bbn)


if __name__== "__main__":
  warnings.simplefilter("ignore")
  main()
