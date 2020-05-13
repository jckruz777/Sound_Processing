#!/usr/bin/env python3

# Libraries

from matplotlib.mlab import specgram
from numpy.linalg import norm
from scikits.samplerate import resample
from scipy import signal, fftpack
from scipy.io.wavfile import read
from scipy.fft import fft, fftshift
from scipy.signal import lfilter, hamming

import scipy as sp
import argparse
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import re
import scipy.io.wavfile
import sys
import warnings
import wave

# Constants
a_male_periodic = "./audio/am1_mono.wav"
e_male_periodic = "./audio/em1_mono.wav"
i_male_periodic = "./audio/im1_mono.wav"
o_male_periodic = "./audio/om1_mono.wav"
u_male_periodic = "./audio/um1_mono.wav"
a_male_whisper = "./audio/aw1_mono.wav"
e_male_whisper = "./audio/ew1_mono.wav"
i_male_whisper = "./audio/iw1_mono.wav"
o_male_whisper = "./audio/ow1_mono.wav"
u_male_whisper = "./audio/uw1_mono.wav"

result_folder = "./results/"
sample_rate_16k = 16000.0

T_v = 50 #ms
limA_ms = 200
limB_ms = limA_ms + T_v

# Global Variables
norm_sqrt_fft_a = None
norm_sqrt_fft_e = None
norm_sqrt_fft_i = None
norm_sqrt_fft_o = None
norm_sqrt_fft_u = None
freqs_a = None
freqs_e = None
freqs_i = None
freqs_o = None
freqs_u = None

K_a = round(2 * sample_rate_16k / 1000)
K_e = round(1 * sample_rate_16k / 1000)
K_i = round(1.5 * sample_rate_16k / 1000)
K_o = round(2 * sample_rate_16k / 1000)
K_u = round(2.25 * sample_rate_16k / 1000)

############################################ UTILS ############################################


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Creating directory: " + folder_path)


############################################ PLOTS ############################################


def plot_time_signal(signal, title, filename, xlim_a=None, xlim_b=None, sample_rate = sample_rate_16k):
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
    plt.savefig(result_folder + filename)
    plt.show()


def plot_fft(signal, time, title, filename, xlims=(-0.5,3000)):
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
    if xlims:
        plt.xlim(xlims)
    abs_fft = np.abs(fft)
    norm_abs_fft = abs_fft / np.max(abs_fft)
    #plt.stem(f, norm_abs_fft)
    plt.plot(f, norm_abs_fft)
    plt.grid(True)
    plt.savefig(result_folder + filename)
    plt.show()
    return (f, abs_fft)


def plot_sqrt_fft(abs_sqrt_fft, freqs, title, filename, xlims=(-0.5,3000)):
    plt.title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Sqrt Frequency [√Hz]")
    if xlims:
        plt.xlim(xlims)
    #plt.stem(freqs, abs_sqrt_fft)
    plt.plot(freqs, abs_sqrt_fft)
    plt.grid(True)
    plt.savefig(result_folder + filename)
    plt.show()


def plot_fft_transfer_function(filename, title, freqs_a, abs_sqrt_fft, legend_a, freqs_b, abs_tfer_func, legend_b, xlims=(-0.5,3000), labels=None):
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [√Hz]")
    if xlims:
        plt.xlim(xlims)
    ax.plot(freqs_a, abs_sqrt_fft, color='b', label=legend_a)
    ax.plot(freqs_b, abs_tfer_func, color='r', label=legend_b)
    if labels:
        for label in labels:
            x_coord = label[0]
            y_coord = label[1]
            F_n = label[2]
            plt.text(x_coord, y_coord, r' $F$' + str(F_n))
            ax.plot([x_coord, x_coord], [0, y_coord], "r--h")
            print("F_" + str(F_n) + ": (" + str(x_coord) + ", " + str(y_coord) + ")")
    ax.legend()
    plt.grid(True)
    plt.savefig(result_folder + filename)
    plt.show()


########################################### SIGNALS ###########################################


def open_int_wav_file(wav_path):
    print("Opening: " + wav_path)
    fm, s = read(wav_path)
    return (fm, s)


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


def get_lpc(signal, order):
    if signal.ndim > 1:
        raise ValueError("Array of rank > 1 not supported yet")
    if order > signal.size:
        raise ValueError("Input signal must have a lenght >= lpc order")
    if order > 0:
        p = order + 1
        r = np.zeros(p, signal.dtype)
        nx = np.min([p, signal.size])
        x = np.correlate(signal, signal, 'full')
        r[:nx] = x[signal.size-1:signal.size + order]
        phi = np.dot(sp.linalg.inv(sp.linalg.toeplitz(r[:-1])), -r[1:])
        return np.concatenate(([1.], phi))
    else:
        return np.ones(1, dtype = signal.dtype)


def outmidear(n, fs):
    f = np.array([0, .02, .05, .1,.2, .5, .6, .7, 1, 2, 3, 4, 5, 8,  9, 10, 12, 13, 14, 15]) * 1000
    g = np.array([float('-inf'), -39, -19, -12.5, -8, -2, -1, 0, 0, 4, 8, 8, 5, -9, -11, -11, -9, -13, -19, -15])
    m = 10 ** (g/20)
    if fs/2 > f[-1]:
        f = np.concatenate((f, [fs/2]))
        m = np.concatenate((m, [0]))
    else:
        mend = np.interp(fs/2, f, m)
        i = f < fs/2
        f = np.concatenate((f[i], [fs/2]))
        m = np.concatenate((m[i], [mend]))
    oddsize = 2 * math.ceil(n/2) + 1
    b = signal.firwin2(oddsize, f/(fs/2), m)
    return b


def get_approx_tranfer_vocal(b, a, fm, L=512):
    w, V = signal.freqz(b, a, L)
    f = w/math.pi * fm/2
    V = np.absolute(V)
    RV = np.sqrt(V);
    #NRV = RV / norm(RV)
    NRV = RV / np.max(RV)
    return f, V, RV, NRV


def plot_vowel_vocal_system_response(vowel_wav_path, norm_sqrt_fft, freqs, title, filename, K, labels=None, sample_rate=sample_rate_16k):
    _, vowel_signal = open_int_wav_file(vowel_wav_path)
    vowel_signal = vowel_signal / np.iinfo(vowel_signal.dtype).max
    vowel_signal = get_sub_signal(vowel_signal, limA_ms, limB_ms)
    a_vector = get_lpc(vowel_signal, K)
    b_vector = 1
    (freqs_V, V, RV, NRV) = get_approx_tranfer_vocal(b_vector, a_vector, sample_rate)
    legend_a = "√|S(f)|"
    legend_b = "√|V(f)|"
    xlims = (-0.5, 3000) 
    plot_fft_transfer_function(filename, title, freqs, norm_sqrt_fft, legend_a, freqs_V, NRV, legend_b, xlims, labels)


def plot_vowel_vocal_system_response_outmidear(vowel_wav_path, norm_sqrt_fft, freqs, title, filename, K, labels=None, sample_rate=sample_rate_16k):
    _, vowel_signal = open_int_wav_file(vowel_wav_path)
    vowel_signal = vowel_signal / np.iinfo(vowel_signal.dtype).max
    vowel_signal = get_sub_signal(vowel_signal, limA_ms, limB_ms)
    a_vector = get_lpc(vowel_signal, K)
    n = round(0.001 * sample_rate)
    b_vector = outmidear(n, sample_rate)
    y = lfilter(b_vector, 1, vowel_signal)
    (freqs_V, V, RV, NRV) = get_approx_tranfer_vocal(b_vector, a_vector, sample_rate)
    legend_a = "√|S(f)|"
    legend_b = "√|V(f)|"
    xlims = (-0.5, 3000) 
    plot_fft_transfer_function(filename, title, freqs, norm_sqrt_fft, legend_a, freqs_V, NRV, legend_b, xlims, labels)


def get_signal_info(vowel_wav_path, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title, sample_rate=sample_rate_16k):
    _, vowel_signal = open_int_wav_file(vowel_wav_path)
    vowel_signal = get_sub_signal(vowel_signal, limA_ms, limB_ms)
    plot_time_signal(vowel_signal, t_title, t_filename)
    xlims = (-0.5, 3000)
    time_ms = (limB_ms - limA_ms)
    time = np.linspace(0, time_ms/1000.0, (time_ms/1000.0) * sample_rate)
    freqs, spectrum = plot_fft(vowel_signal, time, f_title, f_filename, xlims)
    sqrt_fft = np.sqrt(spectrum)
    norm_sqrt_fft = sqrt_fft / np.max(sqrt_fft)
    plot_sqrt_fft(norm_sqrt_fft, freqs, sqf_title, sqf_filename, xlims)
    return freqs, norm_sqrt_fft


def plot_glotis_estimation(vowel_wav_path, K, title, filename, fm=sample_rate_16k):
    _, vowel_signal = open_int_wav_file(vowel_wav_path)
    vowel_signal = vowel_signal / np.iinfo(vowel_signal.dtype).max
    vowel_signal = get_sub_signal(vowel_signal, limA_ms, limB_ms)
    a = get_lpc(vowel_signal, K)
    u = lfilter(a, 1, vowel_signal)
    t = np.arange(len(u)) / fm
    plt.plot(1000 * t, u)
    plt.ylabel('$u(t)$')
    plt.xlabel('Time (ms)')
    plt.grid(True)
    plt.title(title)
    plt.savefig(result_folder + filename)
    plt.show()
    return t, u


def plot_autocorrelation(u, t, title, filename):
    r = np.correlate(u, u, mode='full')
    r = np.pad(r[np.argmax(r):-1], (0,1), 'constant')
    r = r / np.max(r)
    plt.plot(1000 * t, r);
    plt.ylabel('$r(t)$'); 
    plt.xlabel('Time (ms)');
    plt.grid(True);
    plt.title(title)
    plt.savefig(result_folder + filename)
    plt.show()


########################################### ITEM 1 #######################################

def solve_item1():
    print("\n\n#########################################################################")
    print("##                               ITEM 1                                ##")
    print("#########################################################################\n")

    global norm_sqrt_fft_a
    global norm_sqrt_fft_e
    global norm_sqrt_fft_i
    global norm_sqrt_fft_o
    global norm_sqrt_fft_u
    global freqs_a
    global freqs_e
    global freqs_i
    global freqs_o
    global freqs_u

    # ITEM 1.A

    # U vowel
    t_title = "Signal of vowel 'u'"
    t_filename = "u_time_1a.pdf"
    f_title = "Spectrum of signal vowel 'u'"
    f_filename = "u_spectrum_1a.pdf"
    sqf_title = "Square root of the spectrum of signal vowel 'u'"
    sqf_filename = "u_sqrt_spectrum_1a.pdf"
    freqs_u, norm_sqrt_fft_u = get_signal_info(u_male_periodic, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)

    # A vowel
    t_title = "Signal of vowel 'a'"
    t_filename = "a_time_1a.pdf"
    f_title = "Spectrum of signal vowel 'a'"
    f_filename = "a_spectrum_1a.pdf"
    sqf_title = "Square root of the spectrum of signal vowel 'a'"
    sqf_filename = "a_sqrt_spectrum_1a.pdf"
    freqs_a, norm_sqrt_fft_a = get_signal_info(a_male_periodic, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)

    # E vowel
    t_title = "Signal of vowel 'e'"
    t_filename = "e_time_1a.pdf"
    f_title = "Spectrum of signal vowel 'e'"
    f_filename = "e_spectrum_1a.pdf"
    sqf_title = "Square root of the spectrum of signal vowel 'e'"
    sqf_filename = "e_sqrt_spectrum_1a.pdf"
    freqs_e, norm_sqrt_fft_e = get_signal_info(e_male_periodic, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)

    # I vowel
    t_title = "Signal of vowel 'i'"
    t_filename = "i_time_1a.pdf"
    f_title = "Spectrum of signal vowel 'i'"
    f_filename = "i_spectrum_1a.pdf"
    sqf_title = "Square root of the spectrum of signal vowel 'i'"
    sqf_filename = "i_sqrt_spectrum_1a.pdf"
    freqs_i, norm_sqrt_fft_i = get_signal_info(i_male_periodic, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)

    # O vowel
    t_title = "Signal of vowel 'o'"
    t_filename = "o_time_1a.pdf"
    f_title = "Spectrum of signal vowel 'o'"
    f_filename = "o_spectrum_1a.pdf"
    sqf_title = "Square root of the spectrum of signal vowel 'o'"
    sqf_filename = "o_sqrt_spectrum_1a.pdf"
    freqs_o, norm_sqrt_fft_o = get_signal_info(o_male_periodic, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)

    # ITEM 1.B

    # U vowel
    title = "Spectrum of vowel U and transfer function estimated by LPC"
    filename = "u_transfer_function_1b.pdf"
    plot_vowel_vocal_system_response(u_male_periodic, norm_sqrt_fft_u, freqs_u, title, filename, K_u)

    # A vowel
    title = "Spectrum of vowel A and transfer function estimated by LPC"
    filename = "a_transfer_function_1b.pdf"
    plot_vowel_vocal_system_response(a_male_periodic, norm_sqrt_fft_a, freqs_a, title, filename, K_a)

    # E vowel
    title = "Spectrum of vowel E and transfer function estimated by LPC"
    filename = "e_transfer_function_1b.pdf"
    plot_vowel_vocal_system_response(e_male_periodic, norm_sqrt_fft_e, freqs_e, title, filename, K_e)

    # I vowel
    title = "Spectrum of vowel I and transfer function estimated by LPC"
    filename = "i_transfer_function_1b.pdf"
    plot_vowel_vocal_system_response(i_male_periodic, norm_sqrt_fft_i, freqs_i, title, filename, K_i)

    # O vowel
    title = "Spectrum of the vowel O and transfer function estimated by LPC"
    filename = "o_transfer_function_1b.pdf"
    plot_vowel_vocal_system_response(o_male_periodic, norm_sqrt_fft_o, freqs_o, title, filename, K_o)


########################################### ITEM 2 #######################################

def solve_item2():
    print("\n\n#########################################################################")
    print("##                               ITEM 2                                ##")
    print("#########################################################################\n")

    # ITEM 2.A

    # U vowel
    title = "Spectrum of vowel U and vocal tract transfer function improved"
    filename = "u_transfer_function_2a.pdf"
    labels = [(324, 1, 1), (782, 0.28, 2)]
    plot_vowel_vocal_system_response_outmidear(u_male_periodic, norm_sqrt_fft_u, freqs_u, title, filename, K_u, labels)

    # A vowel
    title = "Spectrum of vowel A and vocal tract transfer function improved"
    filename = "a_transfer_function_2a.pdf"
    labels = [(125, 1, 1), (657, 0.88, 2)]
    plot_vowel_vocal_system_response_outmidear(a_male_periodic, norm_sqrt_fft_a, freqs_a, title, filename, K_a, labels)

    # E vowel
    title = "Spectrum of vowel E and vocal tract transfer function improved"
    filename = "e_transfer_function_2a.pdf"
    labels = [(375, 1, 1), (2560, 0.38, 2)]
    plot_vowel_vocal_system_response_outmidear(e_male_periodic, norm_sqrt_fft_e, freqs_e, title, filename, K_e, labels)

    # I vowel
    title = "Spectrum of vowel I and vocal tract transfer function improved"
    filename = "i_transfer_function_2a.pdf"
    labels = [(250, 1, 1), (2256, 0.26, 2)]
    plot_vowel_vocal_system_response_outmidear(i_male_periodic, norm_sqrt_fft_i, freqs_i, title, filename, K_i, labels)

    # O vowel
    title = "Spectrum of vowel O and vocal tract transfer function improved"
    filename = "o_transfer_function_2a.pdf"
    labels = [(421, 1, 1), (856, 0.4, 2)]
    plot_vowel_vocal_system_response_outmidear(o_male_periodic, norm_sqrt_fft_o, freqs_o, title, filename, K_o, labels)


########################################### ITEM 3 #######################################

def solve_item3():
    print("\n\n#########################################################################")
    print("##                               ITEM 3                                ##")
    print("#########################################################################\n")

    # ITEM 3.A

    # U vowel
    title = "Excitation signal estimation for vowel U"
    filename = "u_glotis_estimation_3a.pdf"
    u_time, u_usignal = plot_glotis_estimation(u_male_periodic, K_u, title, filename)
    title = "Autocorrelation of the estimated excitation signal for vowel U"
    filename = "u_autocorrelation_3c.pdf"
    plot_autocorrelation(u_usignal, u_time, title, filename)

    # A vowel
    title = "Excitation signal estimation for vowel A"
    filename = "a_glotis_estimation_3a.pdf"
    a_time, a_usignal = plot_glotis_estimation(a_male_periodic, K_a, title, filename)
    title = "Autocorrelation of the estimated excitation signal for vowel A"
    filename = "a_autocorrelation_3c.pdf"
    plot_autocorrelation(a_usignal, a_time, title, filename)

    # E vowel
    title = "Excitation signal estimation for vowel E"
    filename = "e_glotis_estimation_3a.pdf"
    e_time, e_usignal = plot_glotis_estimation(e_male_periodic, K_e, title, filename)
    title = "Autocorrelation of the estimated excitation signal for vowel E"
    filename = "e_autocorrelation_3c.pdf"
    plot_autocorrelation(e_usignal, e_time, title, filename)

    # I vowel
    title = "Excitation signal estimation for vowel I"
    filename = "i_glotis_estimation_3a.pdf"
    i_time, i_usignal = plot_glotis_estimation(i_male_periodic, K_i, title, filename)
    title = "Autocorrelation of the estimated excitation signal for vowel I"
    filename = "i_autocorrelation_3c.pdf"
    plot_autocorrelation(i_usignal, i_time, title, filename)

    # O vowel
    title = "Excitation signal estimation for vowel O"
    filename = "o_glotis_estimation_3a.pdf"
    o_time, o_usignal = plot_glotis_estimation(o_male_periodic, K_o, title, filename)
    title = "Autocorrelation of the estimated excitation signal for vowel O"
    filename = "o_autocorrelation_3c.pdf"
    plot_autocorrelation(o_usignal, o_time, title, filename)


########################################### ITEM 4 #######################################

def solve_item4():
    print("\n\n#########################################################################")
    print("##                               ITEM 4                                ##")
    print("#########################################################################\n")

    # ITEM 4.A

    # U vowel

    t_title = "Whispered signal of vowel 'u'"
    t_filename = "u_whispered_time_4a.pdf"
    f_title = "Spectrum of the whispered signal vowel 'u'"
    f_filename = "u_whispered_spectrum_4a.pdf"
    sqf_title = "Square root of the spectrum of the whispered signal vowel 'u'"
    sqf_filename = "u_whispered_sqrt_spectrum_4a.pdf"
    freqs_u, norm_sqrt_fft_u = get_signal_info(u_male_whisper, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)
    title = "Spectrum of whispered vowel U and transfer function estimated by LPC"
    filename = "u_whispered_transfer_function_4a1.pdf"
    plot_vowel_vocal_system_response(u_male_whisper, norm_sqrt_fft_u, freqs_u, title, filename, K_u)
    title = "Spectrum of the whispered vowel U and transfer function improved"
    filename = "u_whispered_transfer_function_4a2.pdf"
    labels = [(36, 1, 1), (724, 0.23, 2)]
    plot_vowel_vocal_system_response_outmidear(u_male_whisper, norm_sqrt_fft_u, freqs_u, title, filename, K_u, labels)
    title = "Excitation signal estimation for whispered vowel U"
    filename = "u_whispered_glotis_estimation_4a.pdf"
    u_time, u_usignal = plot_glotis_estimation(u_male_whisper, K_u, title, filename)
    title = "Autocorrelation of the estimated excitation signal for whispered vowel U"
    filename = "u_whispered_autocorrelation_4a.pdf"
    plot_autocorrelation(u_usignal, u_time, title, filename)


    # A vowel

    t_title = "Whispered signal of vowel 'a'"
    t_filename = "a_whispered_time_4a.pdf"
    f_title = "Spectrum of the whispered signal vowel 'a'"
    f_filename = "a_whispered_spectrum_4a.pdf"
    sqf_title = "Square root of the spectrum of the whispered signal vowel 'a'"
    sqf_filename = "a_whispered_sqrt_spectrum_4a.pdf"
    freqs_a, norm_sqrt_fft_a = get_signal_info(a_male_whisper, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)
    title = "Spectrum of whispered vowel A and transfer function estimated by LPC"
    filename = "a_whispered_transfer_function_4a1.pdf"
    plot_vowel_vocal_system_response(a_male_whisper, norm_sqrt_fft_a, freqs_a, title, filename, K_a)
    title = "Spectrum of the whispered vowel A and transfer function improved"
    filename = "a_whispered_transfer_function_4a2.pdf"
    labels = [(756, 0.35, 1), (1288, 0.28, 2)]
    plot_vowel_vocal_system_response_outmidear(a_male_whisper, norm_sqrt_fft_a, freqs_a, title, filename, K_a, labels)
    title = "Excitation signal estimation for whispered vowel A"
    filename = "a_whispered_glotis_estimation_4a.pdf"
    a_time, a_usignal = plot_glotis_estimation(a_male_whisper, K_a, title, filename)
    title = "Autocorrelation of the estimated excitation signal for whispered vowel A"
    filename = "a_whispered_autocorrelation_4a.pdf"
    plot_autocorrelation(a_usignal, a_time, title, filename)


    # E vowel

    t_title = "Whispered signal of vowel 'e'"
    t_filename = "e_whispered_time_4a.pdf"
    f_title = "Spectrum of the whispered signal vowel 'e'"
    f_filename = "e_whispered_spectrum_4a.pdf"
    sqf_title = "Square root of the spectrum of the whispered signal vowel 'e'"
    sqf_filename = "e_whispered_sqrt_spectrum_4a.pdf"
    freqs_e, norm_sqrt_fft_e = get_signal_info(e_male_whisper, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)
    title = "Spectrum of whispered vowel E and transfer function estimated by LPC"
    filename = "e_whispered_transfer_function_4a1.pdf"
    plot_vowel_vocal_system_response(e_male_whisper, norm_sqrt_fft_e, freqs_e, title, filename, K_e)
    title = "Spectrum of the whispered vowel E and transfer function improved"
    filename = "e_whispered_transfer_function_4a2.pdf"
    labels = [(756, 0.21, 1), (1808, 0.15, 2)]
    plot_vowel_vocal_system_response_outmidear(e_male_whisper, norm_sqrt_fft_e, freqs_e, title, filename, K_e, labels)
    title = "Excitation signal estimation for whispered vowel E"
    filename = "e_whispered_glotis_estimation_4a.pdf"
    e_time, e_usignal = plot_glotis_estimation(e_male_whisper, K_e, title, filename)
    title = "Autocorrelation of the estimated excitation signal for whispered vowel E"
    filename = "e_whispered_autocorrelation_4a.pdf"
    plot_autocorrelation(e_usignal, e_time, title, filename)


    # I vowel

    t_title = "Whispered signal of vowel 'i'"
    t_filename = "i_whispered_time_4a.pdf"
    f_title = "Spectrum of the whispered signal vowel 'i'"
    f_filename = "i_whispered_spectrum_4a.pdf"
    sqf_title = "Square root of the spectrum of the whispered signal vowel 'i'"
    sqf_filename = "i_whispered_sqrt_spectrum_4a.pdf"
    freqs_i, norm_sqrt_fft_i = get_signal_info(i_male_whisper, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)
    title = "Spectrum of whispered vowel I and transfer function estimated by LPC"
    filename = "i_whispered_transfer_function_4a1.pdf"
    plot_vowel_vocal_system_response(i_male_whisper, norm_sqrt_fft_i, freqs_i, title, filename, K_i)
    title = "Spectrum of the whispered vowel I and transfer function improved"
    filename = "i_whispered_transfer_function_4a2.pdf"
    labels = [(792, 0.72, 1), (2111, 1, 2)]
    plot_vowel_vocal_system_response_outmidear(i_male_whisper, norm_sqrt_fft_i, freqs_i, title, filename, K_i, labels)
    title = "Excitation signal estimation for whispered vowel I"
    filename = "i_whispered_glotis_estimation_4a.pdf"
    i_time, i_usignal = plot_glotis_estimation(i_male_whisper, K_i, title, filename)
    title = "Autocorrelation of the estimated excitation signal for whispered vowel I"
    filename = "i_whispered_autocorrelation_4a.pdf"
    plot_autocorrelation(i_usignal, i_time, title, filename)


    # O vowel

    t_title = "Whispered signal of vowel 'o'"
    t_filename = "o_whispered_time_4a.pdf"
    f_title = "Spectrum of the whispered signal vowel 'o'"
    f_filename = "o_whispered_spectrum_4a.pdf"
    sqf_title = "Square root of the spectrum of the whispered signal vowel 'o'"
    sqf_filename = "o_whispered_sqrt_spectrum_4a.pdf"
    freqs_o, norm_sqrt_fft_o = get_signal_info(o_male_whisper, t_filename, t_title, f_filename, f_title, sqf_filename, sqf_title)
    title = "Spectrum of whispered vowel O and transfer function estimated by LPC"
    filename = "o_whispered_transfer_function_4a1.pdf"
    plot_vowel_vocal_system_response(o_male_whisper, norm_sqrt_fft_o, freqs_o, title, filename, K_o)
    title = "Spectrum of the whispered vowel O and transfer function improved"
    filename = "o_whispered_transfer_function_4a2.pdf"
    labels = [(375, 0.5, 1), (980, 0.36, 0)]
    plot_vowel_vocal_system_response_outmidear(o_male_whisper, norm_sqrt_fft_o, freqs_o, title, filename, K_o, labels)
    title = "Excitation signal estimation for whispered vowel O"
    filename = "o_whispered_glotis_estimation_4a.pdf"
    o_time, o_usignal = plot_glotis_estimation(o_male_whisper, K_o, title, filename)
    title = "Autocorrelation of the estimated excitation signal for whispered vowel O"
    filename = "o_whispered_autocorrelation_4a.pdf"
    plot_autocorrelation(o_usignal, o_time, title, filename)


############################################ MAIN ########################################


def main():
    create_folder(result_folder)
    solve_item1()
    solve_item2()
    solve_item3()
    solve_item4()

if __name__== "__main__":
  warnings.simplefilter("ignore")
  main()
