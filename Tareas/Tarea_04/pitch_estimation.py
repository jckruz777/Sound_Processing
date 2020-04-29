#!/usr/bin/env python3

# Libraries
import argparse
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import re
import sys
import warnings
import wave

from pitch_algorithms.a1 import *
from pitch_algorithms.a2 import *
from pitch_algorithms.a3 import *
from pitch_algorithms.a4 import *
from pitch_algorithms.a5 import *
from pitch_algorithms.a6 import *
from pitch_algorithms.a7 import *
from math import pi, ceil
from matplotlib.mlab import specgram
from scikits.samplerate import resample
from scipy import signal
from scipy.fft import fft, fftshift
from scipy.io.wavfile import write

# Global Variables
result_folder = "./results/"
periodic_path = "./periodic_audios/"
speech_path = "./speech/pirata.wav"
m_speech_folder = "./benchmarking/M_speech/"
f_speech_folder = "./benchmarking/F_speech/"

############################################ UTILS ############################################

def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return ceil(n * multiplier) / multiplier

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        print("Creating directory: " + folder_path)

def from_file_columns_to_arrays(path):
    col1 = []
    col2 = []
    with open(path) as f_obj:
        for line in f_obj:
            row = line.split()
            col1.append(float(row[0]))
            col2.append(float(row[1]))
    return col1, col2

################################### PITCH ESTIMATION ALGORITHMS ###############################

def hps(signal, samplig_rate, dt=None):
    (p, t, s, pc, S) = a1(signal, samplig_rate, dt=dt)
    return (p,t,s,pc,S)

def shs(signal, samplig_rate, dt=None):
    (p, t, s, pc, S) = a2(signal, samplig_rate, dt=dt)
    return (p,t,s,pc,S)

def shs_weighted(signal, samplig_rate, dt=None):
    (p, t, s, pc, S) = a3(signal, samplig_rate, dt=dt)
    return (p,t,s,pc,S)

def shr(signal, samplig_rate, tw=None, dt=None):
    (p, t, s, pc, S) = a4(signal, samplig_rate, tw=tw, dt=dt)
    return (p,t,s,pc,S)

def autocorrelation_wk(signal, samplig_rate, n=None, plim=None, tw=None, dt=None):
    (p, t, s, pc, S) = a5(signal, samplig_rate, n=n, plim=plim, tw=tw, dt=dt)
    return (p,t,s,pc,S)

def autocorrelation_wk_sqrt(signal, samplig_rate, tw=None, dt=None):
    (p, t, s, pc, S) = a6(signal, samplig_rate, tw=tw, dt=dt)
    return (p,t,s,pc,S)

def autocorrelation_wk_cochlea(signal, samplig_rate, n=None, plim=None, tw=None, dt=None):
    (p, t, s, pc, S) = a7(signal, samplig_rate, n=n, plim=plim, tw=tw, dt=dt)
    return (p,t,s,pc,S)

######################################### PLOTS & SIGNALS #####################################

def open_wav_file(wav_path, channel_id=0, t_limA_ms=None, t_limB_ms=None, sample_rate = 8000):
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

def get_sawtooth_wav(filename, frequency, duration, sample_rate, do_write=False):
    Tm = 1/sample_rate
    t = arange(0, duration, Tm)
    x = sawtooth(2 * pi * frequency * t)
    if do_write:
        write (filename, sample_rate, x/2)
    return x/2

def get_pure_tone_wav(filename, frequency, amplitude, duration, sample_rate, do_write=False):
    Tm = 1/sample_rate
    t = arange(0, duration, Tm)
    x = amplitude * np.cos(2 * pi * frequency * t)
    if do_write:
        write (filename, sample_rate, x/2)
    return x/2

def plot_matrix(S, t, pc, title, filename):
    # rojo = alto, azul = bajo
    ax = plt.gca()
    yticks = np.around(np.arange(0, len(pc), len(pc) / 10)).astype(int)
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.around(pc[yticks]).astype(int))
    xticks = np.around(np.arange(0, len(t), len(t) / 10)).astype(int)
    t_xticks = np.zeros(len(t[xticks]))
    for tk in range(len(t[xticks])):
        t_xticks[tk] = round_up(t[xticks][tk], 2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(t_xticks)
    plt.ylabel('Pitch (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.imshow(S, aspect='auto', cmap='jet')
    plt.savefig(filename)
    plt.show()

def plot_central_column(S, pc, title, filename):
    plt.plot(pc, S[:, round(S.shape[1]/2)])
    plt.ylabel('Score');
    plt.xlabel('Pitch (Hz)');
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_wave_time(signal, title, xlims, filename, do_save=False, sample_rate = 80000):
    """
    Plot a time domain signal from an audio file
    """
    plt.figure(1)
    plt.title(title)
    N_samples = len(signal)
    dt = 1.0 / sample_rate
    resize_factor = dt * 1000.0
    sig_time = np.arange(N_samples) * resize_factor
    if xlims:
        xlim_a = xlims[0] * resize_factor
        xlim_b = xlims[1] * resize_factor
        plt.xlim(xlims[0], xlims[1])
    plt.plot(sig_time, signal)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    if do_save:
        plt.savefig(filename)
    plt.show()

def plot_canditates_in_time(S, pc, title, filename, xlims=(0,800)):
    N_rows = S.shape[0]
    N_cols = S.shape[1]
    middle_index = int(N_cols / 2)
    column = np.zeros(N_rows)
    for i in range(N_rows):
        column[i] = S[i][middle_index]
    x = np.linspace(0, N_rows, N_rows)
    plt.stem(pc, abs(column), use_line_collection=True)
    plt.ylabel('Scores');
    plt.xlabel('Candidate (Hz)');
    plt.title(title)
    plt.xlim(xlims[0], xlims[1])
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_scores_in_time(t, s, title, filename):
    plt.plot(t, s)
    plt.ylabel('Scores');
    plt.xlabel('Time (s)');
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_pitch_vs_time(t, p, title, filename, xlims=None, ylims=None):
    plt.plot(t, p)
    plt.ylabel('Pitch (Hz)');
    plt.xlabel('Time (s)');
    plt.title(title)
    plt.grid(True)
    if xlims:
        plt.xlim(xlims[0], xlims[1])
    if ylims:
        plt.ylim(ylims[0], ylims[1])
    plt.savefig(filename)
    plt.show()

def get_winner_avg(s, t, title, filename):
    plt.plot(1000*t, s)
    plt.ylabel('Score of the Winner')
    plt.xlabel('Time (ms)')
    plt.title(title)
    plt.grid(True)
    plt.savefig(filename)
    plt.show()
    m = np.mean(s)
    return m

########################################### ITEM 1 #######################################

def solve_item_01():
    print("\n###########################################################")
    print("##                      ITEM 1                           ##")
    print("###########################################################\n")

    # ITEM 1.A
    frequency = 200
    sample_rate = 10000
    duration = 0.5
    filename = result_folder + 'diente-sierra_1a.wav'
    signal = get_sawtooth_wav(filename, frequency, duration, sample_rate, True)
    print("ITEM 1.A: " + filename + " generated.\n")

    # ITEM 1.B
    frequency = 200
    sample_rate = 10000
    duration = 0.5
    amplitude = 1
    filename = result_folder + 'pure_tone_1b_{}.wav'.format(frequency)
    signal = get_pure_tone_wav(filename, frequency, amplitude, duration, sample_rate, True)
    print("ITEM 1.B: " + filename + " generated.\n")

    # ITEM 1.C
    frequency = 200
    sample_rate = 10000
    duration = 0.5
    amplitude = 1
    folder = result_folder + "audio_1c/"
    create_folder(folder)
    for n in range(1, 5):
        frequency = 200 + n
        filename = folder + 'pure_tone_1c_{}.wav'.format(frequency)
        signal = get_pure_tone_wav(filename, frequency, amplitude, duration, sample_rate, True)
        frequency = 200 - n
        filename = folder + 'pure_tone_1c_{}.wav'.format(frequency)
        signal = get_pure_tone_wav(filename, frequency, amplitude, duration, sample_rate, True)
    print("ITEM 1.C: Estimated pitch = 202 Hz.\n")

    # ITEM 1.D
    frequency = 200
    sample_rate = 8000
    duration = 0.5
    signal = get_sawtooth_wav("", frequency, duration, sample_rate, False)
    (p, t, s, pc, S) = hps(signal, sample_rate)
    print("ITEM 1.D: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 1.E
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_1e.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 1.E: Score matrix plotted and saved.")

    # ITEM 1.F
    title = "Score per candidate"
    filename = result_folder + "score_per_candidate_1f.pdf"
    plot_central_column(S, pc, title, filename)
    print("ITEM 1.F: Score per candidate plotted and saved.")

########################################### ITEM 2 #######################################

def solve_item_02():
    print("\n###########################################################")
    print("##                      ITEM 2                           ##")
    print("###########################################################\n")

    # ITEM 2.A
    frequency = 200
    sample_rate = 8000
    duration = 0.5
    amplitude = 1.0
    signal = get_pure_tone_wav("", frequency, amplitude, duration, sample_rate, False)
    (p, t, s, pc, S) = hps(signal, sample_rate)
    print("ITEM 2.A: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 2.B
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_2b.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 2.B: Score matrix plotted and saved.\n")

    # ITEM 2.C
    print("ITEM 2.C:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_2c.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 2.D
    print("ITEM 2.D: HPS was not able to provide a pitch estimation.\n")

    # ITEM 2.E
    print("ITEM 2.E: SHS algorithm implemented in the file: ./pitch_algorithms/a2.py\n")

    # ITEM 2.F
    frequency = 200
    sample_rate = 8000
    duration = 0.5
    amplitude = 1.0
    signal = get_pure_tone_wav("", frequency, amplitude, duration, sample_rate, False)
    (p, t, s, pc, S) = shs(signal, sample_rate)
    print("ITEM 2.F: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 2.G
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_2g.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 2.G: Score matrix plotted and saved.")

    # ITEM 2.H
    print("ITEM 2.H:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_2h.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 2.I
    print("ITEM 2.I: SHS was able to provide a pitch estimation. However, the election could be risky since some candidates obtained very similar scores.\n")

########################################### ITEM 3 #######################################

def solve_item_03():
    print("\n###########################################################")
    print("##                      ITEM 3                           ##")
    print("###########################################################\n")

    # ITEM 3.A
    print("ITEM 3.A: Weighted SHS algorithm implemented in the file: ./pitch_algorithms/a3.py\n")

    # ITEM 3.B
    frequency = 200
    sample_rate = 8000
    duration = 0.5
    amplitude = 1.0
    signal = get_pure_tone_wav("", frequency, amplitude, duration, sample_rate, False)
    (p, t, s, pc, S) = shs_weighted(signal, sample_rate)
    print("ITEM 3.B: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 3.C
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_3c.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 3.C: Score matrix plotted and saved.\n")

    # ITEM 3.D
    print("ITEM 3.D:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_3d.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 3.E
    title = "Scores of the winner along the time"
    filename = result_folder + "scores_of_winner_3e.pdf"
    winner_avg = get_winner_avg(s, t, title, filename)
    print("ITEM 3.E:")
    print("Winner AVG = " + str(winner_avg))
    print("20% of the Winner AVG = " + str(winner_avg * 0.20))
    print(title + " plotted and saved.\n")

    # ITEM 3.F
    print("ITEM 3.F: Weighted SHS was able to determine a winner.")
    print("It could be considered as a risky estimation since the score at 100 Hz (0.20)")
    print("is lower than the score at the winner ~200 Hz (0.24) in less than a 20%.")
    print("However, the winner is a bit higher than 200 Hz, being at a distance") 
    print("longer than 1 octave from 100 Hz. So, the estimation is not risky in this case.\n")

########################################### ITEM 4 #######################################

def solve_item_04():
    print("\n###########################################################")
    print("##                      ITEM 4                           ##")
    print("###########################################################\n")

    # ITEM 4.A
    sample_rate = 8000
    duration = 0.5
    noise_signal = np.random.random_sample((int(duration * sample_rate),))
    (p, t, s, pc, S) = shs_weighted(noise_signal, sample_rate)
    print("ITEM 4.A: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 4.B
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_4b.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 4.B: Score matrix plotted and saved.\n")

    # ITEM 4.C
    print("ITEM 4.C:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_4c.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 4.D
    title = "Scores of the winner along the time"
    filename = result_folder + "scores_of_winner_4d.pdf"
    winner_avg = get_winner_avg(s, t, title, filename)
    print("ITEM 4.D:")
    print("Winner AVG = " + str(winner_avg))
    print(title + " plotted and saved.")
    print("This value is not insignificant because it is not less than the 20% of the maximum score obtained with the pure tone of 200 Hz.\n")

    # ITEM 4.E
    print("ITEM 4.E: SHR algorithm implemented in the file: ./pitch_algorithms/a4.py\n")

    # ITEM 4.F
    frequency = 200
    sample_rate = 8000
    duration = 0.5
    amplitude = 1.0
    signal = get_pure_tone_wav("", frequency, amplitude, duration, sample_rate, False)
    noise_signal = np.random.random_sample((int(duration * sample_rate),))
    result_signal = np.concatenate((signal, noise_signal), axis=None)  
    (p, t, s, pc, S) = shr(result_signal, sample_rate)
    print("ITEM 4.F: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 4.G
    title = "Score matrix plotted as color scale"
    filename = result_folder + "score_matrix_4g.pdf"
    plot_matrix(S, t, pc, title, filename)
    print("ITEM 4.G: Score matrix plotted and saved.\n")

    # ITEM 4.H
    title = "Scores of the winner along the time"
    filename = result_folder + "scores_of_winner_4h.pdf"
    winner_avg = get_winner_avg(s, t, title, filename)
    print("ITEM 4.H:")
    print("Winner AVG = " + str(winner_avg))
    print(title + " plotted and saved.\n")

    # ITEM 4.I
    print("This value is not insignificant because it is not less than the 20% of the maximum score obtained with the pure tone of 200 Hz.\n")


########################################### ITEM 5 #######################################

def solve_item_05():
    print("\n###########################################################")
    print("##                      ITEM 5                           ##")
    print("###########################################################\n")

    # ITEM 5.A
    freq1 = 650
    freq2 = 950
    freq3 = 1250
    duration = 5.0
    amplitude = 1.0
    sample_rate = 8000
    signal_1 = get_pure_tone_wav("", freq1, amplitude, duration, sample_rate, False)
    signal_2 = get_pure_tone_wav("", freq2, amplitude, duration, sample_rate, False)
    signal_3 = get_pure_tone_wav("", freq3, amplitude, duration, sample_rate, False)
    signal = signal_1 + signal_2 + signal_3
    title = "25 ms of a signal composed by frequencies: 650, 950 and 1250 Hz"
    filename = result_folder + "multitone_signal_5a.pdf"
    plot_wave_time(signal, title, (0,25), filename, True, sample_rate)
    print("ITEM 5.A: Signal plotted and saved")

    # ITEM 5.B
    window_size_segs = 1
    (p, t, s, pc, S) = shr(signal, sample_rate, tw=window_size_segs)
    print("ITEM 5.B: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 5.C
    print("ITEM 5.C:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_5c.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 5.D
    print("ITEM 5.D:\nThe candidate with the highest score does not correspond to the pitch perceived by the majority (330 Hz)") 
    print("nor to the pitch perceived by the next most significant group (650 Hz).\n")

    # ITEM 5.E
    print("ITEM 5.E: Autocorrelation + Wiener-Khinchin algorithm implemented in the file: ./pitch_algorithms/a5.py\n")

    # ITEM 5.F
    freq1 = 650
    freq2 = 950
    freq3 = 1250
    duration = 5.0
    amplitude = 1.0
    sample_rate = 8000
    window_size_segs = 1
    signal_1 = get_pure_tone_wav("", freq1, amplitude, duration, sample_rate, False)
    signal_2 = get_pure_tone_wav("", freq2, amplitude, duration, sample_rate, False)
    signal_3 = get_pure_tone_wav("", freq3, amplitude, duration, sample_rate, False)
    signal = signal_1 + signal_2 + signal_3
    (p, t, s, pc, S) = autocorrelation_wk(signal, sample_rate, tw=window_size_segs)
    print("ITEM 5.F: Winner of each window: ")
    print(str(p) + "\n")

    # ITEM 5.G
    print("ITEM 5.G:")
    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_5g.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

    # ITEM 5.H
    print("ITEM 5.H: The candidate with the highest score does not correspond to the height perceived by the majority (330 Hz)")
    print("nor to the height perceived by the next most significant group (650 Hz).\n")

########################################### ITEM 6 #######################################

def solve_item_06():
    print("\n###########################################################")
    print("##                      ITEM 6                           ##")
    print("###########################################################\n")

    # ITEM 6.A
    print("ITEM 6.A: Autocorrelation + Wiener-Khinchin + spectrum sqrt algorithm implemented in the file: ./pitch_algorithms/a6.py\n")

    freq1 = 650
    freq2 = 950
    freq3 = 1250
    duration = 5.0
    amplitude = 1.0
    sample_rate = 8000
    window_size_segs = 1
    signal_1 = get_pure_tone_wav("", freq1, amplitude, duration, sample_rate, False)
    signal_2 = get_pure_tone_wav("", freq2, amplitude, duration, sample_rate, False)
    signal_3 = get_pure_tone_wav("", freq3, amplitude, duration, sample_rate, False)
    signal = signal_1 + signal_2 + signal_3
    (p, t, s, pc, S) = autocorrelation_wk_sqrt(signal, sample_rate, tw=window_size_segs)
    print("Winner of each window: ")
    print(str(p) + "\n")

    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_6a.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

########################################### ITEM 7 #######################################

def solve_item_07():
    print("\n###########################################################")
    print("##                      ITEM 7                           ##")
    print("###########################################################\n")

    # ITEM 7.A
    print("ITEM 7.A: Autocorrelation + Wiener-Khinchin + spectrum sqrt + cochlea distribution algorithm implemented in the file: ./pitch_algorithms/a7.py\n")

    freq1 = 650
    freq2 = 950
    freq3 = 1250
    duration = 5.0
    amplitude = 1.0
    sample_rate = 8000
    window_size_segs = 1
    signal_1 = get_pure_tone_wav("", freq1, amplitude, duration, sample_rate, False)
    signal_2 = get_pure_tone_wav("", freq2, amplitude, duration, sample_rate, False)
    signal_3 = get_pure_tone_wav("", freq3, amplitude, duration, sample_rate, False)
    signal = signal_1 + signal_2 + signal_3
    (p, t, s, pc, S) = autocorrelation_wk_cochlea(signal, sample_rate, tw=window_size_segs)
    print("Winner of each window: ")
    print(str(p) + "\n")

    print("S Dimensions: " + str(S.shape[0]) + " X " + str(S.shape[1]))
    print("-> " + str(S.shape[0]) + " candidates of pitch")
    print("-> " + str(S.shape[1]) + " samples over time")
    title = "Scores for the instant ~" + str((duration / 2) * 1000) + " ms"
    filename = result_folder + "score_of_middle_time_7a.pdf"
    plot_canditates_in_time(S, pc, title, filename)
    print(title + " plotted and saved.\n")

########################################### ITEM 8 #######################################

def solve_item_08():
    print("\n###########################################################")
    print("##                      ITEM 8                           ##")
    print("###########################################################\n")

    m_periodic_names = ["a", "e", "u"]
    f_periodic_names = ["i", "o", "m"]
    sample_rate = 16000
    window_size_segs = 1
    print("ITEM 8.A:")
    for index in range(len(m_periodic_names)):
        wav_path_m = periodic_path + m_periodic_names[index] + "m1.wav"
        wav_path_f = periodic_path + f_periodic_names[index] + "f1.wav"
        signal_m = open_wav_file(wav_path_m)
        signal_f = open_wav_file(wav_path_f)
        (p_m, t_m, s_m, pc_m, S_m) = autocorrelation_wk_cochlea(signal_m, sample_rate, tw=window_size_segs, dt=0.001)
        (p_f, t_f, s_f, pc_f, S_f) = autocorrelation_wk_cochlea(signal_f, sample_rate, tw=window_size_segs, dt=0.001)
        #print("Winner of each window: ")
        #print(str(p))
        filename_m = result_folder + m_periodic_names[index] + "_score_vs_time_8a_m.pdf"
        filename_f = result_folder + f_periodic_names[index] + "_score_vs_time_8a_f.pdf"
        title_m = "Scores vs Time for " + m_periodic_names[index] + "m1.wav"
        title_f = "Scores vs Time for " + f_periodic_names[index] + "f1.wav"
        plot_scores_in_time(t_m, s_m, title_m, filename_m)
        plot_scores_in_time(t_f, s_f, title_f, filename_f)
        filename_m = result_folder + m_periodic_names[index] + "_pitch_vs_time_8a_m.pdf"
        filename_f = result_folder + f_periodic_names[index] + "_pitch_vs_time_8a_f.pdf"
        title_m = "Pitch vs Time for " + m_periodic_names[index] + "m1.wav"
        title_f = "Pitch vs Time for " + f_periodic_names[index] + "f1.wav"
        plot_pitch_vs_time(t_m, p_m, title_m, filename_m)
        plot_pitch_vs_time(t_f, p_f, title_f, filename_f)

########################################### ITEM 9 #######################################

def solve_item_09():
    print("\n###########################################################")
    print("##                      ITEM 9                           ##")
    print("###########################################################\n")

    sample_rate = 16000
    window_size_segs = 1
    print("ITEM 9.A:")
    signal = open_wav_file(speech_path)
    (p, t, s, pc, S) = autocorrelation_wk_cochlea(signal, sample_rate, tw=window_size_segs, dt=0.001)
    #print("Winner of each window: ")
    #print(str(p))
    filename = result_folder + "poem_score_vs_time_9a.pdf"
    title = "Scores vs Time for a poem: pirata.wav"
    plot_scores_in_time(t, s, title, filename)
    filename = result_folder + "poem_pitch_vs_time_9a.pdf"
    title = "Pitch vs Time for a poem: pirata.wav"
    xlims = (0,50)
    ylims = (0,900)
    plot_pitch_vs_time(t, p, title, filename, xlims, ylims)


########################################### ITEM 10 #######################################

def solve_item_10():
    print("\n###########################################################")
    print("##                      ITEM 10                           ##")
    print("###########################################################\n")

    filenames_m = os.listdir(m_speech_folder)
    filenames_f = os.listdir(f_speech_folder)
    N = len(filenames_m)
    window_size_segs = 1
    num_name = ["%03d" % i for i in range(1, int(N/2 + 1))]
    sample_rate = 20000
    error_M = []
    error_F = []
    tmp_error = []
    #for j in range(int(N/2)):
    for j in range(1):
        sound_M = m_speech_folder + "rl" + num_name[j] + ".wav"
        sound_F = f_speech_folder + "sb" + num_name[j] + ".wav"
        scores_M = m_speech_folder + "rl" + num_name[j] + ".f0"
        scores_F = f_speech_folder + "sb" + num_name[j] + ".f0"
        ref_data_M_a, ref_data_M_b = from_file_columns_to_arrays(scores_M)
        ref_data_F_a, ref_data_F_b = from_file_columns_to_arrays(scores_F)
        signal_M = open_wav_file(sound_M, sample_rate=sample_rate)
        signal_F = open_wav_file(sound_F, sample_rate=sample_rate)
        (p_M, t_M, s_M, pc_M, S_M) = autocorrelation_wk(signal_M, sample_rate, n=10, plim=[69,1264], tw=window_size_segs, dt=0.001)
        (p_F, t_F, s_F, pc_F, S_F) = autocorrelation_wk(signal_F, sample_rate, n=10, plim=[120,400], tw=window_size_segs, dt=0.001)
        print(ref_data_M_a)
        print(ref_data_M_b)
        print("-----------------------------------------")
        print(pc_M)
        print(s_M)

############################################ MAIN ########################################

def main():
    create_folder(result_folder)
    solve_item_01()
    solve_item_02()
    solve_item_03()
    solve_item_04()
    solve_item_05()
    solve_item_06()
    solve_item_07()
    solve_item_08()
    solve_item_09()
    #solve_item_10()

if __name__== "__main__":
  main()