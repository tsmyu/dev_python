import os
import sys
import glob
import struct
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fftpack
import pandas as pd


def read_wav(wavfile):
    """
    wav読み込みの関数
    input: waveファイル
    output: [frame data(np array), sampling frequency(int)]
    """
    wave_data = wave.open(wavfile,"r")
    frame_data = wave_data.readframes(wave_data.getnframes())
    frame_data = np.frombuffer(frame_data, dtype="int16")
    frame_data = frame_data/np.max(frame_data)
    fs = wave_data.getframerate()

    return [frame_data, fs]

def FFT(wav_data):
    """
    FFT関数
    input: wav_data([framedata, fs])
    output: spectrum
    """

    freq_data = fftpack.fft(wav_data[0])
    fft_fre = fftpack.fftfreq(n=wav_data[0].size, d=1/wav_data[1])

    return freq_data, fft_fre

def STFT(wav_data, win_width):
    """
    STFT関数
    input: wav_data([framedata, fs])
    output: spectrogram
    """

    f, t, spectrogram = signal.spectrogram(wav_data[0], fs=wav_data[1], nperseg=win_width, noverlap=win_width/2)

    return spectrogram


def get_bin_from_f(pickup_fre, fs, win_width):

    return int(pickup_fre / (fs / win_width))

def calc_pulse_num(all_data, noise_data, step_t, all_time, Fs):
    # print(type(all_data))
    cut_time = 0.05
    cut_len = int(cut_time / step_t)
    ana_num = len(all_data) // cut_len
    pulse_num = 0
    threshold = 1.8e-8
    for i in range(1,ana_num+1):
        cut_all_data = all_data[cut_len*(i-1):cut_len*i]    #0.1秒でカットしたdata　all 55k
        cut_noise_data = noise_data[cut_len*(i-1):cut_len*i]    #0.1で秒でカットしたdata noise 35k
        peak_all_data = cut_all_data[cut_all_data > threshold]  #閾値より大きいdata all
        peak_all_data_idx = np.where(cut_all_data >= threshold)[0]  #閾値と同じか，それより大きいピーク値 all
        peak_noise_data = cut_noise_data[cut_noise_data > threshold]    #閾値より大きいdata noise
        peak_noise_data_idx = np.where(cut_noise_data >= threshold)[0]  #閾値と同じか，それより大きいピーク値 noise
        # print(f"all data peak:{peak_all_data}, idx:{peak_all_data_idx}")
        # print(f"noise data peak:{peak_noise_data}, idx:{peak_noise_data_idx}")
        if len(peak_all_data) != 0:
            for noise_idx in peak_noise_data_idx:
                # print(noise_idx)
                peak_all_data_idx = peak_all_data_idx[~((noise_idx-1 == peak_all_data_idx) | (peak_all_data_idx == noise_idx + 1) | (peak_all_data_idx == noise_idx))]
                # peak_idx = peak_all_data_idx[~np.any((noise_idx-1 == peak_all_data_idx) | (peak_all_data_idx == noise_idx + 1) | (peak_all_data_idx == noise_idx))]

                # print(peak_all_data_idx)
            if len(peak_all_data_idx) > 0:
                pulse_num += 1

                print(np.mean(peak_all_data_idx)/Fs+(i-1)*0.1)

    print(f"pulse times : {pulse_num}")

    if pulse_num != 0:
        print(round(all_time / pulse_num, 2))
    else:
        print("NaN")



    return 0

def main(wavfile_list, pickup_all_f, pickup_noise_f):

 # read wave data
    wav_data_list = [read_wav(wav_data)
                     for wav_data in wavfile_list]
    # wav_raw_list = []
    # for wav_data in wav_data_list:
    #     print(wav_data)
    #     Fs = int(wav_data[1])
    #     wav_data = np.array(wav_data[0])
        # wav_noise = np.array(read_wav(wavfile_noise))[0]
        # wav_raw_data = np.array(wav_data[0])[:len(wav_noise)]
        # corr = get_correlation(wav_noise, wav_raw_data)
        # estimated_delay = corr.argmax() - (len(wav_raw_data) - 1)
        # print("estimated delay is " + str(estimated_delay))
        # noise_fin = np.zeros(estimated_delay)
        # wav_noise_tmp = np.concatenate(
        #     [wav_noise[estimated_delay:], noise_fin])
        # wav_tmp = wav_raw_data - wav_noise
        # create_wav(wav_data, Fs)
        # wav_name = os.path.splitext(os.path.basename(wav_data_name))[0]
        # wav_raw_list.append([wav_name, wav_data])

    # calc FFT
    for wav_data in wav_data_list:
        freq_data = FFT(wav_data)

        Fs = int(wav_data[1])


        # calc STFT
        win_width = 512
        spectrogram = STFT(wav_data, win_width)
        pickup_all_bin = get_bin_from_f(pickup_all_f, Fs, win_width)
        pickup_noise_bin = get_bin_from_f(pickup_noise_f, Fs, win_width)
        all_data = spectrogram[pickup_all_bin]
        noise_data = spectrogram[pickup_noise_bin]
        plt.plot(all_data)
        plt.plot(noise_data)
        plt.show()
    
    
        step_t = win_width / 2 / Fs
        all_time = round(len(wav_data[0])/Fs, 2)
        pulse_num = calc_pulse_num(all_data, noise_data, step_t, all_time, Fs)

        print(wav_data)


        # pulse_freq = len(np.round(wav_data[0]))/Fs/pulse_num
        # print(f"Pulse freqency:{pulse_freq}")

        # print(round(len(wav_data[0])/Fs, 2) / pulse_num)



        # show spectrogram
        # plt.imshow(np.log(np.flipud(spectrogram)))
        # fig = plt.figure(tight_layout=True, figsize = (12,8))
        # ax = fig.add_subplot(111)
        # ax.imshow(np.log(spectrogram), origin="lower")
        # plt.show()

        # print(spectrogram)

if __name__ == '__main__':
    argvs = sys.argv
    if len(argvs) <= 2:
        print(f"Usage: python {argvs[0]} [wav] [pickup all frequency [kHz]] [pickup noise frequency [kHz]]")

    # kHz input
    pickup_all_f = int(argvs[2]) * 1000
    # kHz input
    pickup_noise_f = int(argvs[3]) * 1000

    wavfile_list = sorted(glob.glob(f'{argvs[1]}/*.wav'))
    print(wavfile_list)

    main(wavfile_list, pickup_all_f, pickup_noise_f)