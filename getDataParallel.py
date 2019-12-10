import matplotlib.pyplot as plt
from scipy.io import wavfile as wav
from scipy.fftpack import fft
from scipy.signal import convolve
import wave
import time as Time
import numpy as np
import os
import math
import pickle



# interval for cutting off the sound data
interval = 1

# train:test split ratio
train_test_split = 9


# Set soundFileDir to the directory containing your sound data file
soundFileListDir = ""
soundFileList = os.listdir(soundFileListDir)

dataUnderList = []
dataPeripheralList = []

def read_wave_data(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype = np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0/framerate)
    return wave_data, time, framerate


def cut_wave_data(wave_data, time, interval):
    idx_anchor = 0
    wave_data_per_second = []
    time_data_per_second = []
    second = interval
    wave_data_this_second = []
    time_data_this_second = []
    for idx_anchor in range(len(time)):
        if time[idx_anchor] < second:
            wave_data_this_second.append(wave_data[0][idx_anchor]/10)
            time_data_this_second.append(time[idx_anchor])
        else:
            wave_data_per_second.append(wave_data_this_second)
            time_data_per_second.append(time_data_this_second)
            wave_data_this_second = []
            time_data_this_second = []
            second += interval
    return wave_data_per_second, time_data_per_second


def align_under_peripheral_data(wave_data_peripheral, time_data_peripheral,
                                wave_data_under, time_data_under, framerate, sampleSecond):
    wave_data_under_sample = wave_data_under[0][0: framerate*sampleSecond]
    wave_data_peripheral_sample = wave_data_peripheral[0][0: framerate*sampleSecond]
    wave_data_under_sample_reverse = np.flipud(wave_data_under_sample)
    conv_data_peripheral_under_sample = convolve(wave_data_peripheral_sample, wave_data_under_sample_reverse)
    phase_diff = int((len(wave_data_peripheral_sample) + len(wave_data_under_sample))/2) \
                 - np.argmax(conv_data_peripheral_under_sample)
    if phase_diff > 0:
        for i in range(phase_diff, len(wave_data_peripheral[0]) - phase_diff):
            wave_data_peripheral[0][i - phase_diff] = wave_data_peripheral[0][i]
    elif phase_diff < 0:
        phase_diff = -phase_diff
        for i in range(phase_diff, len(wave_data_under[0]) - phase_diff):
            wave_data_under[0][i - phase_diff] = wave_data_peripheral[0][i]
    phase_diff = abs(phase_diff)
    print("phase_diff:", phase_diff)
    return wave_data_peripheral[:][0: len(wave_data_peripheral[0]) - phase_diff], \
           time_data_peripheral[:][0: len(wave_data_peripheral[0]) - phase_diff], \
           wave_data_under[:][0: len(wave_data_under[0]) - phase_diff], \
           time_data_under[:][0: len(wave_data_under[0]) - phase_diff]

def getData():
    for soundFile in soundFileList:
        flag_withCar = True
        if ".wav" not in soundFile:
            continue
        if soundFile.split("_")[1].strip() == "withoutCar":
            flag_withCar = False
            soundFilePeripheral = soundFile
            if "twl" in soundFile:
                soundFileUnder= soundFile.replace("twl", "ssq")
            else:
                soundFileUnder = soundFile.replace("ssq", "twl")
        else:
            if "noise" in soundFile:
                continue
            if soundFile.split("_")[2].strip() == "peripheral":
                soundFilePeripheral = soundFile
                soundFileUnder = soundFilePeripheral.replace("peripheral", "under")
                if "twl" in soundFilePeripheral:
                    soundFileUnder = soundFileUnder.replace("twl", "ssq")
                else:
                    soundFileUnder = soundFileUnder.replace("ssq", "twl")
            else:
                soundFileUnder = soundFile
                soundFilePeripheral = soundFileUnder.replace("under", "peripheral")
                if "twl" in soundFileUnder:
                    soundFilePeripheral = soundFilePeripheral.replace("twl", "ssq")
                else:
                    soundFilePeripheral = soundFilePeripheral.replace("ssq", "twl")
        print("soundFileUnder:", soundFileUnder)
        print("soundFilePeripheral:", soundFilePeripheral)
        print("flag_withCar:", flag_withCar)

        wave_data_peripheral, time_peripheral, framerate = read_wave_data(
            soundFileListDir + soundFilePeripheral)
        wave_data_under, time_under, _ = read_wave_data(soundFileListDir + soundFileUnder)
        # wave_data_peripheral, time_peripheral, wave_data_under, time_under = \
        #     align_under_peripheral_data(wave_data_peripheral, time_peripheral, wave_data_under, time_under, framerate, 5)
        wave_data_per_second_peripheral, time_data_per_second_peripheral = cut_wave_data(wave_data_peripheral, time_peripheral, interval)
        wave_data_per_second_under, time_data_per_second_under = cut_wave_data(wave_data_under, time_under, interval)
        for i in range(int(2 / interval), int((time_data_per_second_peripheral[-1][-1] - 3) / interval)):
            # get peripheral time/wave data
            time_data_this_second_peripheral = time_data_per_second_peripheral[i]
            wave_data_this_second_peripheral = wave_data_per_second_peripheral[i]
            sum_peripheral = 0
            for j in range(len(wave_data_this_second_peripheral)):
                sum_peripheral += abs(wave_data_this_second_peripheral[j])
            for j in range(len(wave_data_this_second_peripheral)):
                wave_data_this_second_peripheral[j] = wave_data_this_second_peripheral[j] / sum_peripheral
            # get under time/wave data
            time_data_this_second_under = time_data_per_second_under[i]
            wave_data_this_second_under = wave_data_per_second_under[i]
            sum_under = 0
            for j in range(len(wave_data_this_second_under)):
                sum_under += abs(wave_data_this_second_under[j])
            for j in range(len(wave_data_this_second_under)):
                wave_data_this_second_under[j] = wave_data_this_second_under[j] / sum_under
            freq_data_peripheral = np.arange(len(time_data_this_second_peripheral) / 2)
            fft_data_peripheral = abs(fft(wave_data_this_second_peripheral))[0: len(freq_data_peripheral)]
            freq_data_under = np.arange(len(time_data_this_second_under) / 2)
            fft_data_under = abs(fft(wave_data_this_second_under))[0: len(freq_data_under)]
            fft_data_under_smooth = []
            fft_data_peripheral_smooth = []
            smooth_core = 30
            for j in range(len(fft_data_under)):
                avg_under = 0
                avg_peripheral = 0
                for i in range(-int(smooth_core / 2), smooth_core):
                    if j + i < 0 or j + i >= len(fft_data_under):
                        avg_under += 0
                        avg_under += 0
                    else:
                        avg_under += fft_data_under[i + j]
                        avg_peripheral += fft_data_peripheral[i + j]
                fft_data_under_smooth.append(avg_under / smooth_core)
                fft_data_peripheral_smooth.append(avg_peripheral / smooth_core)

            freq_data_ratio = freq_data_peripheral
            fft_data_ratio = [fft_data_under_smooth[x] / fft_data_peripheral_smooth[x] for x in
                              range(len(freq_data_ratio))]

            freq_data_ratio = freq_data_ratio
            fft_data_ratio = fft_data_ratio


            if flag_withCar == True:
                dataUnderList.append(fft_data_ratio)
            else:
                dataPeripheralList.append(fft_data_ratio)
        print("turn to the next")

    tagUnderList = [1 for x in range(len(dataUnderList))]
    tagPeripheralList = [0 for x in range(len(dataPeripheralList))]
    dataList = dataUnderList
    dataList.extend(dataPeripheralList)
    tagList = tagUnderList
    tagList.extend(tagPeripheralList)
    dataTagList = []
    for i in range(len(dataList)):
        dataTagList.append([dataList[i], tagList[i]])
    np.random.shuffle(dataTagList)
    trainList = dataTagList[0: int(len(dataTagList)*train_test_split/(train_test_split+1))]
    testList = dataTagList[int(len(dataTagList)*(train_test_split/(train_test_split+1))): len(dataTagList)]
    print("train size:", len(trainList))
    print("test size:", len(testList))

    BATCH_SIZE = 32
    trainBatch = []
    trainDataOneBatch = []
    tagDataOneBatch = []
    print("convert train/test data to batch")
    trainList = np.array(trainList)
    testList = np.array(testList)
    for i in range(len(trainList) - 2):
        trainDataOneBatch.append(trainList[i][0])
        tagDataOneBatch.append(trainList[i][1])
        if i % BATCH_SIZE == BATCH_SIZE - 1:
            trainBatch.append((trainDataOneBatch, tagDataOneBatch))
            trainDataOneBatch = []
            tagDataOneBatch = []

    testBatch = []
    for i in range(len(testList)):
        testBatch.append(testList[i][:, np.newaxis])

    print("store train/test data into file")
    f = open("train_data.pkl", "wb")
    pickle.dump(trainBatch, f)
    f.close()
    f = open("test_data.pkl", "wb")
    pickle.dump(testBatch, f)
    f.close()
    return trainList, testList
