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


def read_wave_data(file_path):
    f = wave.open(file_path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    print("framerate:", framerate)
    print("sampwidth:", sampwidth)
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype = np.short)
    wave_data.shape = -1, 2
    wave_data = wave_data.T
    time = np.arange(0, nframes) * (1.0/framerate)
    return wave_data, time


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


soundFileListDir = "../sound_data/"
soundFileList = os.listdir(soundFileListDir)


dataUnderList = []
dataPeripheralList = []
interval = 0.04


def getData():
    for soundFile in soundFileList:
        flag_under = False
        print("soundFile:", soundFile)
        if ".wav" not in soundFile:
            continue
        if soundFile.split("_")[1].strip() == "under":
            flag_under = True

        wave_data, time = read_wave_data(soundFileListDir + soundFile)
        wave_data_per_second, time_data_per_second = cut_wave_data(wave_data, time, interval)

        for i in range(int(2/interval), int((time_data_per_second[-1][-1] - 3)/interval)):
            time_data_this_second = time_data_per_second[i]
            wave_data_this_second = wave_data_per_second[i]
            sum = 0
            for j in range(len(wave_data_this_second)):
                sum += abs(wave_data_this_second[j])
            for j in range(len(wave_data_this_second)):
                wave_data_this_second[j] = wave_data_this_second[j] / sum

            conv_data_this_second = convolve(np.flipud(wave_data_this_second), wave_data_this_second)
            conv_series_this_second = np.arange(len(conv_data_this_second) / 2)
            conv_data_this_second = conv_data_this_second[0: len(conv_series_this_second)]


            fft_data = abs(fft(wave_data_this_second)) / (len(time_data_this_second) / 2)
            freq_data = np.arange(len(time_data_this_second) / 2)

            # freq_data_conv = np.arange(len(conv_series_this_second) / 2)
            # fft_data_conv = abs(fft(conv_data_this_second))/(len(freq_data_conv))
            # fft_data_conv = fft_data_conv[0: len(freq_data_conv)]

            # choose data

            if flag_under == True:
                dataUnderList.append(fft_data)
            else:
                dataPeripheralList.append(fft_data)

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

    trainList = dataTagList[0: int(len(dataTagList)*0.9)]
    testList = dataTagList[int(len(dataTagList)*0.9): len(dataTagList)]
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
        print("convert:", str(i))
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
    f = open("test_data.pkl" ,"wb")
    pickle.dump(testBatch, f)
    f.close()
    return trainList, testList


