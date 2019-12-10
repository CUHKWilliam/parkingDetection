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


soundFileListDir = "../test-10-13/"
soundFileList = os.listdir(soundFileListDir)


dataUnderList = []
dataPeripheralList = []
interval = 0.2


def getData():
    for soundFile in soundFileList:
        flag_under = False
        print("soundFile:", soundFile)
        if ".wav" not in soundFile:
            continue
        if soundFile.split("_")[1].strip() == "under":
            flag_under = True

        wave_data, time,_ = read_wave_data(soundFileListDir + soundFile)
        wave_data_per_second, time_data_per_second = cut_wave_data(wave_data, time, interval)

        print("end time:", time_data_per_second[-1][-1] - 3 )
        print("len(wave_data_per_second)", len(wave_data_per_second))
        print("len(wave_data_per_second[0])", len(wave_data_per_second[0]))

        counter_thisfile = 0
        for i in range(int(2/interval), int((time_data_per_second[-1][-1] - 3)/interval)):
            time_data_this_second = time_data_per_second[i]
            wave_data_this_second = wave_data_per_second[i]
            sum = 0
            for j in range(len(wave_data_this_second)):
                sum += abs(wave_data_this_second[j])
            for j in range(len(wave_data_this_second)):
                wave_data_this_second[j] = wave_data_this_second[j] / sum

            # conv_data_this_second = convolve(np.flipud(wave_data_this_second), wave_data_this_second)
            # conv_series_this_second = np.arange(len(conv_data_this_second) / 2)
            # conv_data_this_second = conv_data_this_second[0: len(conv_series_this_second)]

            freq_data = np.arange(len(time_data_this_second) / 2)
            fft_data = abs(fft(wave_data_this_second))[0: len(freq_data)]

            # freq_data_conv = np.arange(len(conv_series_this_second) / 2)
            # fft_data_conv = abs(fft(conv_data_this_second))/(len(freq_data_conv))
            # fft_data_conv = fft_data_conv[0: len(freq_data_conv)]
            fft_data_smooth = []
            smooth_core = 30
            for j in range(len(fft_data)):
                avg = 0
                for i in range(-int(smooth_core/2), smooth_core):
                    if j + i < 0 or j + i >= len(fft_data):
                        avg += 0
                    else:
                        avg += fft_data[i + j]
                fft_data_smooth.append(avg / smooth_core)
            # choose data

            fft_data_smooth = fft_data_smooth[0:150]
            counter_thisfile += 1
            if flag_under == True:
                dataUnderList.append(fft_data_smooth)
            else:
                dataPeripheralList.append(fft_data_smooth)
        print("counter_thisfile:", counter_thisfile)

    train_test_split = 9
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



def getDataParallel():
    for soundFile in soundFileList:
        flag_under = False
        print("soundFile:", soundFile)
        if ".wav" not in soundFile:
            continue
        if soundFile.split("_")[1].strip() == "under":
            flag_under = True

        wave_data, time,_ = read_wave_data(soundFileListDir + soundFile)
        wave_data_per_second, time_data_per_second = cut_wave_data(wave_data, time, interval)

        print("end time:", time_data_per_second[-1][-1] - 3 )
        print("len(wave_data_per_second)", len(wave_data_per_second))
        print("len(wave_data_per_second[0])", len(wave_data_per_second[0]))

        counter_thisfile = 0
        for i in range(int(2/interval), int((time_data_per_second[-1][-1] - 3)/interval)):
            time_data_this_second = time_data_per_second[i]
            wave_data_this_second = wave_data_per_second[i]
            sum = 0
            for j in range(len(wave_data_this_second)):
                sum += abs(wave_data_this_second[j])
            for j in range(len(wave_data_this_second)):
                wave_data_this_second[j] = wave_data_this_second[j] / sum

            # conv_data_this_second = convolve(np.flipud(wave_data_this_second), wave_data_this_second)
            # conv_series_this_second = np.arange(len(conv_data_this_second) / 2)
            # conv_data_this_second = conv_data_this_second[0: len(conv_series_this_second)]

            freq_data = np.arange(len(time_data_this_second) / 2)
            fft_data = abs(fft(wave_data_this_second))[0: len(freq_data)]

            # freq_data_conv = np.arange(len(conv_series_this_second) / 2)
            # fft_data_conv = abs(fft(conv_data_this_second))/(len(freq_data_conv))
            # fft_data_conv = fft_data_conv[0: len(freq_data_conv)]
            fft_data_smooth = []
            smooth_core = 30
            for j in range(len(fft_data)):
                avg = 0
                for i in range(-int(smooth_core/2), smooth_core):
                    if j + i < 0 or j + i >= len(fft_data):
                        avg += 0
                    else:
                        avg += fft_data[i + j]
                fft_data_smooth.append(avg / smooth_core)
            # choose data

            fft_data_smooth = fft_data_smooth[0:150]

            wave_data_peripheral, time_peripheral, framerate = read_wave_data(
                "../sound_data_without_engine/bmw_peripheral_ssq.wav")
            wave_data_under, time_under, _ = read_wave_data("../sound_data_without_engine/bmw_under_twl.wav")

            # align process
            wave_data_peripheral, time_peripheral, wave_data_under, time_under = \
                align_under_peripheral_data(wave_data_peripheral, time_peripheral, wave_data_under, time_under,
                                            framerate, 5)
            wave_data_per_second_peripheral, time_data_per_second_peripheral = cut_wave_data(wave_data_peripheral,
                                                                                             time_peripheral, interval)
            wave_data_per_second_under, time_data_per_second_under = cut_wave_data(wave_data_under, time_under,
                                                                                   interval)

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

                # print("wave_data_this_second_peripheral", wave_data_this_second_peripheral[0: ])
                # print("wave_data_this_second_under", wave_data_this_second_under)

                # get peripheral conv/fft/fft_conv data
                # conv_data_this_second_peripheral = convolve(np.flipud(wave_data_this_second_peripheral), wave_data_this_second_peripheral)
                # conv_series_this_second_peripheral = np.arange(len(conv_data_this_second_peripheral) / 2)
                # conv_data_this_second_peripheral = conv_data_this_second_peripheral[0: len(conv_series_this_second_peripheral)]

                freq_data_peripheral = np.arange(len(time_data_this_second_peripheral) / 2)
                fft_data_peripheral = abs(fft(wave_data_this_second_peripheral))[0: len(freq_data_peripheral)]

                # freq_data_conv_peripheral = np.arange(len(conv_series_this_second_peripheral) / 2)
                # fft_data_conv_peripheral = abs(fft(conv_data_this_second_peripheral)) / (len(freq_data_conv_peripheral))
                # fft_data_conv_peripheral = fft_data_conv_peripheral[0: len(freq_data_conv_peripheral)]

                # get under conv/fft/fft_conv data

                # conv_data_this_second_under = convolve(np.flipud(wave_data_this_second_under),
                #                                             wave_data_this_second_under)
                # conv_series_this_second_under = np.arange(len(conv_data_this_second_under) / 2)
                # conv_data_this_second_under = conv_data_this_second_under[0: len(conv_series_this_second_under)]

                freq_data_under = np.arange(len(time_data_this_second_under) / 2)
                fft_data_under = abs(fft(wave_data_this_second_under))[0: len(freq_data_under)]

                # freq_data_conv_under = np.arange(len(conv_series_this_second_under) / 2)
                # fft_data_conv_under = abs(fft(conv_data_this_second_under)) / (len(freq_data_conv_under))
                # fft_data_conv_under = fft_data_conv_under[0: len(freq_data_conv_under)]

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
            counter_thisfile += 1
            if flag_under == True:
                dataUnderList.append(fft_data_smooth)
            else:
                dataPeripheralList.append(fft_data_smooth)
        print("counter_thisfile:", counter_thisfile)

    train_test_split = 9
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
