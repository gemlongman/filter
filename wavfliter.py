# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 16:35:41 2017

@author: fleite
"""

import matplotlib.pyplot as plt
import numpy as np
import wave
import scipy.fftpack as sfft

outname = 'filtered.wav'

lowcut = 100
highcut = 1000
plotFigure = False

def bandpass_filtering(dataarray, lowcut, highcut, df):
    # calculate the low corner frequency of the band pass filter
    lowcutindex = int(lowcut/df)
    # calculate the high corner frequency of the band pass filter
    highcutindex = int(highcut/df)

# eliminate the data outside the filter's bandwidth
    for index in range(n):
        if(index < ((n/2) -1 - highcutindex )):
            dataarray[index] = 0
        if( ( index > ((n/2) -1 - lowcutindex )) & ( index < (n/2) )):
            dataarray[index] = 0
        if( ( index < ((n/2) -1 + lowcutindex )) & ( index > (n/2) )):
            dataarray[index] = 0
        if(index > ((n/2) -1 + highcutindex )):
            dataarray[index] = 0
                    
    return dataarray

def interpret_wav(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):

    if sample_width == 1:
        dtype = np.uint8 # unsigned char
    elif sample_width == 2:
        dtype = np.int16 # signed 2-byte short
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    channels = np.fromstring(raw_bytes, dtype=dtype)

    if interleaved:
        # channels are interleaved, i.e. sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # channels are not interleaved. All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)

    return channels

def convertToUINT8(dataarray):
    
    abssignalifft = abs(dataarray)

    filteredSignalInt = []

    for index in range(len(dataarray)):
        filteredSignalInt.append((abssignalifft[index]))
    
    filteredSignal = np.uint8(np.array(filteredSignalInt, dtype=int))
    
    return filteredSignal

######################################
#init

# open WAVE file
spf = wave.open('simple440add1000.wav','r')

# Extract WAVE file header informations
sampleRate = spf.getframerate()
ampWidth = spf.getsampwidth()
nChannels = spf.getnchannels()
nFrames = spf.getnframes()
print('\nWAVE HEADER INFORMATION:')
print('Sample Rate: ', sampleRate, 'Hz')
print('Amp Width: ', ampWidth)
print('Number of Channels: ', nChannels, 'channels')
print('Number of Frames: ', nFrames, 'frames\n')

# Extract Raw Audio from multi-channel Wav File
signal = spf.readframes(nFrames*nChannels)
channels = interpret_wav(signal, nFrames, nChannels, ampWidth, True)

# close WAVE file
spf.close()

######################################
#fft filtering
#

# calculate the sampling period
T = 1.0/sampleRate

signalfft = sfft.fft(channels[0])
signalfft = sfft.fftshift(signalfft)

n = signalfft.size
freq = sfft.fftfreq(n, d=T)
freq = sfft.fftshift(freq)

if(plotFigure == True):
    plt.figure(1)
    plt.title('Signal Spectrum')
    plt.plot(freq,signalfft)

# calculate the frequency step of FFT
df = freq[int((n/2)+1)]

signalfft = bandpass_filtering(signalfft, lowcut, highcut, df)

if(plotFigure == True):
    plt.figure(2)
    plt.title('Filtered Signal Spectrum')
    plt.plot(freq,signalfft)

#perform the inverse FFT
signalifft = sfft.ifft(signalfft)

# convert the filtered data to a uint8 array
filteredSignal = convertToUINT8(signalifft)
    
#plot the time domain signal
if(plotFigure == True):
    plt.figure(3)
    plt.title('Signal')
    plt.plot(channels[0])

#plot the time domain filtered signal
if(plotFigure == True):
    plt.figure(4)
    plt.title('Filtered Signal')
    plt.plot(filteredSignal)

#end of fft filtering
######################################


######################################
#output filtered WAVE file

#open the output WAVE file
wav_file = wave.open(outname, "w")
#write the header into the WAVE file
wav_file.setparams((2, ampWidth, sampleRate, nFrames, spf.getcomptype(), spf.getcompname()))
#write the samples into the WAVE file
wav_file.writeframes(filteredSignal.tobytes('C'))
print('The file ' + outname + ' was generated')
#close the output WAVE file
wav_file.close()

#end of output filtered WAVE file
######################################