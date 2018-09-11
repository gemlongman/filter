import numpy as np
import math as m
import scipy.io.wavfile as wi
import wave as w
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import struct

waveparam = None

def open_file(path):
	op=wi.read(path)
	liste_frames=op[1]
	#rate=op[0]
	#print(rate) #framerate
	#print(len(liste_frames)) #nFrames

	# open WAVE file
	spf = w.open(path,'r')

	# Extract WAVE file header informations
	global waveparam
	waveparam	= spf.getparams()

	print('\nWAVE HEADER INFORMATION:',path)
	print('Sample Rate: ', waveparam.framerate, 'Hz')
	print('Amp Width: ', waveparam.sampwidth)
	print('Number of Channels: ', waveparam.nchannels, 'channels')
	print('Number of Frames: ', waveparam.nframes, 'frames')
	print('comptype : ', waveparam.comptype)
	print('compname : ', waveparam.compname)

	spf.close()
	return(liste_frames)

def writefile(e,nom):
	out=w.open(nom,'w')
	print('\nWAVE out INFORMATION:',nom)
	print(waveparam)
	# out.setparams((waveparam) )
	# waveparam.nchannels maybe 2 ,but we can not support it yet
	nchannels=1
	out.setparams((nchannels, waveparam.sampwidth, waveparam.framerate, waveparam.nframes, waveparam.comptype, waveparam.compname))
	if nchannels == 1:
		for k in e:
			fra=w.struct.pack("h", int(k))
			out.writeframes(fra)
	elif nchannels == 2:
		for k in e:
			fra=w.struct.pack("hh",k[0],k[1])
			out.writeframes(fra)
	else:
		for k in e:
			fra=w.struct.pack("h",k[0])
			out.writeframes(fra)
	out.close()

def channels2one(e):
	##debug
	#print("len(e[0])",len(e[0]))
	if len( e[0] ) == 1:
		return e 
	#else e is Double channel
	s=[ e[0][0] / 2 + e[0][1] / 2 ]
	for k in range(len(e)):
		s += [ e[k][0] / 2 + e[k][1] / 2 ]
		##debug
		#print( k, e[k][0], e[k][1], s[k] )
	return(s)

def fourrier(signal):
	amplitude=abs(fft(signal))
	# amplitude=amplitude[0:(1/Te)//2]
	# freq=abs(f.fftfreq(len(signal),Te))
	# freq=freq[0:(1/Te)//2]
	# pl.plot(freq,amplitude)
	# pl.show()

	# debug 
	print( amplitude )
	return amplitude

def filtrephaut2(e,fo):
	wo=2*m.pi*fo
	Te=1/44100
	Q=1
	Y=np.zeros((2,len(e)))
	Y[0,0]=0
	Y[1,0]=0
	Ho=0.5
	for k in range(2,len(e)-2):
		Y[0,k+1]=Te*Y[1,k]+Y[0,k]
		Y[1,k+1]=Y[1,k]+Te*(Ho*(e[k+1]-2*e[k]+e[k-1])/(Te**2)-Y[0,k]*wo*wo-Y[1,k]*wo/Q)
	return(Y[0,:])
	
def filtrephaut1(e,fo):
	wo=2*m.pi*fo
	Te=1/44100
	s=[e[0]]
	for k in range(len(e)-2):
		s+=[e[k+1]-e[k]+s[k]*(1-Te*wo)]
	return(s)
	
def filtrepbas1(e,fo):
	s=[e[0]]
	tau=1/fo
	Te=1/44100
	for k in range(len(e)-1):
		s+=[(Te/tau)*e[k]+s[k]*(1-(Te/tau))]
	return(s)

def filtrepbas2(e,fo):
	Te=1/44100
	wo=2*m.pi*fo
	Q=1
	Ho=0.5
	a=(2*Q)/wo
	b=1/(wo**2)
	Y=np.zeros((2,len(e)))
	Y[0,0]=e[0]
	Y[1,0]=e[0]
	for k in range(len(e)-1):
		Y[0,k+1]=Te*Y[1,k]+Y[0,k]
		Y[1,k+1]=(Te/b)*(-Y[0,k]-a*Y[1,k]+e[k])+Y[1,k]
	return(Y[0,:])
	
def filtrepbande1(e,fo1,fo2):
	return(filtrepbas1(filtrephaut1(e,fo1),fo2))
	
def filtrepbande2(e,fo1,fo2):
	return(filtrepbas2(filtrephaut2(e,fo1),fo2))
