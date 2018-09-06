import numpy as np
import math as m
import scipy.fftpack as f
import matplotlib.pyplot as pl
import scipy.io.wavfile as wi
import wave as w
import struct



Te=3*10**-4
fo=100
tau=1/(2*m.pi*fo)

A1=1
A2=1
A3=1

f1=100
f2=1000
f3=440


listet=np.linspace(0,0.1,(1/Te))
e=[A1*m.sin(2*m.pi*f1*t)+A2*m.sin(2*m.pi*f2*t +m.pi)+A3*m.sin(2*m.pi*f3*t+m.pi/2) for t in listet]

def fourrier(signal):
	amplitude=abs(f.fft(signal))
	amplitude=amplitude[0:(1/Te)//2]
	freq=abs(f.fftfreq(len(signal),Te))
	freq=freq[0:(1/Te)//2]
	pl.plot(freq,amplitude)
	pl.show()

def open_file(path):
	op=wi.read(path)
	
	liste_frames=op[1]
	#rate=op[0]
	#print("liste_frames:",liste_frames)
	return(liste_frames)

def filtre(e):
	nom="testout4.wav"
	out=w.open(nom,'w')
	out.setparams((1,2,44100,len(e),'NONE','not compressed'))

	s=[e[0]]
	
	# print("e:",e)
	# print("s:",s)
	for k in range(len(e)-1):
		a=(Te/tau)*e[k]+s[k]*(1-(Te/tau))
		#debug:
		s+=[a]
		#print("s:",s)
		b=w.struct.pack("h",int(a[0]))
		out.writeframes(b)
	
	out.close()

	
	

filtre(open_file("simple440add1000.wav"))


