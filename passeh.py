import numpy as np
import math as m
import scipy.fftpack as f
import matplotlib.pyplot as pl
import scipy.io.wavfile as wi
import wave as w
import struct

def open_file(path):
	op=wi.read(path)
	
	liste_frames=op[1]
	rate=op[0]
	return(liste_frames)
fo=3000
wo=2*m.pi*fo
Te=1/44100
def filtre(e):
	nom="testout10.wav"
	out=w.open(nom,'w')

	out.setparams((1,2,44100,len(e),'NONE','not compressed'))

	
	s=[e[0]]
	for k in range(len(e)-2):
		a=int(e[k+1]-e[k]+s[k]*(1-Te*wo))
		
		s+=[a]
		b=w.struct.pack("h",a)
		out.writeframes(b)
	
	out.close()

filtre(open_file("test3.wav"))
