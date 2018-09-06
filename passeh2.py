import numpy as np
import math as m
import scipy.fftpack as f
import matplotlib.pyplot as pl
from tkinter import *
import tkinter.filedialog as fd
import scipy.io.wavfile as wi
import wave as w

Te=1/44100
fo=3000
wo=2*m.pi*fo
Q=1
Ho=0.5

def open_file(path):
	op=wi.read(path)
	
	liste_frames=op[1]
	rate=op[0]
	return(liste_frames)	
	
def filtre(e):
	Y=np.zeros((2,len(e)))
	nom="toccah.wav"
	out=w.open(nom,'w')
	out.setparams((1,2,44100,len(e),'NONE','not compressed'))
	Y[0,0]=0
	Y[1,0]=0
	for k in range(2,len(e)-2):
		fri=int(Y[0,k])
		frb=w.struct.pack("h",fri)
		out.writeframes(frb)
		Y[0,k+1]=Te*Y[1,k]+Y[0,k]
		Y[1,k+1]=Y[1,k]+Te*(Ho*(e[k+1]-2*e[k]+e[k-1])/(Te**2)-Y[0,k]*wo*wo-Y[1,k]*wo/Q)
		print(Y[1,k+1])
	out.close()
filtre(open_file("simple440add1000.wav"))
