import numpy as np
import math as m
import scipy.fftpack as f
import matplotlib.pyplot as pl
from tkinter import *
import tkinter.filedialog as fd
import scipy.io.wavfile as wi
import wave as w

Te=1/44100
fo=150
wo=2*m.pi*fo
Q=1

tau=1/(2*m.pi*fo)
A1=1
A2=0
A3=0

f1=100
f2=5000
f3=8000

a=(2*Q)/wo
b=1/(wo**2)

listet=np.linspace(0,0.1,(1/Te))
e=[A1*m.sin(2*m.pi*f1*t)+A2*m.sin(2*m.pi*f2*t)+A3*m.sin(2*m.pi*f3*t) for t in listet]

#fenetre=Tk()
#class affichage:
	#def __init__(self):
		#self.patho=None
		#self.pathf=None
		#self.pulsation=60
	#def ouvrir(self):
		#a=fd.askopenfile()
		#b=str(a).split(" ")[1]
		#self.patho=b.split("'")[1]
		#labouvrir['text']=self.patho
	#def enregistrersous(self):
		#a=fd.asksaveasfile()
		#b=str(a).split(" ")[1]
		#self.pathf=b.split("'")[1]
		#labfermer['text']=self.pathf
#af=affichage()

def open_file(path):
	op=wi.read(path)
	
	liste_frames=op[1]
	rate=op[0]
	return(liste_frames)	
	


def filtre(e):
	Y=np.zeros((2,len(e)))
	nom="toccab.wav"
	out=w.open(nom,'w')
	out.setparams((1,2,44100,len(e),'NONE','not compressed'))
	Y[0,0]=e[0]
	Y[1,0]=e[0]
	for k in range(len(e)-1):
		fri=int(Y[0,k])
		frb=w.struct.pack("h",fri)
		out.writeframes(frb)
		Y[0,k+1]=Te*Y[1,k]+Y[0,k]
		Y[1,k+1]=(Te/b)*(-Y[0,k]-a*Y[1,k]+e[k])+Y[1,k]
	out.close()
	


filtre(open_file("tocca.wav"))

#bouton_ouvrir=Button(fenetre, text="open", command=af.ouvrir)
#bouton_ouvrir.grid(row=200,column=150,columnspan=20,rowspan=20)
#bouton_enregistrer=Button(fenetre, text="save as", command=af.enregistrersous)
#bouton_enregistrer.grid(row=400,column=150,columnspan=20,rowspan=20)
