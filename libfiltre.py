import numpy as np
import math as m
import scipy.io.wavfile as wi
import wave as w

def open_file(path):
	op=wi.read(path)
	
	liste_frames=op[1]
	rate=op[0]
	return(liste_frames)
	
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

def writefile(e,nom):
	out=w.open(nom,'w')
	out.setparams((1,2,44100,len(e),'NONE','not compressed'))
	for k in e:
		try:
			fra=w.struct.pack("h",int(k))
			out.writeframes(fra)
		except:
			fra=w.struct.pack("h",0)#ecrit un silence à la place d'un son saturé (c'est moin moche...)
			out.writeframes(fra)
	out.close()

	
