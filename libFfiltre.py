import numpy as np
import math as m
import scipy.io.wavfile as wi
import wave as w
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
import struct

class wavFilter:

	waveparam = None

	SamplingRate = 44100
	NumberofChannels = 1
	NumberofFrames = 0

	TranLossPrecent = []

	def open_file(self, path):
		op=wi.read(path)
		liste_frames=op[1]
		#rate=op[0]
		#print(rate) #framerate
		#print(len(liste_frames)) #nFrames

		# open WAVE file
		spf = w.open(path,'r')

		# Extract WAVE file header informations
		#global waveparam
		self.waveparam	= spf.getparams()

		self.SamplingRate = self.waveparam.framerate
		self.NumberofFrames = self.waveparam.nframes

		print('\nWAVE HEADER INFORMATION:',path)
		print('Sample Rate: ', self.waveparam.framerate, 'Hz')
		print('Amp Width: ', self.waveparam.sampwidth)
		print('Number of Channels: ', self.waveparam.nchannels, 'channels')
		print('Number of Frames: ', self.waveparam.nframes, 'frames')
		print('comptype : ', self.waveparam.comptype)
		print('compname : ', self.waveparam.compname)

		spf.close()
		return(liste_frames)

	def writefile(self, e, nom):
		out=w.open(nom,'w')
		print('\nWAVE out INFORMATION:',nom)
		print(self.waveparam)
		# out.setparams((waveparam) )
		# waveparam.nchannels maybe 2 ,but we can not support it yet
		nchannels=1
		out.setparams((nchannels, self.waveparam.sampwidth, self.waveparam.framerate, self.waveparam.nframes, self.waveparam.comptype, self.waveparam.compname))
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

	def channels2one(self, e):
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

	def CaculateTranLossPrecent(self, arr) :
		self.TranLossPrecent = [1.0] * self.SamplingRate

		for i in range(1, int( self.SamplingRate / 2 ) ):
			currentFreq = i
			precent = 1.0
			# if i > self.SamplingRate / 2 :
			# 	currentFreq = self.SamplingRate - i

			for j in range( len( arr )  ):
				# precent *= m.pow( arr[j].a, arr[j].f / currentFreq ) 
				frepercen = (arr[j][0] / currentFreq ) if ( arr[j][0] < currentFreq )  else ( currentFreq / arr[j][0]  )
				precent *= m.pow( arr[j][1], frepercen ) 

			self.TranLossPrecent[i] = precent
			self.TranLossPrecent[self.SamplingRate - i] = precent
		# debug :
		for i in range( len( self.TranLossPrecent ) ):
			print("debug TranLossPrecent  ",i,self.TranLossPrecent[i])

	def fourrier(self, signal):
		
		self.show_test(signal)

		# debug 
		# for amp in amplitude:
		# 	print( amp.real , amp.imag )
		
		# ffttest = fft(signal)
		# iffttest = ifft(ffttest)	
		ifftdata = self.transloss(signal)

		# debug 
		# for i in range( len(amplitude) ):
		# 	print( i, amplitude[i].real , amplitude[i].imag )

		self.showResual(ifftdata)
		return ifftdata

	def transloss(self,signal):
		startPos=0
		ret=[]
		#fftFreq = np.linspace(0, self.SamplingRate/2, self.SamplingRate/2)
		while startPos < len(signal):
			## todo test if it is 1.5 seconds should be zero
			tempSiganl = signal[startPos : startPos+ self.SamplingRate]
			startPos += self.SamplingRate
			fftTeamp = fft(tempSiganl)
			for fftIndex in range(  len(fftTeamp)  ):
				# freq = 0
				# if fftIndex < self.SamplingRate :
				# 	freq =  fftIndex 
				# else :
				# 	freq =  self.SamplingRate -fftIndex
				
				# ## test: there should be tranLossArr[0:44100] 
				# if 390 < freq and freq < 410 :
				# 	print("debug--------------freq: " , freq)
				# 	# fftTeamp[fftIndex].imag *= 0.1
				# 	# fftTeamp[fftIndex].real *= 0.1
				# 	fftTeamp[fftIndex] *= 0.1
				fftTeamp[fftIndex] *= self.TranLossPrecent[fftIndex]

			ifftTeamp = ifft(fftTeamp)
			ret.extend( ifftTeamp )
		return ret

	
	# todo: my 傅里叶变换
	def fft1(self, signal):

		# t = np.linspace(0, 1.0, len(signal))
		# f = np.arange(len(signal)/2+1, dtype=complex)
		# for index in range(len(f)):
		# 	f[index]=complex(np.sum(np.cos(2*np.pi*index*t)*signal), -np.sum(np.sin(2*np.pi*index*t)*signal))
		return fft(signal)

	def show_test(self, signal):
		# time：  len(signal) =》 NumberofFrames
		x=np.linspace(0, self.NumberofFrames / self.SamplingRate , len(signal) )      

		y=signal

		yf1 = abs( fft(y) ) / len(x)           #归一化处理
		yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

		plt.subplot(121)
		plt.plot(x,y)   
		plt.title('Original wave')

		freqs = np.linspace(0, self.SamplingRate/2, len(yf1)/2)
		plt.subplot(122)
		plt.plot(freqs,yf2[0:len(freqs)],'b')
		plt.title('FFT of wave)',fontsize=10,color='#F08080')
		plt.show()

	def showResual(self, signal):
		# time：  len(signal) =》 NumberofFrames
		x=np.linspace(0, self.NumberofFrames / self.SamplingRate , len(signal) )      

		# 设置需要采样的信号，频率分量有180，390和600
		# y=7*np.sin(2*np.pi*400*x) # + 2.8*np.sin(2*np.pi*390*x)+5.1*np.sin(2*np.pi*600*x)
		y=signal

		# yy=fft(y)                     #快速傅里叶变换
		# yreal = yy.real               # 获取实数部分
		# yimag = yy.imag               # 获取虚数部分

		# yf=abs(fft(y))                # 取绝对值
		yf1 = abs( fft(y) ) / len(x)           #归一化处理
		yf2 = yf1[range(int(len(x)/2))]  #由于对称性，只取一半区间

		# xf = np.arange(len(y))        # 频率
		# xf1 = xf
		# xf2 =  np.linspace(0, self.SamplingRate/2, len(yf2)/2+1) #xf[range(int(len(x)/2))]  #取一半区间


		plt.subplot(121)
		plt.plot(x,y)   
		plt.title('Original wave')

		# plt.subplot(222)
		# plt.plot(xf,yf,'r')
		# plt.title('FFT of wave(two sides frequency range)',fontsize=7,color='#7A378B')  #注意这里的颜色可以查询颜色代码表

		# plt.subplot(223)
		# plt.plot(xf1,yf1,'g')
		# plt.title('FFT of wave(normalization)',fontsize=9,color='r')

		freqs = np.linspace(0, self.SamplingRate/2, len(yf1)/2)
		plt.subplot(122)
		plt.plot(freqs,yf2[0:len(freqs)],'b')
		plt.title('FFT of wave)',fontsize=10,color='#F08080')
		plt.show()




# ------------------------------------------ #
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
