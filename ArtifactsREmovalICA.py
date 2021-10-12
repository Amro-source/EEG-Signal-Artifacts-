# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:28:47 2021

@author: Zikantika
"""

#Did you notice your "components" are exactly the original signal scaled
# and upside down? That is because you cannot get more components 
# than signals.
#
#You need to perform the following steps:
#
#feed all EEG channels into the ICA
#manually remove the components that contain eye blinks or other artifacts
#reconstruct using the inverse transform
from PyEDF import EDFReader
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import FastICA, PCA

file_in = EDFReader()
file_in.open('EEG-1.edf')

#for x in numChannels:
  
ch1=file_in.readSamples(0, 0, 15000)
ch2=file_in.readSamples(1, 0, 15000)
ch3=file_in.readSamples(2, 0, 15000)
ch4=file_in.readSamples(3, 0, 15000)
ch5=file_in.readSamples(4, 0, 15000)

S = np.c_[ch1,ch2,ch3,ch4,ch5]


plt.plot(S)
plt.title("Raw EEG Signal  Channels 1-2-3-4-5" )
plt.xlabel('t (sec)')
plt.show()



transformer = FastICA(n_components=5,
         random_state=0)
X_transformed = transformer.fit_transform(S)

#reconstruct signal with independent components
components = transformer.fit_transform(X_transformed)
X_restored = transformer.inverse_transform(components)



plt.plot(X_restored)
plt.title("Raw EEG Signal  Channels 1-2-3-4-5 After FastICA" )
plt.xlabel('t (sec)')
plt.show()

print(X_transformed.shape)

X1=X_restored[0:,0]

plt.plot(X1)
plt.title("Raw EEG Signal  Channels 1 After FastICA" )
plt.xlabel('t (sec)')
plt.show()

X2=X_restored[0:,1]

plt.plot(X2)
plt.title("Raw EEG Signal  Channels 2 After FastICA" )
plt.xlabel('t (sec)')
plt.show()

X3=X_restored[0:,2]

plt.plot(X3)
plt.title("Raw EEG Signal  Channels 3 After FastICA" )
plt.xlabel('t (sec)')
plt.show()


X4=X_restored[0:,3]

plt.plot(X3)
plt.title("Raw EEG Signal  Channels 4 After FastICA" )
plt.xlabel('t (sec)')
plt.show()

X5=X_restored[0:,4]

plt.plot(X3)
plt.title("Raw EEG Signal  Channels 5 After FastICA" )
plt.xlabel('t (sec)')
plt.show()


##The channel containing some eye-blinks
#X = f1ep1_data[:,[4]]

##run ICA on signal
#ica = FastICA(n_components=2)
#ica.fit(X)
#
##reconstruct signal with independent components
#components = ica.fit_transform(X)
#X_restored = ica.inverse_transform(components)
#
#fig1 = plt.figure()
#plt.subplot(3,1,1)
#plt.title("Original signal")
#plt.plot(f1ep1_timescale, X)
#
#plt.subplot(3,1,2)
#plt.title("Components")
#plt.plot(f1ep1_timescale, components)
#
#plt.subplot(3,1,3)
#plt.title("Signal Reconstructed")
#plt.plot(f1ep1_timescale, X_restored)
#plt.draw()