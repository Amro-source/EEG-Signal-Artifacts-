# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:11:30 2021

@author: Zikantika
"""

import numpy as np
import scipy.signal as sps
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

np.random.seed(42)

n = 1000
fs = 100
noise = 3

# simulate EEG data with eye blinks
t = np.arange(n)
alpha = np.abs(np.sin(10 * t / fs)) - 0.5
alpha[n//2:] = 0
blink = np.zeros(n)
blink[n//2::200] += -1
blink = sps.lfilter(*sps.butter(2, [1*2/fs, 10*2/fs], 'bandpass'), blink)

frontal = blink * 200 + alpha * 10 + np.random.randn(n) * noise
central = blink * 100 + alpha * 15 + np.random.randn(n) * noise
parietal = blink * 10 + alpha * 25 + np.random.randn(n) * noise

eeg = np.stack([frontal, central, parietal]).T  # shape = (100, 3)

# plot original data
plt.subplot(3, 1, 1)
plt.plot(frontal + 50)
plt.plot(central + 100)
plt.plot(parietal + 150)
plt.yticks([50, 100, 150], ['Fz', 'Cz', 'Pz'])
plt.ylabel('original data')

# decompose EEG and plot components
ica = FastICA(n_components=3)
ica.fit(eeg)
components = ica.transform(eeg)

plt.subplot(3, 1, 2)
plt.plot([[np.nan, np.nan, np.nan]])  # advance the color cycler to give the components a different color :)
plt.plot(components + [0.5, 1.0, 1.5])
plt.yticks([0.5, 1.0, 1.5], ['0', '1', '2'])
plt.ylabel('components')

# looks like component #2 (brown) contains the eye blinks
# let's remove them (hard coded)!
components[:, 2] = 0

# reconstruct EEG without blinks
restored = ica.inverse_transform(components)

plt.subplot(3, 1, 3)
plt.plot(restored + [50, 100, 150])
plt.yticks([50, 100, 150], ['Fz', 'Cz', 'Pz'])
plt.ylabel('cleaned data')