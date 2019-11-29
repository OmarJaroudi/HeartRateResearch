# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 12:01:33 2019

@author: Omar Al Jaroudi
"""

from signal_processing import Signal_processing
sp = Signal_processing()
import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import heartpy as hp



ShimmerData = pd.read_csv('Mohd_3.csv',names = ['time','voltage'])
Benchmark = ShimmerData.loc[:,'voltage'].values

with open('Mohd_3_PPG.pkl','rb') as f:
    PPG = pickle.load(f)

temp = Benchmark[:700]
hr = np.array([])
for i in range(700,len(Benchmark)-300,300):
    j = i+300
    temp = np.append(temp,(Benchmark[i:j]))
    wd, m = hp.process(temp, 100)
    hr = np.append(hr,m['bpm'])
    temp = temp[300:]
heartRate = sp.HRV(PPG,0,0,120.3333333,IBI = True,BPM = True,SDNN = True)
print("")
wd,m = hp.process(Benchmark,sample_rate=100)
print("HR_REAL = " +str(m['bpm']))
print("IBI_REAL = " + str(m['ibi']))
print("SDNN_REAL = " +str(m['sdnn']))


x = np.linspace(0,120,len(heartRate))
x2 = np.linspace(0,120,len(hr))
plt.xlabel('Time (sec)')
plt.ylabel('HR (bpm)')
plt.plot(x2,hr,label = 'Real')
plt.plot(x,heartRate,'k',label = 'Estimated')
plt.legend(loc='upper right')
plt.ylim(50,120)
plt.show()
#print(len(PPG))
