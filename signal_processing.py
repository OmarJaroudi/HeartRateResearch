import numpy as np
from scipy.signal import firwin,resample
import pandas as pd
from sklearn.decomposition import PCA
from jade import jadeR
import math
from matplotlib import pyplot as plt
import scipy 
from scipy.signal import butter, lfilter
from scipy.signal import freqz

class Signal_processing():
    def __init__(self):
        self.a = 1
       
    def normalize(self,data):
        """
        @input:     data =  data to normalize (1d numpy array)
        @function:  subtracts the mean and divides by the standard deviation (mutable)
        @return:    normalized data
        """
        std = np.std(data,axis = 0)
        data = data-np.mean(data,axis = 0)
        data = data/std
        return data
    def bandpass_firwin(self,ntaps, lowcut, highcut, fs, window='hamming'):
        """
        @input:     ntaps = # of points desired (int)
        @input:     lowcut = low frequency in Hz (double)
        @input:     highcut = high frequency in Hz (double)
        @input:     fs = sampling frequency in Hz
        
        @function: Implements a hamming window with 'ntaps' points
        
        @returns: Hammign window filter to be convolved with a signal later on
        """
        nyq = 0.5 * fs
        taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
                      window=window, scale=False)
        return taps

    def running_mean(self,x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        return (cumsum[N:] - cumsum[:-N]) / float(N)

        
    def Analyze(self,Rsig,Gsig,Bsig,fps,old):
        
        """
        @input:     Rsig = temporally accumulated values of Red pixels(list)
        @input:     Gsig = temporally accumulated values of Green pixels(list)
        @input:     Bsig = temporally accumulated values of Blue pixels(list)
        @input:     fps = sampling frequency i.e. frame rate of camera (int/double)
        @input:     old = heart rate reading computed in previous window (int)
        
        @function:  -> convolve each signal with a bandpass filter
                    -> merge signals in pandas DataFrame
                    -> perform PCA
                    -> output = 3 signals (principal components)
                    -> perform fourier analysis to find frequency for each principal component
                    -> choose principal component with highest PSD magnitude
                        -> conditional: if PSD magnitudes are within 1000 of each other, take the value thats closest to the 
                                        previous bpm reading
                                        
        @return:    bpm = heart rate reading corresponding to the RGB values passed to the function (int)
                    Raw principal component which produced this heart rate reading (numpy array)

        """
        Gsig1 = Gsig
        Bsig1 = Bsig
        Rsig1 = Rsig
        
        #preparing the bandpass filter
        bandpass_filter = self.bandpass_firwin(128,0.7,3,fps,'hamming')
    
        Gsig1 = np.convolve(Gsig1,bandpass_filter)
        Rsig1 = np.convolve(Rsig1,bandpass_filter)
        Bsig1 = np.convolve(Bsig1,bandpass_filter)
        
        Gsig1 = Gsig1[50:350]
        Bsig1 = Bsig1[50:350]
        Rsig1 = Rsig1[50:350]
        #preparing the pandas Dataframe   
        dataset = pd.DataFrame({'Gsig':Gsig1,'Rsig':Rsig1,'Bsig':Bsig1})
        features = ['Gsig','Rsig','Bsig']
        data = dataset.loc[:,features].values
        pca = PCA (n_components=3) #preparing the PCA model 
        pca_res = pca.fit_transform(data) #actually performing PCA
        
        principalDf = pd.DataFrame(data = pca_res,columns = ['PC1','PC2','PC3'])
        
        #seperating each principal component
        PC1 = principalDf.loc[:,'PC1'].values
        PC2 = principalDf.loc[:,'PC2'].values
        PC3 = principalDf.loc[:,'PC3'].values


        #computing the PSD for each component
        ps1 = np.abs(np.fft.fft(PC1))**2
        ps2 = np.abs(np.fft.fft(PC2))**2
        ps1 = max(ps1)
        ps2 = max(ps2)
        

        #performing fourier analysis
        w = np.fft.fft(PC1)
        freqs = np.fft.fftfreq(len(w))
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * fps)
        bpm1 = int(freq_in_hertz*60)

        
        #performing fourier analysis
        w = np.fft.fft(PC2)
        freqs = np.fft.fftfreq(len(w))
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * fps)
        bpm2 = int(freq_in_hertz*60)
        
        
        #choosing the source signal to be used for fourier analysis based on PSD peak
        if(ps1-ps2)>1000:
            return (bpm1,PC1[210:])
        elif (ps2-ps1)>1000:
            return bpm2,PC2[210:]
        elif abs(ps1-ps2)<1000:
            if old==0:
                return bpm1,PC1
            else:
                if abs(bpm1-old)<abs(bpm2-old):
                    return bpm1,PC1[210:]
                else:
                    return bpm2,PC2[210:]
        
    def detrend(self,signal, Lambda):
        # Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
        # Written by Guillaume Heusch <guillaume.heusch@idiap.ch>,
        # 
        # This file is part of bob.rpgg.base.
        # 
        # bob.rppg.base is free software: you can redistribute it and/or modify
        # it under the terms of the GNU General Public License version 3 as
        # published by the Free Software Foundation.
        # 
        # bob.rppg.base is distributed in the hope that it will be useful,
        # but WITHOUT ANY WARRANTY; without even the implied warranty of
        # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
        # GNU General Public License for more details.
        # 
        # You should have received a copy of the GNU General Public License
        # along with bob.rppg.base. If not, see <http://www.gnu.org/licenses/>.
        
        """
        This code is based on the following article "An advanced detrending method with application
        to HRV analysis". Tarvainen et al., IEEE Trans on Biomedical Engineering, 2002.
          
        **Parameters**
        ``signal`` (1d numpy array):
        The signal where you want to remove the trend.
        
        ``Lambda`` (int):
        The smoothing parameter.
        
        **Returns**
          
        ``filtered_signal`` (1d numpy array):
        The detrended signal.
          """
        signal_length = signal.shape[0]

        # observation matrix
        H = np.identity(signal_length) 
        
        # second-order difference matrix
        from scipy.sparse import spdiags
        ones = np.ones(signal_length)
        minus_twos = -2*np.ones(signal_length)
        diags_data = np.array([ones, minus_twos, ones])
        diags_index = np.array([0, 1, 2])
        D = spdiags(diags_data, diags_index, (signal_length-2), signal_length).toarray()
        filtered_signal = np.dot((H - np.linalg.inv(H + (Lambda**2) * np.dot(D.T, D))), signal)
        
        return filtered_signal
         
    

    def HRV(self,data,Min,Max,window,IBI=False,SDNN = False,BPM = False):
        hr = []
        """
        @input:     data = raw PPG signal to perform HRV analysis
        @input:     Min = minimum extracted heart rate from the PPG signal over time
        @input:     Max = maximum extracted heart rate from the PPg signal over time
        
        @function:  perform HRV analysis by computing several HRV metrics
                    -> bandpass filter the PPG signal by using Min and Max as bounding frequencies
                    -> interpolate the PPg signal to a frequency of 240Hz
                    -> compute the peaks in a 0.25 sec sliding window
                    -> in each 10 sec window:
                        -> reject peaks that are 20% different from the median
                        -> reject the window as a whole if more than 50% of the peaks have been removed
                    
        """
#        low = (Min-5)/60
#        high = (Max-5)/60
        f_old = len(data)/window
#        f = self.butter_bandpass_filter(data,low,high,f_old,2)
        
        f = self.interpolate(data,f_old,240,window)

        peaks = []
        x = 0
        j = 0
        correctedIntervals = np.array([])

        while j<=(len(f)-2400):
            temp = f[j:j+2400]
            i = 0
            while i <= (len(temp)-60):
                timeWindow = temp[i:i+60]
                m = -99999
                idx = 0
                
                for y in timeWindow:
                    if y>m:
                        m = y
                        idx = x
                    x+=1
                peaks.append((idx,m))
                i+=60
            peaks2 = []
            i=0
            while i<len(peaks):
                if i==len(peaks)-1 and peaks[i][1]>peaks[i-1][1]:
                    peaks2.append(peaks[i])
                    break
                if i==0 and peaks[i][1]>peaks[i+1][1]:
                    peaks2.append(peaks[i])
                    i+=2
                    continue
                if peaks[i][1]>peaks[i-1][1] and peaks[i][1]>peaks[i+1][1]:
                    peaks2.append(peaks[i])
                    i+=2
                    continue
                i+=1
            intervals = []
            for i in range(1,len(peaks2)):
                intervals.append((peaks2[i][0]-peaks2[i-1][0])/240)
            
            intervals = np.array(intervals)
            median = np.median(intervals)
            tempIntervals = np.array([])
            for i in intervals:
                if (abs(i-median)/((median+i)/2))<=0.2:
                    tempIntervals = np.append(tempIntervals,i)
            if (len(tempIntervals)>0.3*len(intervals)):
                hr.append(60/np.mean(tempIntervals))
                correctedIntervals = np.append(correctedIntervals,tempIntervals)
            if len(f)-j<2400:
                j = len(f)-2400
                continue
            j+=2400
            
        correctedIntervals.flatten()
        if BPM==True:
            print("HR = " + str(np.mean(hr)))
        if IBI==True:
            print("IBI = " + str(np.mean(correctedIntervals)*1000))
        if SDNN==True and window>70:
            print("SDNN = " + str(np.std(correctedIntervals)*1000))
        elif SDNN==True and window<70:
            print("Warning: SDNN value requested but window size is less than 70 sec")
        
        
        return hr
        
    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        """
        intermediate function used to build a bandpass filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        """
        bandpass filter
        """
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    
    def interpolate(self,data,f_old,f_new,window):
        """
        @input: data = data to interpolate
        @input: f_old = original frequency of the data i.e. sampling rate
        @input: f_new = new desired frequency
        @input: window = time window of the data
        
        @function: upsamples the data through cubic spline interpolation
        
        @output: interpolated data
        
        
        """
        from scipy.interpolate import interp1d
        x = np.linspace(0,window,num=f_old*window)
        f = interp1d(x,data,kind='cubic')
        xNew = np.linspace(0,window,f_new*window)
        dataNew = f(xNew)
        return dataNew


        
        
        
        
        
        