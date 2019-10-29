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
        @input:     Rsig(list)  = temporally accumulated values of Red pixels
        @input:     Gsig(list)  = temporally accumulated values of Green pixels 
        @input:     Bsig(list)  = temporally accumulated values of Blue pixels 
        @input:     fps = sampling frequency i.e. frame rate of camera (int/double)
        @input:     old = heart rate reading computed in previous window (int)
        
        @function:
                    ->convolve each signal with a bandpass filter
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


        #computing the PSD for each component
        ps1 = np.abs(np.fft.fft(PC1))**2
        ps2 = np.abs(np.fft.fft(PC2))**2
        ps1 = max(ps1)
        ps2 = max(ps2)
        
    
        PC1 = self.running_mean(PC1,2)
        PC2 = self.running_mean(PC2,2)
        
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
            return (bpm1,PC1)
        elif (ps2-ps1)>1000:
            return bpm2,PC2
        elif abs(ps1-ps2)<1000:
            if old==0:
                return bpm1,PC1
            else:
                if abs(bpm1-old)<abs(bpm2-old):
                    return bpm1,PC1
                else:
                    return bpm2,PC2
         
    

    

        
    
    


        
        
        
        
        
        
