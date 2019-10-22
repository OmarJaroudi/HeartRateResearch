import numpy as np
import scipy
from scipy.signal import firwin
import pandas as pd
from sklearn.decomposition import PCA

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

    def Analyze(self,Rsig,Gsig,Bsig,fps):
        """
        @input:     Rsig = temporally accumulated values of Red pixels(list)
        @input:     Gsig = temporally accumulated values of Green pixels(list)
        @input:     Bsig = temporally accumulated values of Blue pixels(list)
        @input:     fps = sampling frequency i.e. frame rate of camera (int/double)
        
        @function:  detrend each signal -> normalize each signal -> merge signals in pandas DataFrame
                    -> perform PCA -> output = 3 signals (principal components) -> choose component with highest PSD magnitude
                    -> convolve chosen signal with bandpass filter -> perform fourier analysis to find frequency
        
        @return:    bpm = heart rate reading corresponding to the RGB values passed to the function (int)

        """
        
        
        Gsig1 = Gsig
        Bsig1 = Bsig
        Rsig1 = Rsig
        
        #detrending each signal using the detrend method with order 10
        Gsig1 = self.detrend(Gsig1,10)
        Rsig1 = self.detrend(Rsig1,10)
        Bsig1 = self.detrend(Bsig1,10)
        
        #normalizing the data
        Gsig1 = self.normalize(Gsig1)
        Rsig1 = self.normalize(Rsig1)
        Bsig1 = self.normalize(Bsig1)
        
        #preparing the bandpass filter
        bandpass_filter = self.bandpass_firwin(128,0.7,3,fps,'hamming')
    
    
        #preparing the pandas Dataframe   
        dataset = pd.DataFrame({'Gsig':Gsig1,'Rsig':Rsig1,'Bsig':Bsig1})
        features = ['Gsig','Rsig','Bsig']
        data = dataset.loc[:,features].values
        pca = PCA(n_components=3) #preparing the PCA model 
        pca_res = pca.fit_transform(data) #actually perfroming PCA
        
        principalDf = pd.DataFrame(data = pca_res,columns = ['PC1','PC2','PC3'])
        
        #seperating each principal component
        PC1 = principalDf.loc[:,'PC1'].values
        PC2 = principalDf.loc[:,'PC2'].values
        PC3 = principalDf.loc[:,'PC3'].values
        
        #computing the PSD for each component
        ps1 = np.abs(np.fft.fft(PC1))**2
        ps2 = np.abs(np.fft.fft(PC2))**2
        ps3 = np.abs(np.fft.fft(PC3))**2
    
        ps1 = max(ps1)
        ps2 = max(ps2)
        ps3 = max(ps3)
      
        
        Source = np.array([])
        #choosing the source signal to be used for fourier analysis based on PSD peak
        if(ps1>=ps2 and ps1>=ps3):
            Source = PC1
        elif ps2>=ps1 and ps2>=ps3:
            Source = PC2
        else:
            Source = PC3
        
        #bandpass filtering the chosen Source signal
        Source = np.convolve(Source,bandpass_filter)
        Source = Source[50:]
      
        #performing fourier analysis
        w = np.fft.fft(Source)
        freqs = np.fft.fftfreq(len(w))
        idx = np.argmax(np.abs(w))
        freq = freqs[idx]
        freq_in_hertz = abs(freq * fps)
        bpm = int(freq_in_hertz*60)
        
        
        print("bpm = "+str(bpm))
        return bpm
        
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
         
        
        
        
        
        
        
        
        
        
        
        
        
