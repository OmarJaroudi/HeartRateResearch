<h1>Heart Rate Research</h1>
<p>The goal of this repo is to mesure heart rate based on variation of color in the forehead reigon.</p>
<h2>Summary</h2>
<p>We start by tracking the forehead and extracting the RGB pixels at every frame and adding the mean of the red component green component and blue component to the buffer. </p>
<p> normalize the last 3 seconds of each signal </p>
<p> bandpass filter each of the 3 signals to capture realistic human heart rate frequencies (128 point hamming window)  </p>
<p> Perform PCA </p>
<p> Compute the Power spectral density of each principal component </p>
<p> The one with the highest peak is the raw PPG signal </p>
<p> Note: if the peaks of the PSD are within 1000 units of each other, we select the signal based on the previous heart rate reading </p>

<img src = "HR_Detection_Flow.png" alt = "research summary"/>
