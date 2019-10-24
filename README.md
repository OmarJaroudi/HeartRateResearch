<h1>Heart Rate Research</h1>
<p>The goal of this repo is to mesure heart rate based on variation of color in the forehead reigon.</p>
<h2>Summary</h2>
<p>We start by tracking the forehead and extracting the RGB pixels at every frame and adding the mean of the red component green component and blue component to the buffer. Then we detrend and normalize the data in the input buffer after which we perform PCA. </p>
<p>Once we finish the PCA step we compute the power spectral density and take the highest peak then perform FFT to get the final heart rate reading.</p>
<img src = "HR_Detection_Flow.png" alt = "research summary"/>
