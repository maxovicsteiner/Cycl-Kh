# Function of this file is to set up a baseline model (to be compared with future models)
# wav file -> normalize -> onset detection -> fft -> input into transformer (fixed context size) => predict next pulse -> play with ifft

import librosa
import numpy as np

#load
y, sr = librosa.load("sample.wav")
#normalize
y_normalized = y / np.max(np.abs(y))
#onset detection
onsetsframes = librosa.onset.onset_detect(y=y_normalized, sr=sr)

hoplength = 512
#ffts
ffts = []
for i in range(len(onsetsframes) - 1):
    start = onsetsframes[i] * hoplength
    end = onsetsframes[i+1] * hoplength
    
    fft = np.fft.fft(y_normalized[start:end])
    fft_rounded = np.round(fft, decimals=2)
    ffts.append(list(fft_rounded))



with open("data.txt", "w") as f:
    f.write(str(ffts))