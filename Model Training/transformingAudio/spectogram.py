import librosa.display
import numpy as np
import matplotlib.pyplot as plt


audio_path = r'''Data\genres_original\blues\blues.00000.wav'''
y, sr = librosa.load(audio_path, sr=None)  # load the audio path with the original sampling rate,
# sr=None means no resampling


D = librosa.stft(y)  # compute the short time fourier transform of y and assign it to D

S_db = librosa.amplitude_to_db(np.abs(D)) # convert the amplitude of the STFT to decibels

# 4) Plot the spectrogram
plt.figure()
librosa.display.specshow(S_db, sr=22500, x_axis='time', y_axis='hz', cmap='inferno')
plt.title('Spectrogram (dB)')
plt.colorbar(format="%+2.f dB")
plt.show() # plot the spectrogram
