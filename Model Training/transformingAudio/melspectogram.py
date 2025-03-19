import librosa
import librosa.feature
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

audio_path = r'''Data\genres_original\blues\blues.00000.wav'''
y, sr = librosa.load(audio_path, sr=None)
melspectogram = librosa.feature.melspectrogram(y=y, sr=sr)
melspectogram_db = librosa.power_to_db(melspectogram)

plt.figure(figsize=(10, 4))
librosa.display.specshow(melspectogram_db, sr=sr, x_axis="time", y_axis="mel")
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
# plt.xlabel("Time (s)")
# plt.ylabel("Mel Frequency")
plt.show()

