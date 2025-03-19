import librosa.feature
import numpy as np
from matplotlib import pyplot as plt

audio_path = r'''Data\genres_original\blues\blues.00000.wav'''
y, sr = librosa.load(audio_path, sr=None)

chromagram = librosa.feature.chroma_stft(y=y, sr=sr)

plt.figure(figsize=(12, 6))
librosa.display.specshow(chromagram, sr=sr, x_axis="time", y_axis="chroma")
pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
plt.yticks(ticks=range(len(pitch_classes)), labels=pitch_classes)
plt.colorbar()
plt.title("Chromagram")
plt.xlabel("Time (s)")
plt.ylabel("Pitch Class")
plt.show()
