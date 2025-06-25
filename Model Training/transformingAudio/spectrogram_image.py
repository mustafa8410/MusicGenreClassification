import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

datasetPath = r'Data\genres_original'
outputPath = r'spectogram_images'
failedProcesses = []

chunkLength = 10  # seconds
totalLength = 30  # seconds

for genre in os.listdir(datasetPath):
    genrePath = os.path.join(datasetPath, genre)
    if os.path.isdir(genrePath):
        genreOutputPath = os.path.normpath(os.path.join(outputPath, genre))
        os.makedirs(genreOutputPath, exist_ok=True)
        for song in os.listdir(genrePath):
            songPath = os.path.normpath(os.path.join(genrePath, song))
            if os.path.isfile(songPath) and song.endswith('.wav'):
                try:
                    y, sr = librosa.load(songPath, sr=None)
                    samplesPerChunk = sr * chunkLength
                    for i in range(3):
                        startSample = i * samplesPerChunk
                        endSample = (i + 1) * samplesPerChunk
                        yChunk = y[startSample:endSample]

                        D = librosa.stft(yChunk)
                        S_db = librosa.amplitude_to_db(np.abs(D))

                        fig = plt.figure(figsize=(4.32, 2.88), dpi=100)
                        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', cmap='inferno')
                        plt.axis('off')

                        songOutputPath = os.path.join(genreOutputPath, song.replace('.wav', f'_{i + 1}.png'))
                        plt.savefig(songOutputPath, bbox_inches='tight', pad_inches=0)
                        plt.close(fig)
                        print(f"Saved spectrogram for {song} chunk {i + 1}")
                    print(f"Processed {song} in {genre}")
                except Exception as e:
                    print(f"Error processing {song} in {genre}: {e}")
                    failedProcesses.append((genre, song))

print("Processing complete.")
if failedProcesses:
    print("Failed to process the following files:")
    for genre, song in failedProcesses:
        print(f"{genre}/{song}")
