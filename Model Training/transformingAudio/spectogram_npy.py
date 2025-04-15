import os
import librosa
import numpy as np

datasetPath = r'Data\genres_original'
outputPath = r'spectogram_npy'
failedProcesses = []

chunkLength = 10
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
                        yChunk = y[i * samplesPerChunk: (i + 1) * samplesPerChunk]
                        D = librosa.stft(yChunk)
                        S_db = librosa.amplitude_to_db(np.abs(D))

                        outputFile = os.path.join(genreOutputPath, song.replace('.wav', f'_{i + 1}.npy'))
                        np.save(outputFile, S_db)
                        print(f"Saved spectrogram as npy for {song} chunk {i + 1}")
                    print(f"Processed {song} in {genre}")
                except Exception as e:
                    print(f"Error processing {song} in {genre}: {e}")
                    failedProcesses.append((genre, song))

print("Processing complete.")
if failedProcesses:
    print("Failed to process the following files:")
    for genre, song in failedProcesses:
        print(f"{genre}/{song}")