from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import torch
from PIL import Image
import numpy as np
import matplotlib
import librosa
import librosa.display
import matplotlib.pyplot as plt
from torchvision import transforms
import tempfile
import subprocess
import os

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# ==== LOAD MODEL & LABELS ====
MODEL_PATH = "model/efficientnet_b0_melspec.pth"
LABELS_PATH = "model/labels.txt"  # one label per line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

from torchvision.models import efficientnet_b0
model = efficientnet_b0()
num_classes = 10
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# Load class labels
with open(LABELS_PATH) as f:
    idx_to_label = [line.strip() for line in f]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def audio_to_melspectrogram_image(audio_bytes):
    # 1. Load audio
    print("Loading audio...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
        temp_in.write(audio_bytes)
        temp_in.flush()
        temp_in_path = temp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            temp_wav_path = temp_wav.name
    print(f"Temporary files created: {temp_in_path}, {temp_wav_path}")
    print("Subprocess running...")
    subprocess.run(['ffmpeg', '-y', '-i', temp_in_path, temp_wav_path], check=True)

    print(f"Converted audio saved to {temp_wav_path}")
    y, sr = librosa.load(temp_wav_path, sr=None, mono=True)
    print("Audio loaded successfully.")
    # Use only the first 10 seconds
    target_length = 10 * sr
    if len(y) > target_length:
        y = y[:target_length]
    elif len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    # 2. Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, window='hann', hop_length=512)
    S_db = librosa.power_to_db(S, ref=np.max)
    # 3. Plot and save to buffer (simulate your PNG image pipeline)
    fig = plt.figure(figsize=(4.32, 2.88), dpi=100)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    librosa.display.specshow(S_db, sr=sr, x_axis=None, y_axis=None, cmap="magma")
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    # 4. Open as PIL image
    image = Image.open(buf).convert("RGB")

    os.remove(temp_in_path)
    os.remove(temp_wav_path)

    return image

# ==== FASTAPI APP ====
app = FastAPI(title="Music Genre Classifier")

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        image = audio_to_melspectrogram_image(audio_bytes)
        tensor = transform(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()
            label = idx_to_label[pred_idx]
        return JSONResponse({
            "genre": label,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        print(f"An error has occured: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/")
def root():
    return {"message": "Music genre classifier is running!"}
