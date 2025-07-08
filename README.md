#  Music Genre Classification — Full-Stack Deep Learning Project

This project is a modular, microservice-based music genre classification system with:

* Deep learning model training and performance evaluation (PyTorch)
* Python FastAPI microservice for real-time inference
* Secure Spring Boot web backend with user authentication, prediction history, and browser-based audio recording

---

## Project Structure

```
repo-root/
│
├── Model Training/        # Model training scripts (PyTorch, data preprocessing)
├── ML Microservice/       # Python FastAPI microservice for inference
├── Web/                   # Java Spring Boot backend & Thymeleaf frontend
└── README.md
```

---

## Features

* **Trains and evaluates 5 different CNN architectures** on 3 audio image types (mel spectrogram, spectrogram, chromagram)
* **User-friendly web app:** Registration, login, guest access, audio recording, and prediction history
* **Microservice architecture:** Decouples ML inference from the web backend for maintainability and scalability
* **User dashboard:** Stream/download past predictions, view genre/confidence for each audio
* **Secure:** Password hashing, RESTful APIs, clear documentation

---

## Modules Overview

### 1. Model Training (`Model Training/`)

* Generate spectrogram/chromagram/melspectrogram images from audio
* Train and evaluate CNNs (custom and pretrained, e.g., EfficientNet-B0)
* 5-fold cross-validation with metric logging and model saving

---

### 2. ML Microservice (`ML Microservice/`)

* FastAPI service for genre prediction from audio files
* Handles webm/ogg conversion using ffmpeg; generates melspectrogram images for inference
* Loads your best EfficientNet-B0 `.pth` model
* Returns predicted genre and confidence as JSON

**Requirements:**

* Python 3.12
* [ffmpeg](https://ffmpeg.org/download.html) installed and on system PATH
* The libraries that are imported in the code files

---

### 3. Web App (`Web/`)

* Spring Boot backend (Java 24+) with Thymeleaf frontend
* User authentication (Spring Security), guest mode, audio recording and upload, dashboard/history, and download support
* Communicates with the ML microservice for predictions

**Requirements:**

* Java 24+, Maven, PostgreSQL 17

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/<mustafa>/music-genre-classification-fullstack.git
cd music-genre-classification-fullstack
```

### 3. ML Microservice

* Go to `ML Microservice/`
* Install requirements:
* Make sure ffmpeg is installed and in your system PATH
* Run the service:

  ```bash
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
  ```

### 4. Web App

* Go to `Web/`
* Edit `src/main/resources/application.properties` with your PostgreSQL credentials
* Create a `music_genre_classification` database in PostgreSQL
* Run the app:

  ```bash
  mvn spring-boot:run
  ```
* App will be available at [http://localhost:8080](http://localhost:8080)

> **Note:**
> Both the web app and ML microservice must be running for full functionality.

---

## Features

* Register or continue as guest
* Record or upload audio
* Get real-time genre predictions with confidence score
* View, replay, and download your prediction history (if registered)

---

## Troubleshooting

* **Lombok issues:**
  Remove any `<annotationProcessorPaths>` for Lombok in `pom.xml` and ensure the Lombok plugin is enabled in your IDE.
* **ffmpeg errors:**
  Ensure ffmpeg is installed and `bin` is in system PATH.
* **Database issues:**
  Make sure PostgreSQL is running, and your properties are set correctly.

---

## 📄 License / Credits

* Dataset: [GTZAN Music Genre Collection](http://marsyas.info/downloads/datasets.html)

---



For questions or feedback, you can open an issue!
