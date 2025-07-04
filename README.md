# Augmented-Facial-Expression-Filter
# 🎭 Emotion-Based Real-Time Face Filter with OpenCV, MediaPipe & FER

Welcome to **Emotion Filter**, an intelligent real-time facial augmentation project that detects human emotions via webcam and overlays expressive AR-style filters based on detected mood. This project combines the power of **computer vision**, **facial landmark tracking**, and **AI-driven emotion detection** to create an engaging and interactive experience.

## ✨ Features

- 🔍 **Real-time Emotion Detection** using the FER (Facial Expression Recognition) deep learning model.
- 🧠 **Facial Landmark Tracking** with MediaPipe for precise overlay placement.
- 🕶️ **Augmented Reality Filters** tailored to each emotion:
  - 😄 *Happy* – Cool sunglasses + smiling mouth overlay
  - 😠 *Angry* – Devil horns + mustache
  - 😲 *Surprised* – Wide surprised glasses + open shocked mouth
  - 😐 *Neutral* – Neutral face overlay (optional)
- 📷 Seamless webcam integration via OpenCV.

## 🧰 Tech Stack

| Technology | Purpose |
|-----------|---------|
| `Python` | Core language |
| `OpenCV` | Webcam access, frame processing, image rendering |
| `MediaPipe` | Facial landmark detection (468 3D points) |
| `FER` | Deep learning-based emotion classifier |
| `NumPy` | Efficient numerical computations and vector operations |

## 🧠 How It Works

1. **Capture Frame:** Live video is captured using OpenCV from the default webcam.
2. **Detect Emotion:** FER processes the RGB frame to determine facial emotions like *happy*, *angry*, or *surprised*.
3. **Extract Landmarks:** MediaPipe FaceMesh identifies key facial points such as eyes, lips, and forehead.
4. **Render Filter:** Based on the detected emotion, a corresponding transparent PNG filter is resized and overlaid on facial landmarks using precise alpha blending.



