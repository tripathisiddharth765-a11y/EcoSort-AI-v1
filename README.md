# ♻️ EcoSort AI

A real-time waste classification application built using **PyTorch** and **Streamlit**.

EcoSort AI classifies waste into multiple categories using a custom Convolutional Neural Network (CNN). Users can either upload an image or use their webcam for real-time waste classification.

---

## 📷 Demo

### Image Upload
<img width="1411" height="977" alt="image" src="https://github.com/user-attachments/assets/4107753c-00a5-47b8-a6c1-564c4110bbda" />


### Live Webcam Prediction
![Uploading image.png…]()


---

## 🚀 Features

- Image Upload Classification
- Real-Time Webcam Classification
- Confidence Score Display
- PyTorch CNN Model
- Streamlit Web Interface
- Fast Inference Pipeline

---

## 🧠 Model

The model is a custom Convolutional Neural Network built using PyTorch.

### Architecture

Input Image
↓
Conv2D + ReLU + MaxPool
↓
Conv2D + ReLU + MaxPool
↓
Conv2D + ReLU + MaxPool
↓
Flatten
↓
Fully Connected Layer
↓
Softmax Output

---

## 📂 Dataset

This project was trained on publicly available waste classification datasets from Kaggle.

Classes include:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

---

## 🛠️ Tech Stack

- Python
- PyTorch
- Torchvision
- OpenCV
- Streamlit
- Streamlit-WebRTC
- PIL
- NumPy

---

## 📦 Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/EcoSort-AI.git
```

Move into the project

```bash
cd EcoSort-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run APP.py
```

---

## 📸 Screenshots

(Add screenshots here)

---

## 📁 Project Structure

```
EcoSort-AI
│
├── APP.py
├── predict.py
├── dataset.py
├── train.py
├── model.pth
├── requirements.txt
├── README.md
└── assets/
```

---

## 🎯 Future Improvements

- YOLO-based Multi-Object Detection
- Waste Detection Bounding Boxes
- Mobile Deployment
- Edge AI Deployment (Jetson Nano / Raspberry Pi)
- Smart Dustbin Integration
- Recycling Suggestions
- Detection History Dashboard

---

## 📚 What I Learned

Through this project I learned:

- Building CNNs using PyTorch
- Image preprocessing and augmentation
- Model training and evaluation
- Real-time computer vision
- Streamlit application development
- Webcam integration using Streamlit-WebRTC
- Deploying AI applications
- Debugging real-world inference pipelines

---

## 📜 License

This project is intended for learning and educational purposes.

---

## 👨‍💻 Author

Siddharth Tripathi

If you found this project interesting, feel free to ⭐ the repository.
