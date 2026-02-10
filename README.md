# Gesture Based Application Launcher

A real-time hand gesture recognition system built with Python, OpenCV, MediaPipe, and TensorFlow.  
It can detect hand gestures and perform automated actions like opening applications or controlling the system.

---

## Features
- Detects multiple hand gestures in real-time
- Launch apps like Chrome, Calculator, Notepad,Youtube
- Easy to extend with new gestures
- Supports custom datasets collected using MediaPipe
- GUI support planned with Tkinter
- Lightweight and fast for real-time use

---

## Dataset Collection

**Create a own handgesture database using MediaPipe**

1. Capture 62 hand landmark coordinates.
2. Organize your data in folders by gesture classes.
3. save hand landmark coordinates in a CSV.
4. Collect at least 100–200 samples per gesture for better accuracy.
5. Train the TensorFlow/Keras model with your dataset.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RISHNIKA1/HandGesture_Project.git

2. Navigate the project folder:
   cd HandGesture_Project
   
4. Create a virtual environment:
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows

## pip install -r requirements.txt

## usage
   Run the main script to start hand gesture recognition:
   python main.py
   Perform gestures in front of your webcam to trigger actions.
   If you have a custom dataset and trained model, replace models/gesture_model.h5 with your trained model.
   
## Folder Structure

HandGesture_Project/
├── data/               # custom datasets (images and landmark CSVs)
├── src/                # source code files
├── models/             # trained ML models (models.h5)
├── .venv/              # virtual environment
├── requirements.txt    # project dependencies
├── app.py              # main executable script
|--reports/plotts       # save accuracy and loss plots
└── README.md           # project description

## Author
Rishnika M

