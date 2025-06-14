# 🧠 Real-Time Sign Language Recognition

A full-stack deep learning project to recognize sign language using a webcam in real-time, combining CNN for static gestures and LSTM for dynamic sequences. Built with TensorFlow, MediaPipe, and Streamlit.

## 💡 Features

- 🤏 Static alphabet recognition with CNN (ASL dataset)
- 🔁 Dynamic gesture recognition with LSTM on 3D keypoints
- 🎥 Real-time webcam inference
- 🗣 Voice feedback with `pyttsx3`
- 🌐 Streamlit interface (optional)
- 🛠 Modular scripts: collect, train, infer

## 🗂 Project Structure

sign-language-translator/
├── app/
│ └── streamlit_app.py
├── data/
│ ├── gesture_videos/
│ └── processed/
├── models/
│ └── cnn_model.h5
│ └── lstm_model.h5
├── scripts/
│ ├── collect_data.py
│ ├── preprocess.py
│ ├── train_cnn.py
│ ├── train_lstm.py
│ └── realtime_inference.py
├── requirements.txt
├── .gitignore
└── README.md


## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/sign-language-translator.git
cd sign-language-translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

To train LSTM:
python scripts/train_lstm.py

To run real-time inference with speech:
python scripts/realtime_inference.py

To launch Streamlit UI:
streamlit run app/streamlit_app.py
