# deepfake-webapp
# 🕵️ Deepfake Detection WebApp

A Flask-based web application that detects deepfake videos using a custom-built 3D Convolutional Neural Network (CNN) and facial landmark tracking via MediaPipe.

## 🚀 Features

- 🎥 Upload short video clips for analysis
- 🧠 Deepfake detection using 3D CNN over 30-frame facial landmark sequences
- 👁️ Heatmap visualization of suspected tampered regions
- 🧩 Reconstructed face with landmark connections
- 📩 Feedback form with email notifications using Flask-Mail

---




## 📷 How It Works

1. **Facial Landmark Extraction**:
   - Uses [MediaPipe FaceMesh](https://google.github.io/mediapipe/solutions/face_mesh.html) to extract 468 keypoints from each frame.

2. **Model Input**:
   - 30 frames are converted into shape: `(30, 468, 2, 1)` and fed into the 3D CNN.

3. **Classification**:
   - Model outputs a probability (sigmoid) indicating whether the video is a deepfake.

4. **Visualization**:
   - Frame heatmaps, probability graph, and reconstructed face landmarks are displayed via `matplotlib`.

---

## 🧠 Model Architecture

- Built from scratch using `Conv3D`, `MaxPooling3D`, `Flatten`, and `Dense` layers
- Binary classifier (`sigmoid`) trained on real and deepfake sequences
- Optimized using `Adam` and `binary_crossentropy` loss

---

## 📨 Feedback Feature

The app includes a user feedback form (`/feedback`) that sends emails directly to the project owner's Gmail using Flask-Mail.


## 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/drcdebtosh/deepfake-webapp.git
   cd deepfake-webapp
   ```

2. **Install Requirements**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add a `.env` File** with:
   ```
   MAIL_USERNAME=your_email@gmail.com
   MAIL_PASSWORD=your_app_password
   ```

4. **Model Weights**
   - Place your pre-trained weights file in `model/deepfake_model.h5`.
   - Ensure `app.py` loads these weights at startup:
     ```python
     cnn_model.load_weights('model/deepfake_model.h5')
     ```

5. **Run the App**
   ```bash
   python app.py
   ```

---

## 🛡️ Dependencies

- Flask
- TensorFlow
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- Flask-Mail
- python-dotenv

Add these to `requirements.txt`:
```
Flask
tensorflow
mediapipe
opencv-python
numpy
matplotlib
Flask-Mail
python-dotenv
```

---

## 📊 Future Improvements

- Real-time webcam-based detection
- Multi-face support
- Deployment via Render/Heroku/Upstash
- Result logging and database support (SQLite, Firebase, etc.)
- Progress bar or asynchronous processing for large videos

---

## 📬 Contact

📧 Email: [debatoshofficial85@gmail.com](mailto:debatoshofficial85@gmail.com)  
💡 Author: Debatosh Roychowdhury (DRC)

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
