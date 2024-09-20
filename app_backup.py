import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Directory for uploaded videos
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Define the 3D CNN Model for Deepfake Detection
def create_3d_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load the CNN model
input_shape = (30, 468, 2, 1)
cnn_model = create_3d_cnn_model(input_shape)

# Function to extract landmarks from a frame
def extract_landmarks(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            x = landmark.x * frame.shape[1]
            y = landmark.y * frame.shape[0]
            landmarks.append((x, y))
        return np.array(landmarks)
    return None

# Process a video to extract landmark sequences
def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            sequence.append(landmarks.flatten())
            frames.append(frame)

        if len(sequence) >= max_frames:
            break

    cap.release()

    if len(sequence) > 0:
        sequence_padded = pad_sequences([sequence], maxlen=max_frames, padding='post', truncating='post', dtype='float32')[0]
        sequence_padded = np.array(sequence_padded).reshape(1, max_frames, 468, 2, 1)  # Adjust shape for Conv3D
        return sequence_padded, frames
    return None, None

# API route to detect deepfake
@app.route('/detect', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided.'}), 400

    video = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
    video.save(video_path)

    # Process video and predict deepfake probability
    sequence, frames = process_video(video_path)

    if sequence is not None:
        prediction = cnn_model.predict(sequence)
        is_deepfake = prediction[0][0] > 0.5

        return jsonify({
            'deepfake': bool(is_deepfake),
            'confidence': float(prediction[0][0])
        })
    else:
        return jsonify({'error': 'Failed to process video.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
