from flask import Flask, request, jsonify
import os
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Function to extract facial landmarks using MediaPipe
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

# Function to process a video and extract landmark sequences
def process_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    frames = []
    probas = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = extract_landmarks(frame)
        if landmarks is not None:
            sequence.append(landmarks.flatten())
            frames.append(frame)

        if len(sequence) >= max_frames:
            sequence_padded = pad_sequences([sequence], maxlen=max_frames, padding='post', truncating='post', dtype='float32')[0]
            sequence_padded = np.array(sequence_padded).reshape(1, max_frames, 468, 2, 1)  # Adjust shape for Conv3D
            probas.append(cnn_model.predict(sequence_padded)[0][0])

            # break

    cap.release()

    if len(sequence) > 0:
        sequence_padded = pad_sequences([sequence], maxlen=max_frames, padding='post', truncating='post', dtype='float32')[0]
        sequence_padded = np.array(sequence_padded).reshape(1, max_frames, 468, 2, 1)
        return sequence_padded, frames, probas
    return None, None, None

# Define the 3D CNN model for deepfake detection
def create_3d_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))

    model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))

    model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))

    model.add(Conv3D(256, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))

    model.add(Conv3D(512, (3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D((1, 2, 2), padding='same'))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid', dtype='float32'))

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Function to reconstruct the original face from landmarks by connecting lines between landmarks
def reconstruct_original_face(landmarks, frame):
    original_frame = frame.copy()

    # Define the connections for the face mesh based on MediaPipe's topology
    connections = [
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172), (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10), # Outer face
        (151, 108), (108, 69), (69, 104), (104, 68), (68, 71), (71, 107), (107, 151), # Left eye
        (336, 296), (296, 334), (334, 293), (293, 300), (300, 383), (383, 374), (374, 380), (380, 252), # Right eye
        # Additional connections can be added for the nose, mouth, etc.
    ]

    # Iterate over connections and draw lines between landmarks
    for connection in connections:
        point1 = landmarks[connection[0]]
        point2 = landmarks[connection[1]]
        cv2.line(original_frame, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), (0, 255, 0), 2)

    return original_frame

########################################


# Initialize the model
input_shape = (30, 468, 2, 1)
cnn_model = create_3d_cnn_model(input_shape)

# Route for uploading video and detecting deepfake
@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        video_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(video_path)

        # Process the video and make prediction
        sequence, frames, probas = process_video(video_path, max_frames=30)
        if sequence is None:
            return jsonify({"error": "Unable to process video or insufficient data"}), 400

        sequence = sequence.reshape(1, 30, 468, 2, 1)  # Adjust shape for Conv3D
        prediction = cnn_model.predict(sequence)
        
        is_deepfake = prediction > 0.5


        response = {
            "deepfake": bool(is_deepfake),
            "probability": float(prediction)
        }

        return jsonify(response)
    
from flask import Flask, render_template



@app.route('/')
def home():
    return render_template('index.html')

# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
