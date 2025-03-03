from flask import Flask, render_template, request, Response, redirect, url_for, flash
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64
import os
import threading
from playsound import playsound
import pygame

app = Flask(__name__)
app.secret_key = "secret_key"
model = load_model('mask_detector.h5')

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
video_active = False
lock = threading.Lock()
alarm_active = False

# Function to detect faces and masks
def detect_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    confidence_scores = {}
    global alarm_triggered

    for (x, y, w, h) in faces:
        # Crop the face from the image for mask prediction
        face_roi = frame[y:y + h, x:x + w]
        face_resized = cv2.resize(face_roi, (224, 224))
        face_array = np.expand_dims(face_resized, axis=0) / 255.0
        predictions = model.predict(face_array)[0]

        # Labels for the mask prediction
        labels = ['Mask', 'No Mask']
        confidence_scores = {labels[i]: round(predictions[i] * 100, 2) for i in range(len(labels))}

        # Get the label with the highest confidence
        result_label = labels[np.argmax(predictions)]
        max_confidence = confidence_scores[result_label]

        # Set rectangle color based on the result
        color = (0, 255, 0) if result_label == "Mask" else (0, 0, 255)

        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Add text with the prediction and confidence score
        cv2.putText(frame, f"{result_label}: {max_confidence}%", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if result_label == 'No Mask':
            alarm_triggered = True

    return frame, confidence_scores

# Home route
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/documentation')
def demo():
    return render_template('documentation.html')
# @app.route('/home')
# def home():
#     return render_template('home.html')
# Image upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash('No file uploaded', 'error')
            return redirect(request.url)

        # Read the image and convert it to grayscale for face detection
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

        # Initialize confidence_scores outside of the loop to avoid unbound reference
        confidence_scores = {}

        # Debugging: Print how many faces were detected
        print(f"Faces detected: {len(faces)}")

        # Process each face detected
        for (x, y, w, h) in faces:
            # Crop the face from the image for mask prediction
            face_roi = image[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (224, 224))
            face_array = np.expand_dims(face_resized, axis=0) / 255.0
            predictions = model.predict(face_array)[0]

            # Labels for the mask prediction
            labels = ['Mask', 'No Mask']
            confidence_scores = {labels[i]: round(predictions[i] * 100, 2) for i in range(len(labels))}

            # Get the label with the highest confidence
            result_label = labels[np.argmax(predictions)]
            max_confidence = confidence_scores[result_label]

            # Set rectangle color based on the result
            color = (0, 255, 0) if result_label == "Mask" else (0, 0, 255)

            # Draw rectangle around the face
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)

            # Add text with the prediction and confidence score
            cv2.putText(image, f"{result_label}: {max_confidence}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert the image to base64 to display in HTML
        _, buffer = cv2.imencode('.jpg', image)
        result_image = base64.b64encode(buffer).decode('utf-8')

        return render_template('upload.html', result_image=result_image, results=confidence_scores)

    return render_template('upload.html')


def play_alarm():
    """Play an alarm sound in a loop when triggered."""
    global alarm_active
    pygame.mixer.init()
    pygame.mixer.music.load("static/alert.mp3")
    pygame.mixer.music.play(-1)  # Play in a loop
    while alarm_active:
        pass  # Keep playing until alarm_active is False
    pygame.mixer.music.stop()

@app.route('/start_video')
def start_video():
    """Start the live video feed."""
    global video_active
    with lock:
        video_active = True
    return redirect(url_for('video'))

@app.route('/stop_video')
def stop_video():
    """Stop the live video feed."""
    global video_active
    with lock:
        video_active = False
    return redirect(url_for('video'))

def generate_frames():
    """Generate frames for live video feed."""
    global video_active, alarm_active
    cap = cv2.VideoCapture(0)
    while True:
        with lock:
            if not video_active:
                break

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        face_detected_without_mask = False

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            label, confidence = detect_masks(face)

            # Convert confidence to percentage format
            confidence_percent = confidence * 100
            confidence_text = f"{confidence_percent:.2f}%"

            # Set color and label based on mask detection
            if label == "Mask":
                color = (0, 255, 0)  # Green for with mask
            else:
                color = (0, 0, 255)  # Red for no mask

            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Display label and confidence score
            cv2.putText(frame, f"{label} ({confidence_text})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # If no mask detected, activate alarm
            if label == "No Mask":
                face_detected_without_mask = True

        # If face detected without a mask, trigger alarm
        if face_detected_without_mask and not alarm_active:
            alarm_active = True
            threading.Thread(target=play_alarm, daemon=True).start()

        # If no face without a mask, stop alarm
        elif not face_detected_without_mask and alarm_active:
            alarm_active = False

        # Encode the frame to be sent to the client
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    alarm_active = False

@app.route('/video_feed')
def video_feed():
    """Route for live video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


def detect_masks(face):
    """Detect mask on a single face."""
    resized_face = cv2.resize(face, (224, 224))  # Resize to match the model input
    normalized_face = resized_face / 255.0  # Normalize the pixel values
    input_data = np.expand_dims(normalized_face, axis=0)
    prediction = model.predict(input_data)
    label = "Mask" if prediction[0][0] > 0.5 else "No Mask"
    confidence = prediction[0][0] if label == "Mask" else 1 - prediction[0][0]
    return label, confidence

# Video route for live feed
@app.route('/video')
def video():
    return render_template('video.html')
# Start the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
