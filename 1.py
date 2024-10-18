import os
import cv2
from deepface import DeepFace

# Ensure CUDA is not used
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Analyze the frame for emotion
    response = DeepFace.analyze(frame, actions=("emotion",), enforce_detection=False)
    print(response)

    # Check if "dominant_emotion" key exists in the response
    dominant_emotion = response[0]['dominant_emotion'] if response and isinstance(response, list) and 'dominant_emotion' in response[0] else None

    # Draw rectangles around detected faces and annotate with the detected emotion
    for (x, y, w, h) in faces:
        if dominant_emotion:
            cv2.putText(frame, text=dominant_emotion, org=(x, y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

    # Show the frame
    cv2.imshow("Facial Emotion Analysis", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
