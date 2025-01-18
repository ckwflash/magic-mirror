import cv2
from deepface import DeepFace
from collections import deque
import time
import threading
import textwrap
from llm import generate

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize sliding window and timing variables
emotion_window = deque(maxlen=150)  # Store emotions detected over the last 30 frames (~1 second)
last_update_time = time.time()
main_emotion = "neutral"
quote = ''
lock = threading.Lock()  # Lock to manage thread-safe access to the sliding window

def analyze_face(face_roi):
    """
    Analyze the given face ROI using DeepFace in a separate thread.
    Updates the emotion_window with the detected emotion and returns the emotion and age to be drawn.
    """
    global emotion_window
    try:
        # Resize the face ROI for faster computation
        small_face_roi = cv2.resize(face_roi, (64, 64))

        # Perform DeepFace analysis
        analysis = DeepFace.analyze(small_face_roi, actions=['emotion', 'age'], enforce_detection=False)[0]
        emotion = analysis.get('dominant_emotion', 'neutral')
        age = analysis.get('age', 'Unknown')
        
        # Thread-safe update of the sliding window
        with lock:
            emotion_window.append(emotion)

        return emotion, age  # Return both emotion and age

    except Exception as e:
        print(f"Error analyzing face: {e}")
        return "neutral", 0  # Return "neutral" and 0 if there's an error

# Start capturing video
cap = cv2.VideoCapture(0)
width = 1280  # Desired width of the frame
height = 720  # Desired height of the frame

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cv2.namedWindow('Magic Mirror', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Magic Mirror', width, height)  # Set the window to 1280x720 size

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video frame.")
        break

    # Flip the frame for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotions_to_display = []  # List to store the emotions and ages for each face

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = frame[y:y + h, x:x + w]

        # Start a new thread for emotion and age analysis and collect data
        emotion, current_age = analyze_face(face_roi)
        emotions_to_display.append((emotion, current_age, x, y))  # Store the emotion, age, and position for drawing

        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw the emotions and ages for each face after processing all faces
    for emotion, current_age, x, y in emotions_to_display:
        # Draw emotion text
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (52, 189, 252), 2)
        # Draw age text
        cv2.putText(frame, f"Age: {current_age}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ((52, 189, 252)), 2)

    # Determine dominant emotion every 5 seconds
    if time.time() - last_update_time >= 5:
        with lock:
            if emotion_window:
                # Filter out "neutral" unless it's the only emotion
                filtered_emotions = [e for e in emotion_window if e != "neutral"]
                if filtered_emotions:
                    # Calculate the dominant emotion excluding "neutral"
                    main_emotion = max(set(filtered_emotions), key=filtered_emotions.count)
                else:
                    # If only "neutral" is present, use neutral
                    main_emotion = "neutral"
                quote = generate(main_emotion, current_age)
            last_update_time = time.time()

    # Wrap and display the quote text
    y_offset = 40  # Initial y-coordinate for text rendering
    for line in textwrap.wrap(quote, width=55):  # Adjust width as needed
        cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (0, 255, 0), 2)  
        y_offset += 30  # Adjust spacing between lines

    # Show the frame
    cv2.imshow('Magic Mirror', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
