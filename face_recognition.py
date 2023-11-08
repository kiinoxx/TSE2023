import cv2
import numpy as np
import sys
import os

# function that loads the face data from the 'faces' directory
def load_face_data():
    faces = []
    labels = []
    names = []

    # List the directories in the faces directory (each directory contains a person's images)
    for person_id, person_name in enumerate(os.listdir('faces')):
        # Append the person's name to the names list
        names.append(person_name)

        # List the image files in the person's directory (each image file is one training example)
        for image_name in os.listdir(f'faces/{person_name}'):
            # Load the image in grayscale
            image = cv2.imread(f'faces/{person_name}/{image_name}', cv2.IMREAD_GRAYSCALE)

            # Append the image and label to the lists image = training example, label = person id
            faces.append(image)
            labels.append(person_id)

    return faces, labels, names

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize the 'names' list
names = []

# Check if the 'train' argument was provided
if 'train' in sys.argv:
    # Load the training data
    faces, labels, names = load_face_data()

    # Train the face recognizer
    recognizer.train(faces, np.array(labels))

    # Save the trained model
    recognizer.write('trainer.yml')
else:
    # Load the trained model
    recognizer.read('trainer.yml')

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over the face detections
    for (x, y, w, h) in faces:
        # Predict the ID of the face
        id, _ = recognizer.predict(gray[y:y+h, x:x+w])

        if id < len(names):
            # get the name corresponding to the ID
            name = names[id]

        else:
            # If no match found, use 'Unknown'
            name = 'Unknown'


        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the name
        cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()