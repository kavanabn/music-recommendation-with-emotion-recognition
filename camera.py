import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

# Load face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load emotion model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

emotion_dict = {0:"Angry",1:"Disgusted",2:"Fearful",3:"Happy",4:"Neutral",5:"Sad",6:"Surprised"}
music_dist={0:"songs/angry.csv",1:"songs/disgusted.csv",2:"songs/fearful.csv",3:"songs/happy.csv",4:"songs/neutral.csv",5:"songs/sad.csv",6:"songs/surprised.csv"}

# Persistent camera object
camera = cv2.VideoCapture(0)
show_text = [0]

def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name','Album','Artist']]
    df = df.head(15)
    return df

def get_frame():
    success, frame = camera.read()
    if not success:
        # Return a black frame with error
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Cannot access camera", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), music_rec(), emotion_dict[show_text[0]]
    
    frame = cv2.resize(frame, (600, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Improve face detection stability with better parameters
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,  # Reduced from 1.3 for more stable detection
        minNeighbors=5,   # Increased from 5 for better accuracy
        minSize=(30, 30), # Minimum face size
        maxSize=(300, 300) # Maximum face size
    )
    
    detected = False
    for (x, y, w, h) in faces:
        detected = True
        
        # Draw a more prominent face rectangle
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 3)
        
        # Add face detection confidence indicator
        cv2.putText(frame, "Face Detected", (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Process emotion detection
        roi_gray_frame = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        
        # Only update emotion if confidence is high enough
        if confidence > 0.5:  # 50% confidence threshold
            show_text[0] = maxindex
            emotion_name = emotion_dict[maxindex]
            
            # Create emotion label frame with better styling
            label_text = f"{emotion_name} ({confidence:.1%})"
            
            # Get text size for proper background
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            
            # Draw background rectangle for emotion label
            cv2.rectangle(frame, 
                         (x+20, y-60-text_height-10), 
                         (x+20+text_width+20, y-60+baseline+10), 
                         (0, 0, 0), -1)
            
            # Draw border for emotion label
            cv2.rectangle(frame, 
                         (x+20, y-60-text_height-10), 
                         (x+20+text_width+20, y-60+baseline+10), 
                         (255, 255, 255), 2)
            
            # Draw emotion text
            cv2.putText(frame, label_text, (x+30, y-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Add emotion emoji
            emotion_emojis = {
                "Angry": "😠", "Disgusted": "🤢", "Fearful": "😨", 
                "Happy": "😊", "Neutral": "😐", "Sad": "😢", "Surprised": "😮"
            }
            emoji = emotion_emojis.get(emotion_name, "🎭")
            cv2.putText(frame, emoji, (x+w+10, y+h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        else:
            # Low confidence - show uncertain status
            cv2.putText(frame, "Analyzing...", (x+20, y-60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    
    if not detected:
        # Enhanced "No face detected" message
        cv2.putText(frame, "No face detected", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Please position your face in the camera", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg.tobytes(), music_rec(), emotion_dict[show_text[0]]
