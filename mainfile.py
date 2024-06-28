import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the video
video_path = 'match1.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize variables for feature extraction
frame_count = 0
player_positions = []

# Process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Dummy player detection (for demonstration purposes)
    # Replace with actual player detection and tracking logic
    player_pos = (np.random.randint(0, frame.shape[1]), np.random.randint(0, frame.shape[0]))
    player_positions.append((frame_count, player_pos[0], player_pos[1]))

cap.release()

# Convert to DataFrame
df = pd.DataFrame(player_positions, columns=['frame', 'x', 'y'])

# Add labels for demonstration purposes (e.g., 0 for standing, 1 for running)
df['label'] = np.random.randint(0, 2, df.shape[0])

# Prepare data for training
X = df[['frame', 'x', 'y']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')