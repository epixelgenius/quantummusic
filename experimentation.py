import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load a vocal recording
y, sr = librosa.load('/Users/rishi/Downloads/StarWarsMainTheme.mp3')

# Feature extraction
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
rms = librosa.feature.rms(y=y)
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

# Compile features into a single vector
features = np.hstack([
    np.mean(mfcc, axis=1),  # Mean MFCCs
    np.mean(spectral_centroid),  # Mean Spectral Centroid
    np.mean(rms),  # Mean Loudness
    tempo  # Tempo
])

# Create an emotion label for the vocal file (for training)
emotion_label = 1  # e.g., "happy" = 1, "sad" = 0

# Create a dataset and labels (you'll need more vocal samples)
X = [features]  # Feature matrix
y = [emotion_label]  # Emotion labels

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Predict emotions on the test set
y_pred = clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Emotion Classification Accuracy: {accuracy}")