import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# Load CSV
df = pd.read_csv('data/processed/gesture_data.csv')

# Separate features and labels
X = df.drop('label', axis=1).values
y = df['label'].values

# Encode labels (text -> int)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label mapping
np.save('data/processed/label_classes.npy', le.classes_)

# Reshape X to (samples, 40, 63) -> 40 frames, 21 keypoints * 3 coords
X = X.reshape((X.shape[0], 40, 63))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Save preprocessed arrays
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

print("✅ Preprocessing complete. Data saved in 'data/processed/'")
