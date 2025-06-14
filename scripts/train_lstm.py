import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Load preprocessed data
X_train = np.load('data/processed/X_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_test = np.load('data/processed/y_test.npy')

# One-hot encode labels
num_classes = len(np.unique(y_train))
y_train_ohe = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_ohe = tf.keras.utils.to_categorical(y_test, num_classes)

# Build LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(40, 63)),
    Dropout(0.3),
    LSTM(64),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint('models/lstm_model.h5', monitor='val_accuracy', save_best_only=True)

# Train
model.fit(X_train, y_train_ohe, epochs=20, batch_size=32, validation_data=(X_test, y_test_ohe), callbacks=[checkpoint])
