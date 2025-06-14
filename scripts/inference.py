import tensorflow as tf
import numpy as np
import cv2
import os

# Load the trained model
model = tf.keras.models.load_model('models/cnn_model.h5')

# Class labels
class_labels = sorted(os.listdir('data/gesture_videos/asl_alphabet_train'))

# Image size
IMAGE_SIZE = (64, 64)

# Predict function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]
    class_idx = np.argmax(pred)
    confidence = pred[class_idx]

    print(f"Predicted: {class_labels[class_idx]} (Confidence: {confidence:.2f})")

# Example usage
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    predict_image(img_path)
