import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load trained model
MODEL_PATH = 'models/resnet50_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels (same order as training)
classes = ['AHB', 'COVID', 'HMI', 'MI', 'Normal']


def predict_image(img_path):
    """
    Predict class for a single ECG image
    """

    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)

    # Get class index
    class_index = np.argmax(predictions)

    # Confidence
    confidence = np.max(predictions)

    print(f"Predicted Class: {classes[class_index]}")
    print(f"Confidence: {confidence:.4f}")


# Example usage
if __name__ == "__main__":
    predict_image("test.jpg")
