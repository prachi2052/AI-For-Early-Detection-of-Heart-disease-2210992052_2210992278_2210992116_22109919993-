import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = tf.keras.models.load_model('models/resnet50_model.h5')


def evaluate_model(test_data, class_names):
    """
    Evaluate model using confusion matrix and classification report
    """

    # Predictions
    predictions = model.predict(test_data)
    y_pred = np.argmax(predictions, axis=1)

    y_true = test_data.classes

    # Classification report
    print("\n📄 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
