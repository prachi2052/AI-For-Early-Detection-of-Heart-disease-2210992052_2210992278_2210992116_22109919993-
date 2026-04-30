import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load model
model = tf.keras.models.load_model('models/resnet50_model.h5')


def plot_roc_curve(test_data, num_classes=5):
    """
    Plot ROC curve for multiclass classification
    """

    # Predictions
    y_pred = model.predict(test_data)

    # True labels
    y_true = test_data.classes

    # Convert to binary format
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    # Plot ROC for each class
    plt.figure()

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

    # Diagonal line
    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Multiclass)')
    plt.legend()
    plt.grid()
    plt.show()