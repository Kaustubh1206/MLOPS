import numpy as np
import pandas as pd
import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score


def main():

    # -----------------------------
    # Load trained model
    # -----------------------------
    clf = pickle.load(open('model.pkl', 'rb'))

    # -----------------------------
    # Load test data
    # -----------------------------
    test_data = pd.read_csv('./data/features/test_bow.csv')

    X_test = test_data.iloc[:, 0:-1].values
    y_test = test_data.iloc[:, -1].values

    # -----------------------------
    # Make predictions
    # -----------------------------
    y_pred = clf.predict(X_test)

    # -----------------------------
    # Define positive class
    # -----------------------------
    positive_class = 'happiness'

    # -----------------------------
    # Accuracy
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)

    # -----------------------------
    # Precision & Recall
    # -----------------------------
    precision = precision_score(
        y_test,
        y_pred,
        pos_label=positive_class
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=positive_class
    )

    # -----------------------------
    # AUC Score
    # -----------------------------
    # Get probability of positive class safely
    class_index = list(clf.classes_).index(positive_class)
    y_pred_proba = clf.predict_proba(X_test)[:, class_index]

    auc = roc_auc_score(y_test, y_pred_proba)

    # -----------------------------
    # Save metrics
    # -----------------------------
    metrics_dict = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }

    with open('src/metrics.json', 'w') as file:

        json.dump(metrics_dict, file, indent=4)

    print("Model evaluation completed successfully.")
    print(metrics_dict)


if __name__ == "__main__":
    main()
