import numpy as np
from sklearn.metrics import (
    log_loss, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def round_probabilities(probabilities, threshold):

    """
    Round probabilities to labels based on the given threshold

    Parameters
    ----------
    probabilities : numpy.ndarray of shape (n_samples)
        Predicted probabilities

    threshold: float
        Rounding threshold

    Returns
    -------
    labels : numpy.ndarray of shape (n_samples)
        Rounded probabilities
    """

    labels = np.zeros_like(probabilities, dtype=np.uint8)
    labels[probabilities >= threshold] = 1

    return labels


def specificity_score(y_true, y_pred):

    """
    Calculate specificity score (true-negative rate) of predicted labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted labels

    Returns
    -------
    score: float
        Specificity score between 0 and 1
    """

    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    score = tn / (tn + fp)

    return score


def partial_auc(y_true, y_pred, min_tpr=0.80):

    """
    Calculate partial AUC of predicted probabilities

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    Returns
    -------
    partial_auc_score: float
        Partial AUC score between 0 and 1 - min_tpr
    """

    y_true = np.abs(y_true - 1)
    y_pred = 1 - y_pred
    max_fpr = np.abs(1 - min_tpr)
    partial_auc_scaled = roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    partial_auc_score = 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)

    return partial_auc_score


def classification_scores(y_true, y_pred):

    """
    Calculate binary classification metrics on predicted probabilities and labels

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    Returns
    -------
    scores: dict
        Dictionary of classification scores
    """

    scores = {
        'log_loss': log_loss(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred),
        'partial_auc': partial_auc(y_true, y_pred),
    }

    return scores


def classification_curves(y_true, y_pred):

    """
    Calculate binary classification curves on predicted probabilities

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    Returns
    -------
    curves: dict
        Dictionary of classification curves
    """

    curves = {
        'roc': roc_curve(y_true, y_pred),
        'pr': precision_recall_curve(y_true, y_pred),
    }

    return curves
