import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, precision_recall_curve


def partial_auc(y_true, y_pred, min_tpr=0.80):

    """
    Calculate partial AUC of predicted probabilities

    Parameters
    ----------
    y_true: numpy.ndarray of shape (n_samples)
        Ground truth labels

    y_pred: numpy.ndarray of shape (n_samples)
        Predicted probabilities

    min_tpr: float
        Minimum true-positive rate

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
