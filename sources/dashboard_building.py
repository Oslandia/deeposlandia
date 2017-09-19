# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, hamming_loss

import utils

def confusion_matrix_by_label(y_true, y_predicted, index):
    """Return the confusion matrix according to true and predicted labels, for
    the label of index 'index'; the coefficient are ordered as [[true_negative,
    false_positive],[false_negative, true_positive]] (cf sklearn API)

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n, m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than  y_true
    index: integer
        Label to study (between 0 and nb_labels-1)
    
    """
    return confusion_matrix(y_true[:,index], y_predicted[:,index], labels=[0, 1])

def accuracy_by_label(y_true, y_predicted, index):
    """Return the accuracy according to true and predicted labels, for
    the label of index 'index'; the accuracy is equal to (true_positive+true_negative) / nb_observations (cf sklearn API)

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n, m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than  y_true
    index: integer
        Label to study (between 0 and nb_labels-1)
    
    """
    return accuracy_score(y_true[:,index], y_predicted[:,index])

def precision_by_label(y_true, y_predicted, index):
    """Return the precision according to true and predicted labels, for
    the label of index 'index'; the precision is equal to (true_positive+false_positive) / nb_observations (cf sklearn API)

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n, m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than  y_true
    index: integer
        Label to study (between 0 and nb_labels-1)
    
    """
    [_, fp], [_, tp] = confusion_matrix_by_label(y_true, y_predicted, index)
    if fp + tp == 0:
        return 0.0
    return precision_score(y_true[:,index], y_predicted[:,index])

def recall_by_label(y_true, y_predicted, index):
    """Return the recall according to true and predicted labels, for
    the label of index 'index'; the recall is equal to true_positive / (true_positive+false_negative) (cf sklearn API)

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n, m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than  y_true
    index: integer
        Label to study (between 0 and nb_labels-1)
    
    """
    [_, _], [fn, tp] = confusion_matrix_by_label(y_true, y_predicted, index)
    if fn + tp == 0:
        return 0.0
    return recall_score(y_true[:,index], y_predicted[:,index])

def dashboard_by_label(y_true, y_predicted, index):
    """Return a mini-dashboard for each label, i.e. accuracy, precision and
    recall scores in a list format

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n, m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than  y_true
    index: integer
        Label to study (between 0 and nb_labels-1)
    
    """
    return [accuracy_by_label(y_true, y_predicted, index),
            precision_by_label(y_true, y_predicted, index),
            recall_by_label(y_true, y_predicted, index)]

def dashboard_building(y_true, y_predicted):
    """Compute a whole set of metrics to characterize a model accuracy,
    according to a batch of individuals

    Parameters
    ----------
    y_true: list of lists
        True labels for the current batch, format [n,m] with n the number of
    individuals in the batch and m the number of labels
    y_predicted: list of lists
        Labels predicted by the model, same format than y_true
    
    """
    dashboard = []
    hammingloss = hamming_loss(utils.unnest(y_true), utils.unnest(y_predicted))
    dashboard.append(hammingloss)
    total_accuracy = accuracy_score(utils.unnest(y_true),
                                    utils.unnest(y_predicted))
    dashboard.append(total_accuracy)
    total_precision = precision_score(utils.unnest(y_true),
                                      utils.unnest(y_predicted))
    dashboard.append(total_precision)
    total_recall = recall_score(utils.unnest(y_true), utils.unnest(y_predicted))
    dashboard.append(total_recall)
    total_fmeasure = f1_score(utils.unnest(y_true), utils.unnest(y_predicted))
    dashboard.append(total_fmeasure)
    for label in range(len(y_true[0])):
        dashboard = dashboard + dashboard_by_label(y_true, y_predicted, label)
    return dashboard
