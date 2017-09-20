# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

import matplotlib.pyplot as plt
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

def plot_dashboard(dashboard, plot_filename):
    """Plot the model dashboard after a round of training; the training state
    has been stored into a `.csv` file, `dashboard` is supposed to be a pandas
    DataFrame that contains the file information

    Parameters
    ----------
    dashboard: pandas.DataFrame
        model dashboard, that contains one row for each training epoch; the
    different features are `loss`, `accuracy`, `precision` or `recall` (among
    others), the last three ones being detailed for each label (under the
    format `<metric>_label<id>`)
    plot_filename: object
        string designing the name of the file in which the plot has to be saved
    
    """
    fig = plt.figure(figsize=(16, 24))
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.5, 0.5)
    for i in range(66):
        a = plt.subplot2grid((11, 12), (int(i/6), 6+i%6))
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+8], 'r-')
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+9], 'b-')
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+10], 'g-')
        a.set_title("Label "+str(i), size=6)
        if int(i/6) < 10:
            a.xaxis.set_visible(False)
        if i % 6 > 0:
            a.yaxis.set_visible(False)
    a = plt.subplot2grid((11, 12), (0, 0), rowspan=2, colspan=6)
    a.plot(dashboard.epoch, dashboard.loss, 'k-', lw=1.5)
    a.set_yscale('log')
    a.set_xlabel('epoch')
    a.set_ylabel('Standard loss')
    a = plt.subplot2grid((11, 12), (2, 0), rowspan=2, colspan=6)
    a.plot(dashboard.epoch, dashboard.bpmll_loss, 'k-', lw=1.5)
    a.set_xlabel('epoch')
    a.set_ylabel('BPMLL loss')
    a = plt.subplot2grid((11, 12), (4, 0), rowspan=2, colspan=6)
    a.plot(dashboard.epoch, dashboard.hamming_loss, 'k-', lw=1.5)
    a.set_xlabel('epoch')
    a.set_ylabel('Hamming loss')
    a = plt.subplot2grid((11, 12), (6, 0), rowspan=5, colspan=6)
    a.plot(dashboard.epoch, dashboard.accuracy, 'r-')
    a.plot(dashboard.epoch, dashboard.precision, 'b-')
    a.plot(dashboard.epoch, dashboard.recall, 'g-')
    a.plot(dashboard.epoch, dashboard.F_measure, 'y-')
    a.set_xlabel('epoch')
    a.set_ylabel('model evaluation')
    a.legend(['accuracy', 'precision' ,'recall', 'F-measure'], prop={'size':12})
    fig.savefig(plot_filename)
