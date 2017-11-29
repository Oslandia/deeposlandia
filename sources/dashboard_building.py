# Author: Raphael Delhome
# Organization: Oslandia
# Date: september 2017

import matplotlib.pyplot as plt
import numpy as np
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
    return confusion_matrix(y_true[:,index], y_predicted[:,index],
                            labels=[0, 1]).reshape([-1]).tolist()

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
    _, fp, _, tp = confusion_matrix_by_label(y_true, y_predicted, index)
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
    _, _, fn, tp = confusion_matrix_by_label(y_true, y_predicted, index)
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
    return (confusion_matrix_by_label(y_true, y_predicted, index)
            + [accuracy_by_label(y_true, y_predicted, index),
               precision_by_label(y_true, y_predicted, index),
               recall_by_label(y_true, y_predicted, index)])

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
    conf_matrix = (confusion_matrix(utils.unnest(y_true),
                                    utils.unnest(y_predicted))
                   .reshape([-1])
                   .tolist())
    dashboard = dashboard + conf_matrix
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

def plot_dashboard(dashboard, plot_filename, label_to_plot, plot_size=(24, 16)):
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
    label_to_plot: list
        list of integer designing the indices of the label that must be plotted
    plot_filename: object
        string designing the name of the file in which the plot has to be saved
    
    """
    fig = plt.figure(figsize=plot_size)
    plt.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.5, 0.6)
    for i, x in enumerate(label_to_plot):
        a = plt.subplot2grid((11, 12), (int(x/6), 6+x%6))
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+8], 'r-')
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+9], 'b-')
        a.plot(dashboard.epoch, dashboard.iloc[:,3*i+10], 'g-')
        a.set_ylim((0, 1))
        a.set_title("Label "+str(x), size=10)
        a.xaxis.set_visible(False)
        if x % 6 > 0:
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
    a.set_ylim((0, 1))
    a.set_xlabel('epoch')
    a.set_ylabel('model evaluation')
    a.legend(['accuracy', 'precision' ,'recall', 'F-measure'], prop={'size':12})
    fig.savefig(plot_filename)
    plt.close("all")

def dashboard_summary(dashboard):
    """ Give the summary of dashboard, i.e. the situation of global metrics
    each tenth of training step

    Parameters
    ----------
    dashboard: pandas.DataFrame
    training process dashboard, with 12 global variables, and 7 variables for
    each label
    """
    return dashboard.iloc[np.linspace(0, len(dashboard)-1, 11), :12]

def dashboard_label_summary(dashboard, index):
    """Give the summary of a specific label within dashboard, i.e. the
    situation of metrics dedicated to it each tenth of training step

    Parameters
    ----------
    dashboard: pandas.DataFrame
    training process dashboard, with 12 global variables, and 7 variables for
    each label

    """
    return dashboard.iloc[np.linspace(0, len(dashboard)-1, 11),
                          (index*7+12):(index*7+19)]


def dashboard_result(dashboard, step):#, datapath, image_size):
    """Return an extended version of training process dashboard, by giving
    positive and negative labels, as well as positive and negative predicted
    labels, for each glossary item

    Parameters
    ----------
    dashboard: pandas.DataFrame
    training process dashboard, with 12 global variables, and 7 variables for
    each label
    """
    db = dashboard.drop(["epoch", "loss", "bpmll_loss", "hamming_loss",
    "F_measure"], axis=1)
    db = db.loc[step].T.reset_index()
    db.columns = ["metric", "value"]
    db["label"] = np.repeat("global", db.shape[0])
    db.loc[7:, "label"] = ["label_" + str(i) for i in np.repeat(range(66), 7)]
    db["metric"] = np.tile(["tn", "fp", "fn", "tp",
                            "accuracy", "precision", "recall"], 67)
    db = db.set_index(["label", "metric"]).unstack().value
    db['positive'] = db['tp'] + db['fn']
    db['negative'] = db['tn'] + db['fp']
    db['pos_pred'] = db['tp'] + db['fp']
    db['neg_pred'] = db['tn'] + db['fn']
    db['positive_part'] = (100 * db['positive'] /
                                  (db['positive'] +
                                   db['negative']))
    db['pos_pred_part'] = (100 * db['pos_pred'] /
                                  (db['pos_pred'] +
                                   db['neg_pred']))
    return db

def analyze_model_results(dashboard, period=10):
    """Analyze the model response after each training step
    """
    model_results = []
    for step in dashboard.index[0:len(dashboard):period]:
        utils.logger.info("Step {}".format(step))
        db_results = dashboard_result(dashboard, step)
        hist_results = plt.hist(db_results.pos_pred_part[1:],
                                bins=np.linspace(0, 100, 11))
        model_results.append(hist_results[0].tolist())
    return model_results
