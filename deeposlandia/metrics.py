"""Metrics for measuring machine learning algorithm performances
"""

from keras import backend


def iou(actual, predicted):
    """Compute Intersection over Union statistic (i.e. Jaccard Index)

    See https://en.wikipedia.org/wiki/Jaccard_index

    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Intersection over Union value
    """
    actual = backend.flatten(actual)
    predicted = backend.flatten(predicted)
    intersection = backend.sum(actual * predicted)
    union = backend.sum(actual) + backend.sum(predicted) - intersection
    return 1.0 * intersection / union


def iou_loss(actual, predicted):
    """Loss function based on the Intersection over Union (IoU) statistic

    IoU is comprised between 0 and 1, as a consequence the function is set as
    `f(.)=1-IoU(.)`: the loss has to be minimized, and is comprised between 0
    and 1 too

    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Intersection-over-Union-based loss
    """
    return 1.0 - iou(actual, predicted)


def dice_coef(actual, predicted, eps=1e-3):
    """Dice coef

    See https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Examples at:
      -
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L23
      -
    https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/zf_unet_224_model.py#L36


    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels
    eps : float
        Epsilon value to add numerical stability

    Returns
    -------
    float
        Dice coef value
    """
    y_true_f = backend.flatten(actual)
    y_pred_f = backend.flatten(predicted)
    intersection = backend.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + eps) / (
        backend.sum(y_true_f) + backend.sum(y_pred_f) + eps
    )


def dice_coef_loss(actual, predicted):
    """
    Parameters
    ----------
    actual : list
        Ground-truth labels
    predicted : list
        Predicted labels

    Returns
    -------
    float
        Dice-coef-based loss
    """
    return -dice_coef(actual, predicted)
