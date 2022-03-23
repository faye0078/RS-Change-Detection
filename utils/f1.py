import numpy as np

def f1(intersect_area, pred_area, label_area):
    """
    Calculate iou.

    Args:
        intersect_area (Tensor): The intersection area of prediction and ground truth on all classes.
        pred_area (Tensor): The prediction area on all classes.
        label_area (Tensor): The ground truth area on all classes.

    Returns:
        np.ndarray: iou on all classes.
        float: mean iou of all classes.
    """
    intersect_area = intersect_area.numpy()
    pred_area = pred_area.numpy()
    label_area = label_area.numpy()
    union = (pred_area + label_area) * 0.5
    class_f1 = []
    for i in range(len(intersect_area)):
        if union[i] == 0:
            f1 = 0
        else:
            f1 = intersect_area[i] / union[i]
        class_f1.append(f1)
    f1 = np.sum(intersect_area) / np.sum(union)
    return np.array(class_f1), f1