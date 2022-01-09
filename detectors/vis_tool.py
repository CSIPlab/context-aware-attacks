import random

import numpy as np
from matplotlib import pyplot as plot


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

COCO_BBOX_LABEL_NAMES = (
    'person',
    'bicycle',
    'car',
    'motorcycle',
    'airplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
    'fire hydrant',
    'stop sign',
    'parking meter',
    'bench',
    'bird',
    'cat',
    'dog',
    'horse',
    'sheep',
    'cow',
    'elephant',
    'bear',
    'zebra',
    'giraffe',
    'backpack',
    'umbrella',
    'handbag',
    'tie',
    'suitcase',
    'frisbee',
    'skis',
    'snowboard',
    'sports ball',
    'kite',
    'baseball bat',
    'baseball glove',
    'skateboard',
    'surfboard',
    'tennis racket',
    'bottle',
    'wine glass',
    'cup',
    'fork',
    'knife',
    'spoon',
    'bowl',
    'banana',
    'apple',
    'sandwich',
    'orange',
    'broccoli',
    'carrot',
    'hot dog',
    'pizza',
    'donut',
    'cake',
    'chair',
    'couch',
    'potted plant',
    'bed',
    'dining table',
    'toilet',
    'tv',
    'laptop',
    'mouse',
    'remote',
    'keyboard',
    'cell phone',
    'microwave',
    'oven',
    'toaster',
    'sink',
    'refrigerator',
    'book',
    'clock',
    'vase',
    'scissors',
    'teddy bear',
    'hair drier',
    'toothbrush')

# voc index to coco index
voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def vis_image(img, ax=None):
    """Visualize a color image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(height, width, 3)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img
    ax.imshow(img.astype(np.uint8))
    ax.axis('off')
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None, dataset='voc'):
    """Visualize bounding boxes inside image.

    Args:
        img (~numpy.ndarray): An array of shape :math:`(height, width, 3)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(x_{min}, y_{min}, x_{max}, y_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        dataset (str): voc or coco.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.

    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.

    """
    random.seed(1)
    rgbas = [[random.randint(0, 255)/255 for _ in range(3)] + [1] for _ in range(80)] # to unify colors pick from 80 classes
    voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]

    if dataset == 'voc':
        label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
        rgbas = np.array(rgbas)[voc2coco].tolist()
    else:
        label_names = list(COCO_BBOX_LABEL_NAMES) + ['bg']

    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')
    
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    # Draw bbox and captions
    for i, bb in enumerate(bbox):
        
        caption = list()

        if label is not None and label_names is not None:
            lb = int(label[i])
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        ax.add_patch(plot.Rectangle(
            xy, width, height, fill=False, edgecolor=rgbas[lb], linewidth=3))

        if len(caption) > 0:
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    color='white',
                    size="large",
                    style='italic',
                    bbox={'facecolor': rgbas[lb], 'edgecolor': rgbas[lb], 
                    'boxstyle':'round', 'alpha': 0.8, 'pad': 0.2})
    return ax

