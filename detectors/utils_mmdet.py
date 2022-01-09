# this module is related to mmdetection models

import numpy as np
from matplotlib import pyplot as plt


def is_to_rgb(model):
    """check if a model takes rgb images or not
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
    """
    for item in model.cfg.data.test.pipeline[1]['transforms']:
        if 'to_rgb' in item:
            to_rgb = item['to_rgb']
    return to_rgb


def get_conf_thres(model):
    """assign a different confidence threshold for every model
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
    Returns:
        conf_thres (~ float): the confidence threshold
    """
    model_type = model.cfg.model.type
    if model_type in ['FasterRCNN']: # frcnn, foveabox, Libra R-CNN, GN+WS 
        conf_thres = 0.7
    elif model_type in ['RetinaNet', 'SingleStageDetector']: # retina, ssd
        conf_thres = 0.5
    elif model_type in ['YOLOV3']: # yolov3        
        conf_thres = 0.5
    elif model_type in ['FCOS']:       
        conf_thres = 0.2
    elif model_type in ['FOVEA']:
        conf_thres = 0.3
    else:
        conf_thres = 0.1
    return conf_thres


def output2det(outputs, im, conf_thres = 0.5, dataset='voc'):
    """Convert the model outputs to targeted format
    Args: 
        conf_thres (float): confidence threshold
    Returns:
        det (numpy.ndarray): _bboxes(xyxy) - 4, _cls - 1, _prob - 1
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    """
    det = []
    for idx, items in enumerate(outputs):
        for item in items:
            det.append(item[:4].tolist() + [idx] + item[4:].tolist())
    det = np.array(det)
    
    # if det is empty
    if len(det) == 0: 
        return np.zeros([0,6])

    # thresholding the confidence score
    det = det[det[:,-1] >= conf_thres]
    
    if dataset == 'voc':
        # map the labels from coco to voc
        voc2coco = [4, 1, 14, 8, 39, 5, 2, 15, 56, 19, 60, 16, 17, 3, 0, 58, 18, 57, 6, 62]
        for idx, item in enumerate(det):
            if item[4] not in voc2coco:
                item[4] = -1
            else:
                det[idx,4] = voc2coco.index(item[4])
        det = det[det[:,4] != -1]

    # make the value in range
    m, n, _ = im.shape
    for item in det:
        item[0] = min(max(item[0],0),n)
        item[2] = min(max(item[2],0),n)
        item[1] = min(max(item[1],0),m)
        item[3] = min(max(item[3],0),m)
    return det


def get_det(model, im, dataset='voc'):
    """input an image to a model and get the detection
    Args: 
        model (~ mmdet.models.detectors): a mmdet model
        im (~ numpy.ndarray): input image (in rgb format)
        dataset (str): if use 'voc', only the labels within the voc dataset will be returned
    Returns:
        det (~ numpy.ndarray): nx6 array
    """
    from mmdet.apis import inference_detector
    if not is_to_rgb(model):
        im = im[:,:,::-1]
    result = inference_detector(model, im)
    conf_thres = get_conf_thres(model)
    det = output2det(result, im, conf_thres, dataset)
    return det


def show_det(models, im, dataset='voc', save_path=None):
    """show detection of a list of models
    Args: 
        models (~ mmdet.models.detectors or list): a single model or a list of models
        im (~ numpy.ndarray): input image (in rgb format)
    """
    from vis_tool import vis_bbox

    det_all = []
    if isinstance(models,list):    # a list of models
        n_mdoels = len(models)
        fig, ax = plt.subplots(1, n_mdoels, figsize=(6*n_mdoels, 5))
        for idx, model in enumerate(models):
            det = get_det(model, im, dataset) if is_to_rgb(model) else get_det(model, im[:,:,::-1], dataset)
            det_all.append(det)
            bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
            vis_bbox(im, bboxes, labels, scores, ax=ax[idx], dataset=dataset)
    else:   # a single model
        model = models
        det = get_det(model, im, dataset) if is_to_rgb(model) else get_det(model, im[:,:,::-1], dataset)
        det_all.append(det)
        bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(1, 1, 1)
        vis_bbox(im, bboxes, labels, scores, ax=ax, dataset=dataset)
    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
    return det_all


def get_test_data(model, im):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format)
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose
    from mmcv.parallel import collate, scatter

    if not is_to_rgb(model): im = im[:,:,::-1]
    cfg = model.cfg
    device = next(model.parameters()).device
    cfg = cfg.copy()
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)
    datas = []
    data = dict(img=im)
    data = test_pipeline(data)
    datas.append(data)
    data = collate(datas, samples_per_gpu=1)
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    data = scatter(data, [device])[0]
    return data

    
def get_train_data(model, im, pert, data, bboxes, labels):
    """get data format for training
    Args:
        model (~ mmdet.models.detectors): a mmdet model
        im (np.ndarray): input numpy image (in bgr format) / with grad
        bboxes (np.ndarray): desired bboxes
        labels (np.ndarray): desired labels
    Returns:
        data_train (): train data format
    """
    import torch
    from torch.nn import functional as F
    from torchvision import transforms

    # get model device
    device = next(model.parameters()).device

    # BELOW IS TRAIN
    data_train = data.copy()
    data_train['img_metas'] = data_train['img_metas'][0]
    data_train['img'] = data_train['img'][0]
    ''' from file: datasets/pipelines/transforms.py '''
    
    if not is_to_rgb(model): im = im[:,:,::-1]
    img = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().to(device).contiguous()
    img = (img + pert).clamp(0,255)

    # 'type': 'Resize', 'keep_ratio': True, (1333, 800)
    ori_sizes = im.shape[:2]
    image_sizes = data_train['img_metas'][0]['img_shape'][:2]
    w_scale = image_sizes[1] / ori_sizes[1]
    h_scale = image_sizes[0] / ori_sizes[0]
    scale_factor = np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
    gt_bboxes = bboxes * scale_factor
    gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, image_sizes[1])
    gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, image_sizes[0])
    data_train['gt_bboxes'] = [torch.from_numpy(gt_bboxes).to(device)]
    data_train['gt_labels'] = [torch.from_numpy(labels).to(device)]
    # img = F.interpolate(img, size=image_sizes, mode='nearest')
    img = F.interpolate(img, size=image_sizes, mode='bilinear', align_corners=True)

    # 'type': 'Normalize', 'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], 'to_rgb': False
    img_norm_cfg = data_train['img_metas'][0]['img_norm_cfg']
    mean = img_norm_cfg['mean']
    std = img_norm_cfg['std']
    transform = transforms.Normalize(mean=mean, std=std)
    img = transform(img)

    # 'type': 'Pad', 'size_divisor': 32
    pad_sizes = data_train['img_metas'][0]['pad_shape'][:2]
    left = top = 0
    bottom = pad_sizes[0] - image_sizes[0]
    right = pad_sizes[1] - image_sizes[1]
    img = F.pad(img, (left, right, top, bottom), "constant", 0)
    data_train['img'] = img
    return data_train


def get_train_model(config_file, checkpoint_file, device='cuda:0'):
    """return a model in train mode
    Args:
        input the same config_file, checkpoint_file as test models
        device (~ str): indicates which gpu to allocate
    """
    import mmcv
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
    config = mmcv.Config.fromfile(config_file)
    model_train = build_detector(config.model, test_cfg=config.get('test_cfg'))
    map_loc = 'cpu' if device == 'cpu' else None
    checkpoint = load_checkpoint(model_train, checkpoint_file, map_location=map_loc)
    if 'CLASSES' in checkpoint.get('meta', {}):
        model_train.CLASSES = checkpoint['meta']['CLASSES']
    model_train.cfg = config  # save the config in the model for convenience
    model_train.to(device)
    model_train.train()
    return model_train


def get_loss_from_dict(model, loss_dict):
    """Return the correct loss based on the model type
    Args:
        model (~ mmdet.models.detectors): the mmdet model where we get the type, eg: 'FasterRCNN' or 'YOLOV3'
        loss_dict (~ dict): the loss of the model, stored in a dictionary
    Returns:
        losses (~ torch.Tensor): the summation of the loss
    """
    model_type = model.cfg.model.type
    if model_type in ['FasterRCNN']: # frcnn, foveabox, Libra R-CNN, GN+WS 
        losses = loss_dict['loss_cls'] + loss_dict['loss_bbox'] + sum(loss_dict['loss_rpn_cls']) + sum(loss_dict['loss_rpn_bbox'])
    elif model_type in ['RetinaNet', 'SingleStageDetector']: # retina, ssd
        losses = sum(loss_dict['loss_cls']) + sum(loss_dict['loss_bbox'])
    elif model_type in ['YOLOV3']: # yolov3        
        losses = sum(loss_dict['loss_cls']) + sum(loss_dict['loss_conf']) + sum(loss_dict['loss_xy']) + sum(loss_dict['loss_wh'])
    return losses
