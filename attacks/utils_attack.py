# this modeule is independent from any specific models

import random

import numpy as np
from matplotlib import pyplot as plt


def get_area_inter(bbox1, bbox2):
    """Calculate the area of Intersection of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        area_inter (float): area
    """
    w1,h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2,h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    assert all([w1,h1,w2,h2])

    # determine the coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    return area_inter


def get_bbox_dist(bbox1, bbox2):
    """Calculate the area of Intersection of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        dist (float): distance between centers of bboxes
    """
    x1 = (bbox1[0] + bbox1[2]) / 2
    y1 = (bbox1[1] + bbox1[3]) / 2
    x2 = (bbox2[0] + bbox2[2]) / 2
    y2 = (bbox2[1] + bbox2[3]) / 2
    dist = ((x1-x2)**2 + (y1-y2)**2)**(0.5)
    return dist


def get_iou(bbox1, bbox2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        bbox1 (numpy.ndarray): x1,y1,x2,y2
        bbox2 (numpy.ndarray): x1,y1,x2,y2
    Returns:
        iou (float): iou in [0, 1]
    """
    w1,h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
    w2,h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
    try:
        assert all([w1,h1,w2,h2])
    except:
        return 0

    # determine the coordinates of the intersection rectangle
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    area1 = w1*h1
    area2 = w2*h2
    iou = area_inter / float(area1 + area2 - area_inter)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def is_success(detections, target_clean):
    """ see if the detection has target label at the corresponding location with IOU > 0.3
    Args:
        detections (np.ndarray): a list of detected objects. Shape (n,6)
        target_clean (np.ndarray): a single object, our desired output. Shape (1,6) - [xyxy,cls,score]
    Returens:
        (bool): whether the detection is a success or not
    """
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > 0.3 and items[4] == target_clean[0][4]:
            return True
    return False


def is_success_hiding(detections, target_clean):
    """ see if the adv image can fool model
    Args:
        target_clean (): Single object, our desired output. Shape (1,6) - [xyxy,cls,score]
    """
    for items in detections:
        iou = get_iou(items, target_clean[0])
        if iou > 0.3:
            return False
    return True


def show_figure_ap_det(im, adv, attack_plan, attack_step, detections, dataset, save_path=None):
    """Show figures with attack plans and detections
        It has 2 rows, 
            1st row shows the target object and clean det, 
            2nd row shows the attack plan and adversarial det.
    Args:
        im (np.ndarray): clean image
        adv (np.ndarray): perturbed image
        attack_plan (np.ndarray): 
        detections (dict of np.ndarray):
        dataset (str): voc or coco
        save_path (str): if not None, do not display, only save to that path
    """
    from vis_tool import vis_bbox

    target_clean = attack_plan[:1]
    n_mdoels = len(detections)

    fig, ax = plt.subplots(2, n_mdoels+1, figsize=(6*n_mdoels, 5*2))

    # plot target object
    bboxes, labels, scores = target_clean[:,:4], target_clean[:,4], target_clean[:,5]
    ax[0,0].set_title(f'Target Clean', fontsize=15)
    vis_bbox(im, bboxes, labels, scores, ax=ax[0,0], dataset=dataset)
    # plot attack_plan
    bboxes, labels, scores = attack_plan[:,:4], attack_plan[:,4], attack_plan[:,5]
    ax[1,0].set_title(f'Attack Plan {attack_step}', fontsize=15)
    vis_bbox(adv, bboxes, labels, scores, ax=ax[1,0], dataset=dataset)

    for row in range(2):
        for idx, key in enumerate(detections):
            det = detections[key]
            bboxes, labels, scores = det[:,:4], det[:,4], det[:,5]
            ax[row,idx+1].set_title(f"{key}", fontsize=15)
            vis_bbox(im, bboxes, labels, scores, ax=ax[row,idx+1], dataset=dataset)
            
    if save_path is None:
        plt.show()
    else:
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)


def random_box(im, bbox):
    """Generate a random bbox in the image, same size as the victim object
    Args:
        im (np.ndarray): the input image 
        bbox (np.ndarray): bounding box of the victim object, [x1,y1,x2,y2]
    Returns:
        bbox_ (np.ndarray): bounding box of the new object
    """
    H,W,_ = im.shape
    bbox = bbox.astype(np.int)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if np.random.randint(0,2): # random
        # left
        x1 = random.randint(bbox[0]//2, int(bbox[0]))
        y1 = random.randint(bbox[1]//2, int(bbox[1]))
    else:
        # right
        min_x = np.minimum(bbox[0], W - (W-w)//2)
        max_x = np.maximum(bbox[0], W - (W-w)//2)
        x1 = random.randint(int(min_x), int(max_x))

        min_y = np.minimum(bbox[1], H - (H-h)//2)
        max_y = np.maximum(bbox[1], H - (H-h)//2)
        y1 = random.randint(int(min_y), int(max_y))
    x2 = np.minimum(x1+w, W)
    y2 = np.minimum(y1+h, H)
    bbox_ = np.array([x1,y1,x2,y2])
    return bbox_


def random_box_close(im, bbox):
    """Generate a random bbox in the image, same size as the victim object, 
    located very close to the victim object, overlaps with it
    Args:
        im (np.ndarray): the input image 
        bbox (np.ndarray): bounding box of the victim object, [x1,y1,x2,y2]
    Returns:
        bbox_ (np.ndarray): bounding box of the new object
    """
    H,W,_ = im.shape
    bbox = bbox.astype(np.int)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    # in an 8 connected box, for (x1,y1)
    x_min = max(0, bbox[0] - w)
    x_max = min(W-w, bbox[2])
    y_min = max(0, bbox[1] - h)
    y_max = min(H-h, bbox[3])
    x1 = random.randint(x_min, x_max)
    y1 = random.randint(y_min, y_max)
    x2 = x1 + w
    y2 = y1 + h
    bbox_ = np.array([x1,y1,x2,y2])
    return bbox_
    

def random_box_hiding(im, bbox = None):
    """ Generate a bbox in the non-overlapping area of the victim object
        same size as bbox, non-overlapping
    """
    H,W,_ = im.shape
    bbox = bbox.astype(np.int)

    loc0 = max(1,bbox[0])
    loc1 = max(1,bbox[1])
    loc2 = min(bbox[2],W-1)
    loc3 = min(bbox[3],H-1)
    w = int(bbox[2] - bbox[0])
    h = int(bbox[3] - bbox[1])
    x1 = np.random.randint(0,W)
    if x1 < loc0:
        # left
        y1 = np.random.randint(0,H)
        x2 = np.minimum(x1+w,loc0)
        y2 = np.minimum(y1+h,H)
    elif loc0 <= x1 <=  loc2:
        if np.random.randint(0,2): # flip a coin, above the bbox
            # above
            y1 = np.random.randint(0,loc1)
            x2 = np.minimum(x1+w,W)
            y2 = np.minimum(y1+h,loc1)
        else:
            # below
            y1 = np.random.randint(loc3,H)
            x2 = np.minimum(x1+w,W)
            y2 = np.minimum(y1+h,H)
    else:
        # right
        y1 = np.random.randint(0,H)
        x2 = np.minimum(x1+w,W)
        y2 = np.minimum(y1+h,H)
    bbox_ = np.array([x1,y1,x2,y2])
    return bbox_


def generate_box(im, bbox, target_class, helper_class, dist_mat, size_mat, size_mat_single):
    """Generate a box for the helper object
        It determines the size for the object, and the distance to the victim object
    
    Args:
        im (numpy.ndarray): original image where we get the image size from.
        bbox (numpy.ndarray): original bbox of victim object.
        target_class (int): the target label.
        helper_class (int): the helper object label.
        dist_mat (dict of list): distance matrix; with keys f"{target_class}_{helper_class}"
        size_mat (dict of list): size matrix; with keys f"{target_class}_{helper_class}"
    Returns:
        bbox_ (numpy.ndarray): bbox of new helper object.
    """
    H,W,_ = im.shape
    L = np.sqrt(W**2 + H**2)
    bbox = bbox.astype(np.int)
    xc,yc = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]).astype(int)
    o1 = target_class
    o2 = helper_class
    key = f'{o1}_{o2}'
    
    if key in dist_mat.keys():
        # dist = np.array(dist_mat[key]).mean()
        dist = np.array(dist_mat[key])
        dist_ = (dist*L).astype(int)
        # size = np.array(size_mat[key]).mean(axis=0)
        size = np.array(size_mat[key])
        h,w = (size*L).astype(int)
    
        for dist_factor in [1,1/2,1/4]:
            # if cannot find valid location for the original distance
            # halve the distance
            dist = dist_factor * dist_

            # check the valid direction that the bbox does not go out of the image
            # distance from victim center to the right, top, left, bottom
            # calculate the angle that /_ (o1,o2)
            dist0 = int(W - w/2 - xc)
            dist1 = int(yc - h/2)
            dist2 = int(xc - w/2)
            dist3 = int(H - h/2 - yc)

            # if two adjacent directions (0,1,2,3):(right,up,left,down) are valid
            # randomly choose an angle between these two angles
            start_angles = [0,90,180,270] # degree
            directions = [0,1,2,3]
            np.random.shuffle(directions)
            for dir1 in directions:
                if eval(f'dist{dir1}') > dist:
                    for next_direction in [1,-1]:
                        dir2 = (dir1 + next_direction) % 4
                        if eval(f'dist{dir2}') > dist:
                            # print(f"dir1: {dir1}, dir2: {dir2}")
                            start_angle = start_angles[dir1]
                            angle = start_angle + next_direction * np.random.randint(0,90)
                            rad = np.pi/180 * angle
                            cos = np.cos(rad)
                            sin = np.sin(rad)
                            center = np.array([xc+dist*cos,yc-dist*sin]).astype(int)
                            bbox_ = np.array([center[0]-w//2,center[1]-h//2,center[0]+w//2,center[1]+h//2]).astype(int)
                            # make sure the bbox is in the image
                            bbox_ = np.array([max(0,bbox_[0]),max(0,bbox_[1]),max(0,bbox_[2]),max(0,bbox_[3])])
                            bbox_ = np.array([min(W,bbox_[0]),min(H,bbox_[1]),min(W,bbox_[2]),min(H,bbox_[3])])
                            return bbox_
    else:
        # if the helper object has never been seen together with the target object
        # dist is nowhere to be found
        # size can only be obtained by individual information
        # size = np.array(size_mat_single[f'{o2}']).mean(axis=0)
        size = np.array(size_mat_single[f'{o2}'])
        h,w = (size*L).astype(int)
    
    # if the valid direction does not exist, find a random position
    # topleft corner of the bbox
    x1 = random.randint(0, max(0,W-w))
    y1 = random.randint(0, max(0,H-h))
    bbox_ = np.array([x1, y1, x1 + w, y1 + h])
    bbox_ = np.array([max(0,bbox_[0]),max(0,bbox_[1]),max(0,bbox_[2]),max(0,bbox_[3])])
    bbox_ = np.array([min(W,bbox_[0]),min(H,bbox_[1]),min(W,bbox_[2]),min(H,bbox_[3])])
    return bbox_


def get_sequential_attack_plan(im, det, victim_idx, target_class, matrix, dist_mat, size_mat, size_mat_single, n = 5, randomize = False):
    """Sequential attack: perturb existing objects first and then add new objects.
        Eg: have 2 objects in the image
            step1: change the victim object only
            step2: change the other object 
            step3: add a helper object
            step4: add another helper object
            ...
        
        Args:
            im (~ numpy.ndarray): original image where we get the image size from.
            det (~ numpy.ndarray): original detection based on which we perform attack.
            victim_idx (~ int): the index of the object to change in the det.
            target_class (~ int): the target label
            matrix (~ numpy.ndarray): co-occurrence matrix where we get prob from.
            n (~ int): max number of objects to change or add.
            vanishing (~ Bool): if vanishing is True, then perform hiding attack.
        Returns:
            all_attack_plans (~ dict of numpy.ndarray): keys are ap0,ap1...apn. n+1 attack plans in total.
    """
    det = det.copy() # avoid changing the input parameters
    n_labels = len(matrix)
    probabilities = matrix[target_class]

    # only choose helper labels with over 5% probability
    helper_list = np.arange(n_labels)[probabilities > 0.05]
    helper_prob = probabilities[probabilities > 0.05]
    helper_prob = helper_prob / sum(helper_prob)
    if randomize:
        # all labels with equal prob
        helper_list = np.arange(n_labels)
        helper_prob = np.ones_like(probabilities) / n_labels

    victim_row = det[victim_idx]
    # make attack plan:
    # put victim_row in the first row and change the label
    # target clean is the first row
    attack_plan = victim_row.copy()
    attack_plan[4] = target_class
    attack_plan[5] = 0.99
    # other unchanged
    other_idx = np.ones(len(det)).astype(np.bool)
    other_idx[victim_idx] = False
    det_others = det[other_idx]
    attack_plan = np.vstack([attack_plan, det_others])
    num_objs = len(det)
    
    all_attack_plans = {}
    # attack plan contains victim class in the first row
    
    ap = 0
    # ap0 is baseline
    all_attack_plans[f'ap{ap}'] = attack_plan.copy()

    # n is the total number of helper objects
    # number of objects to change and add
    n_change = min(num_objs-1,n)
    n_add = max(0,n-num_objs+1)

    # change the existing labels first
    for i in range(n_change):
        helper_label = np.random.choice(helper_list, 1, p=helper_prob)[0]
        attack_plan[i+1,4] = helper_label
        attack_plan[i+1,5] = 0.99
        ap += 1
        all_attack_plans[f'ap{ap}'] = attack_plan.copy()
    # and then add helper objects
    for i in range(n_add):
        helper_label = np.random.choice(helper_list, 1, p=helper_prob)[0]
        row = victim_row.copy()
        row[:4] = generate_box(im, victim_row[:4], target_class, helper_label, dist_mat, size_mat, size_mat_single)
        row[4] = helper_label
        row[5] = 0.99
        attack_plan = np.vstack([attack_plan, row])
        ap += 1
        all_attack_plans[f'ap{ap}'] = attack_plan.copy()
    return all_attack_plans
