import os
import sys
import random
import argparse
import json as JSON
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from tqdm import tqdm

from utils_attack import is_success, show_figure_ap_det, get_sequential_attack_plan
mmdet_root = Path('../detectors/mmdetection')
sys.path.insert(0, str(mmdet_root))
from mmdet.apis import init_detector
sys.path.append('../detectors/')
from mmdet_model_info import model_info
from vis_tool import voc2coco
from utils_mmdet import get_det, show_det, get_train_model, get_train_data, get_test_data, get_loss_from_dict


def main():
    parser = argparse.ArgumentParser(description="run sequential attacks")    
    parser.add_argument("--eps", nargs="?", default=30, help="perturbation level (linf): 10,20,30")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--ratio", nargs="?", default=4, help="ratio between frcnn and yolo loss, 4:1 turned out to be good")
    parser.add_argument("--dataset", nargs="?", default='voc', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    parser.add_argument("-random", action='store_true', help="randomize the co-occurrence matrix for comparison")
    parser.add_argument("-vanish", action='store_true', help="vanish other objects in the image")
    args = parser.parse_args()

    eps = int(args.eps)
    gpu_id = int(args.gpu)
    result_folder = args.root
    ratio = int(args.ratio)
    dataset = args.dataset
    randomize = args.random
    vanish = args.vanish
    

    # load models
    # if the GPU memory is not enough, we may need to assign models to multiple GPUs
    device = f'cuda:{gpu_id}'
    models = []
    models_train = []
    model_list = ['Faster R-CNN', 'YOLOv3', 'RetinaNet']
    for model_name in model_list:
        config_file = model_info[model_name]['config_file']
        checkpoint_file = model_info[model_name]['checkpoint_file']
        config_file = str(mmdet_root/config_file)
        checkpoint_file = str(mmdet_root/checkpoint_file)
        models.append(init_detector(config_file, checkpoint_file, device=device))
        models_train.append(get_train_model(config_file, checkpoint_file, device=device))

    # use 'Faster R-CNN', 'YOLOv3' as wb; 'RetinaNet' as bb
    holdout_idx = 0
    n_wb = 2
    model_holdout = models[holdout_idx] # indicator model, it represents the detection of the whole ensemble
    models_wb = models[:n_wb]
    models_train_wb = models_train[:n_wb]
    models_bb = models[n_wb:]

    # some training parameters
    alpha = 2
    max_iters = 50
    max_attack_step = 5
    # 0 is baseline, 1-5 is sequentially adding helper objects, 

    # get file name as experiment name
    exp_name = os.path.abspath(sys.argv[0]).split('/')[-1].split('.')[0]
    exp_name = exp_name + f'_eps{eps}'
    # save results, figures, perturbations etc. into different folders
    exp_root = Path(result_folder) / exp_name
    fig_root = exp_root / 'fig'
    fig_root.mkdir(parents=True, exist_ok=True)
    ap_root = exp_root / 'ap'
    ap_root.mkdir(parents=True, exist_ok=True)
    pert_root = exp_root / 'pert'
    pert_root.mkdir(parents=True, exist_ok=True)
    det_root = exp_root / 'det'
    det_root.mkdir(parents=True, exist_ok=True)

    # load co-occurrence matrix
    co_mat = JSON.load(open(f'../context/{dataset}_co_mat.json'))
    co_mat = np.array(co_mat).astype(np.float)
    # convert to probability matrix
    for idx in range(len(co_mat)):
        co_mat[idx,:] = co_mat[idx,:]/np.sum(co_mat[idx,:])
    # load distance matrix and size matrix
    dist_mat = np.load(f'../context/{dataset}_dist_mat_avg.npy',allow_pickle=True).item()
    size_mat = np.load(f'../context/{dataset}_size_mat_avg.npy',allow_pickle=True).item()
    size_mat_single = np.load(f'../context/{dataset}_size_mat_single_avg.npy',allow_pickle=True).item()
    n_labels = len(co_mat)
    # Load test image ids
    test_image_ids = JSON.load(open(f"../data/{dataset}_2to6_objects.json"))
    if dataset == 'voc':
        data_root = Path("../data/VOC/VOC2007/JPEGImages/")
    else:
        data_root = Path("../data/COCO/val2017/")

    count = defaultdict(set)  # count the success of different attack plans
    for im_id in tqdm(test_image_ids[:500]):
        path = data_root / f"{im_id}.jpg"
        im = np.array(Image.open(path).convert('RGB'))
        fig_save_path = fig_root / f"im{im_id}.png"
        show_det(models, im, dataset, fig_save_path)
        
        # save detection result of each attack plan
        detections = {}
        for idx, model in enumerate(models_wb):
            detections[f'wb{idx}'] = get_det(model, im, dataset)
        for idx, model in enumerate(models_bb):
            detections[f'bb{idx}'] = get_det(model, im, dataset)
        det_save_path = det_root / f"im{im_id}"
        np.save(det_save_path, detections)

        det_holdout = get_det(model_holdout, im, dataset)
        all_categories = set(det_holdout[:, 4].astype(int))  # all apperaing objects in the scene

        # save original detections
        # det_save_path = det_root / f"im{im_id}_idx{victim_idx}_f{victim_class}_t{target_class}_{key}"
        # np.save(det_save_path, detections)
        
        # randomly select a victim
        victim_idx = random.randint(0,len(det_holdout)-1)
        victim_class = int(det_holdout[victim_idx,4])

        # randomly select a target
        select_n = 1 # for each victim object, randomly select 5 target objects
        target_pool = list(set(range(n_labels)) - all_categories)
        target_pool = np.random.permutation(target_pool)[:select_n]

        for target_class in target_pool:
            target_class = int(target_class)

            count['attacks'].add(f"im{im_id}_idx{victim_idx}_f{victim_class}_t{target_class}")

            # get the attack plans
            attack_plan_sequential = get_sequential_attack_plan(im, det_holdout, victim_idx, target_class, co_mat, dist_mat, size_mat, size_mat_single, n=max_attack_step, randomize=False)
            if vanish:
                # only keep n+1 objects
                for key in attack_plan_sequential:
                    n = int(key[2:])
                    attack_plan_sequential[key] = attack_plan_sequential[key][:n+1]
            attack_plan_all = {**attack_plan_sequential}
            
            if randomize:
                # the random result will be added after the original result
                attack_plan_sequential_random = get_sequential_attack_plan(im, det_holdout, victim_idx, target_class, co_mat, dist_mat, size_mat, size_mat_single, n=max_attack_step, randomize=True)
                if vanish:
                    # only keep n+1 objects
                    for key in attack_plan_sequential_random:
                        n = int(key[2:])
                        attack_plan_sequential_random[key] = attack_plan_sequential_random[key][:n+1]
                for i in range(1,6):
                    attack_plan_all[f"ap{i+5}"] = attack_plan_sequential_random[f"ap{i}"]

            target_clean = attack_plan_all['ap0'][:1]
            # save attack plan
            ap_save_path = ap_root / f"im{im_id}_idx{victim_idx}_f{victim_class}_t{target_class}"
            np.save(ap_save_path, attack_plan_all)

            for attack_step in attack_plan_all:
                # attack_step is ap0, ap1 ... ap5, oneshot, 
                attack_plan = attack_plan_all[attack_step]
                
                # prepare target label input: convert labels from voc to coco
                bboxes_tgt = attack_plan[:,:4].astype(np.float32)
                labels = attack_plan[:,4].astype(np.long)
                labels_tgt = labels.copy()
                if dataset == 'voc':
                    for i in range(len(labels_tgt)): 
                        labels_tgt[i] = voc2coco[labels_tgt[i]]
                
                # input im
                img0 = torch.from_numpy(im.copy().transpose((2, 0, 1)))[None].float().cuda()
                pert = torch.zeros_like(img0)
                for i in range(max_iters):
                    pert.requires_grad = True
                    loss_joint = []
                    for model, model_train in zip(models_wb, models_train_wb):
                        # make sure all the data are on the same device
                        device = next(model.parameters()).device

                        data = get_test_data(model, im)
                        data_train = get_train_data(model, im, pert.to(device), data, bboxes_tgt, labels_tgt)
                        loss_dict = model_train(return_loss=True, **data_train)
                        losses = get_loss_from_dict(model, loss_dict)
                        if device.index == 1:
                            cuda0 = torch.device('cuda:0')
                            losses = losses.to(cuda0)
                        loss_joint.append(losses)
                    # loss_joint = sum(loss_joint)
                    loss_joint = ratio*loss_joint[0] + loss_joint[1]
                    loss_joint.backward()
                    with torch.no_grad():
                        pert = pert - alpha*torch.sign(pert.grad)
                        pert = pert.clamp(min=-eps, max=eps)
                pert = pert.squeeze().cpu().numpy().transpose(1, 2, 0)
                adv = (im + pert).clip(0, 255)                

                # save detection result of each attack plan
                detections = {}
                for idx, model in enumerate(models_wb):
                    detections[f'wb{idx}'] = get_det(model, adv, dataset)
                for idx, model in enumerate(models_bb):
                    detections[f'bb{idx}'] = get_det(model, adv, dataset)
                det_save_path = det_root / f"im{im_id}_idx{victim_idx}_f{victim_class}_t{target_class}_{attack_step}"
                np.save(det_save_path, detections)


                successes_wb = [int(is_success(detections[key], target_clean)) for key in detections if key[:2]=='wb']
                successes_bb = [int(is_success(detections[key], target_clean)) for key in detections if key[:2]=='bb']
                result = f"im{im_id}_idx{victim_idx}_f{victim_class}_t{target_class}_{attack_step}_wb{successes_wb}_bb{successes_bb}"
                # save the progress
                file = open(exp_root / f'{exp_name}.txt', 'a')
                file.write(f"{result}\n")
                file.close()

                # save perturbation
                pert_save_path = pert_root / result
                np.save(pert_save_path, pert)
                # save perturbation information
                file = open(exp_root / f'pert_info_{exp_name}.txt', 'a')
                linf = np.max(np.abs(pert))
                l2_dist = lambda a, b :  np.linalg.norm((a - b).ravel(), ord = 2)
                pert_info = f"linf {linf}\tl2 {l2_dist(adv/255.0,im/255.0):.6f}"
                file.write(f"{result}\t{pert_info}\n")
                file.close()

                # save adv fig
                fig_save_path = fig_root / f'{result}.png'
                show_figure_ap_det(im, adv, attack_plan, attack_step, detections, dataset, save_path=fig_save_path)
                
if __name__ == "__main__":
    main()