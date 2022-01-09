import re
import sys
import argparse
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import numpy as np

from utils_attack import is_success
mmdet_root = Path('../detectors/mmdetection')
sys.path.insert(0, str(mmdet_root))
from mmdet.apis import init_detector
sys.path.append('../detectors/')
from utils_mmdet import get_det
from mmdet_model_info import model_info


def main():
    parser = argparse.ArgumentParser(description="test perturbations on different blackbox models")
    parser.add_argument("--eps", nargs="?", default=30, help="perturbation level: 10,20,30")
    parser.add_argument("--gpu", nargs="?", default=0, help="GPU ID: 0,1")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument("--dataset", nargs="?", default='voc', help="model dataset 'voc' or 'coco'. This will change the output range of detectors.")
    args = parser.parse_args()

    eps = int(args.eps)
    gpu_id = int(args.gpu)
    result_folder = args.root
    dataset = args.dataset
    device = f'cuda:{gpu_id}'

    # fix the issue with "CUDA illegal memory"
    if gpu_id == 1:
        import torch
        torch.cuda.set_device(1)

    # load models
    models = []
    model_list = ['Libra R-CNN', 'FCOS', 'FoveaBox', 'FreeAnchor', 'DETR', 'Deformable DETR']
    print(f"using {len(model_list)} bb models: \n{model_list}")
    for model_name in model_list:
        config_file = model_info[model_name]['config_file']
        checkpoint_file = model_info[model_name]['checkpoint_file']
        config_file = str(mmdet_root/config_file)
        checkpoint_file = str(mmdet_root/checkpoint_file)
        models.append(init_detector(config_file, checkpoint_file, device=device))

    # save result to txt file
    exp = f'run_sequential_attack_eps{eps}'
    result_root = Path(f"{result_folder}/") / exp
    pert_root = result_root / "pert"
    ap_root = result_root / "ap"
    det_bb_root = result_root / "det_bb" 
    det_bb_root.mkdir(parents=True, exist_ok=True)
    wb_txt_file = open(result_root / f'{exp}.txt', 'r')
    bb_txt_file = open(result_root / f'{exp}_bb.txt', 'a')

    if dataset == 'voc':
        data_root = Path("../data/VOC/VOC2007/JPEGImages/")
    else:
        data_root = Path("../data/COCO/val2017/")

    wb_txt = wb_txt_file.readlines()
    for line in tqdm(wb_txt):
        if line.startswith('im'):
            line = line.strip() # remove \n
            im_id = re.findall(r"im(.+?)\_", line)[0]
            victim_idx = re.findall(r"idx(.+?)\_", line)[0]
            from_class = re.findall(r"f(.+?)\_", line)[0]
            to_class = re.findall(r"t(.+?)\_", line)[0]
            ap = int(re.findall(r"ap(.+?)\_", line)[0])
            ap_name = f"im{im_id}_idx{victim_idx}_f{from_class}_t{to_class}_ap{ap}"

            # load perturbation
            pert_path = pert_root / f"{line}.npy"
            pert = np.load(pert_path)

            # load clean image
            img_path = data_root / f"{im_id}.jpg"
            im = np.array(Image.open(img_path).convert('RGB'))
            adv = (im + pert).clip(0, 255)

            # read attack plans
            attack_plan = np.load(ap_root/f"im{im_id}_idx{victim_idx}_f{from_class}_t{to_class}.npy",allow_pickle=True).item()
            target_clean = attack_plan['ap0'][:1]

            # save detections
            detections = {}
            for idx, model in enumerate(models):
                detections[f'bb{idx}'] = get_det(model, adv, dataset)
            det_save_path = det_bb_root / ap_name
            np.save(det_save_path, detections)

            # get blackbox results
            bb_successes = [int(is_success(detections[key], target_clean)) for key in detections]
            bb_txt_file.write(f"{ap_name}_wb[]_bb{bb_successes}\n")

    bb_txt_file.close()

if __name__ == '__main__':
    main()