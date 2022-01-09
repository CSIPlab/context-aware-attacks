# Context-Aware Adversarial Attacks
Implementation of AAAI 2022 paper *Context-Aware Transfer Attacks for Object Detection*


[Paper](http://arxiv.org/abs/2112.03223) | [Data and Code](https://github.com/CSIPlab/context-aware-attacks) are available.

[Context-Aware Transfer Attacks for Object Detection](http://arxiv.org/abs/2112.03223)  
 Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song,Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury,
 [M. Salman Asif](https://intra.ece.ucr.edu/~sasif/)<br>
 UC Riverside 


## Environment
See `requirements.txt`, some key dependencies are:
```
python==3.7
torch==1.7.0 
torchvision==0.8.1
mmcv-full==1.3.3
...
```

Install mmcv-full https://github.com/open-mmlab/mmcv.

```
pip install mmcv-full==1.3.3 -f https://download.openmmlab.com/mmcv/dist/{cu_version}/torch1.7.0/index.html
# depending on your cuda version
```

## Datasets
Get VOC and COCO datasets under `/data` folder.
```
cd data
bash get_voc.sh
bash get_coco.sh
```

## Object Detection Models
Get mmdetection code repo and download pretrained models.
```
cd detectors
git clone https://github.com/zikuicai/mmdetection
# This will download mmdetection package to detectors/mmdetection/

python mmdet_model_info.py
# This will download checkpoin files into detectors/mmdetection/checkpoints
```

## Attacks and Evaluation
Run sequential attack.
```
cd attacks/attack_mmdetection
python run_sequential_attack.py
```

Calculate fooling rate.
```
cd evaluate/fooling_rate
python get_fooling_rate.py
```

Run transfer attacks on different blackbox models.
```
cd attacks/attack_mmdetection
python run_transfer_attack.py
```

Calculate fooling rate again on blackbox results.
```
cd evaluate/fooling_rate
python get_fooling_rate.py -bb
```

## Overview of Code Structure
- data
    - script to download datasets VOC and COCO
    - indices of images used in our experiments   
- detectors
    - packages for object detectors
    - script to download the pretrained model weights
    - util and visualization functions for mmdetection models
- context
    - co-occurrence matrix
    - distance matrix
    - size matrix
- attacks
    - code to attack the detectors
    - code to transfer attack other blackbox detectors
- evaluate
    - code to calculate the fooling rate of whitebox and blackbox attacks


## Citation
If you find this code repo helpful, please cite:
```
@article{cai2021contextaware,
  title={Context-Aware Transfer Attacks for Object Detection},
  author={Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song,Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif},
  journal = {arXiv preprint arXiv:2112.03223},
  year      = {2021},
}
```