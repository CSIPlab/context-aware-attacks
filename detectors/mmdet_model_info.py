# 'Faster R-CNN', 'RetinaNet', 'SSD', 'YOLOv3', 'GN+WS', 'Libra R-CNN', 'FCOS', 'FoveaBox', 'FreeAnchor', 'DETR'

model_info = {
    'Faster R-CNN': {
        'config_file': 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    },
    'RetinaNet': {
        'config_file': 'configs/retinanet/retinanet_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_1x_coco/retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    },
    'SSD': {
        'config_file': 'configs/ssd/ssd512_coco.py',
        'checkpoint_file': 'checkpoints/ssd512_coco_20200308-038c5591.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/ssd/ssd512_coco/ssd512_coco_20200308-038c5591.pth'
    },
    'YOLOv3': {
        'config_file': 'configs/yolo/yolov3_d53_mstrain-416_273e_coco.py',
        'checkpoint_file': 'checkpoints/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-416_273e_coco/yolov3_d53_mstrain-416_273e_coco-2b60fcd9.pth'
    },
    'GN+WS': {
        'config_file': 'configs/gn+ws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco.py',
        'checkpoint_file': 'checkpoints/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/gn%2Bws/faster_rcnn_r50_fpn_gn_ws-all_1x_coco/faster_rcnn_r50_fpn_gn_ws-all_1x_coco_20200130-613d9fe2.pth'
    },
    'Libra R-CNN': {
        'config_file': 'configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_coco/libra_faster_rcnn_r50_fpn_1x_coco_20200130-3afee3a9.pth'
    },
    'FCOS': {
        'config_file': 'configs/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py',
        'checkpoint_file': 'checkpoints/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth',
        'download_link': 'https://openmmlab.oss-cn-hangzhou.aliyuncs.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco/fcos_r50_caffe_fpn_gn-head_mstrain_640-800_2x_coco-d92ceeea.pth'
    },
    'FoveaBox': {
        'config_file': 'configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py',
        'checkpoint_file': 'checkpoints/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/foveabox/fovea_r50_fpn_4x4_1x_coco/fovea_r50_fpn_4x4_1x_coco_20200219-ee4d5303.pth'
    },
    'FreeAnchor': {
        'config_file': 'configs/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco.py',
        'checkpoint_file': 'checkpoints/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth',
        'download_link': 'http://download.openmmlab.com/mmdetection/v2.0/free_anchor/retinanet_free_anchor_r50_fpn_1x_coco/retinanet_free_anchor_r50_fpn_1x_coco_20200130-0f67375f.pth'
    },
    'DETR': {
        'config_file': 'configs/detr/detr_r50_8x2_150e_coco.py',
        'checkpoint_file': 'checkpoints/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/detr/detr_r50_8x2_150e_coco/detr_r50_8x2_150e_coco_20201130_194835-2c4b8974.pth'
    },
    'Deformable DETR': {
        'config_file': 'configs/deformable_detr/deformable_detr_r50_16x2_50e_coco.py',
        'checkpoint_file': 'checkpoints/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth',
        'download_link': 'https://download.openmmlab.com/mmdetection/v2.0/deformable_detr/deformable_detr_r50_16x2_50e_coco/deformable_detr_r50_16x2_50e_coco_20210419_220030-a12b9512.pth'
    }
}


def main():
    import urllib.request
    from pathlib import Path

    checkpoints_root = Path('mmdetection/checkpoints')
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    existing_files = list(checkpoints_root.glob('*.pth'))
    existing_files = [file.name for file in existing_files]

    for idx,model_name in enumerate(model_info):
        url = model_info[model_name]['download_link']
        file_name = url.split('/')[-1]
        if file_name in existing_files:
            print(f"{model_name} already exist, {idx+1}/{len(model_info)}")
            continue
        print(f'downloading {model_name} {idx+1}/{len(model_info)}')
        file_data = urllib.request.urlopen(url).read()
        with open(checkpoints_root / file_name, 'wb') as f:
            f.write(file_data)


if __name__ == "__main__":
    main()