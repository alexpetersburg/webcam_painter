import os
import warnings
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo
from human_seg.networks import init_model
import cv2
import numpy as np
from PIL import Image

from human_seg.utils.transforms import transform_logits, get_affine_transform

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False
try:
    from mmseg.apis import inference_segmentor, init_segmentor
    has_mmseg = True
except (ImportError, ModuleNotFoundError):
    has_mmseg = False


def get_seg_net(path, device):
    seg_model = init_model('resnet101', num_classes=7, pretrained=None)
    state_dict = torch.load(path)['state_dict']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    seg_model.load_state_dict(new_state_dict)
    seg_model.to(device)
    seg_model.eval()
    return seg_model


def infer_seg_model(model, img, transform, device):
    def _box2cs(box):
        x, y, w, h = box[:4]
        return _xywh2cs(x, y, w, h)

    def _xywh2cs(x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > 1 * h:
            h = w * 1.0 / 1
        elif w < 1 * h:
            w = h * 1
        scale = np.array([w, h], dtype=np.float32)
        return center, scale


    h, w, _ = img.shape
    person_center, s = _box2cs([0, 0, w - 1, h - 1])
    r = 0
    trans = get_affine_transform(person_center, s, r, [512,512])
    tensor = cv2.warpAffine(
        img,
        trans,
        (512, 512),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0))
    # tensor = Image.fromarray(tensor)
    tensor = transform(tensor).unsqueeze(0)
    output = model(tensor.to(device))
    upsample = torch.nn.Upsample(size=[512, 512], mode='bilinear', align_corners=True)
    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
    upsample_output = upsample_output.squeeze()
    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

    logits_result = transform_logits(upsample_output.data.cpu().numpy(), person_center, s, w, h, input_size=[512, 512])
    parsing_result = np.argmax(logits_result, axis=2)
    return parsing_result

def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('seg_config', help='Config file for pose')
    parser.add_argument('seg_checkpoint', help='Checkpoint file for seg')
    parser.add_argument('--video_path', type=str, default='', help='Image file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--out-img-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'
    assert has_mmdet, 'Please install mmseg to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_img_root != '')
    assert args.video_path != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    # seg_model = init_segmentor(args.seg_config, args.seg_checkpoint, device=args.device.lower())
    seg_model = get_seg_net(args.seg_checkpoint, args.device.lower())
    seg_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    ])

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    new_background = cv2.imread('demo/back.jpeg')
    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    # Read until video is completed
    # cap.set(1, 80)
    ret, frame = cap.read()
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter('demo.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                                                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    new_background = cv2.resize(new_background, frame.shape[:2][::-1])
    to_draw = np.zeros(frame.shape)
    draw_steps = 0
    i = 0
    while (cap.isOpened()):
        cap.set(1, i)
        i += 1
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            seg_result = infer_seg_model(seg_model, frame, seg_transform, args.device.lower())
            alpha_mask = np.zeros([*seg_result.shape, 3], dtype=np.uint8)
            alpha_mask[..., 0] = seg_result != 0
            alpha_mask[..., 1] = seg_result != 0
            alpha_mask[..., 2] = seg_result != 0

            result = new_background - new_background * alpha_mask + frame * alpha_mask
            # test a single image, the resulting box is (x1, y1, x2, y2)
            mmdet_results = inference_detector(det_model, frame)

            # keep the person class bounding boxes.
            person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

            # test a single image, with a list of bboxes.

            # optional
            return_heatmap = False

            # e.g. use ('backbone', ) to return backbone feature
            output_layer_names = None

            pose_results, returned_outputs = inference_top_down_pose_model(
                pose_model,
                frame,
                person_results,
                bbox_thr=args.bbox_thr,
                format='xyxy',
                dataset=dataset,
                dataset_info=dataset_info,
                return_heatmap=return_heatmap,
                outputs=output_layer_names)

            for hand_res in pose_results:
                keypoints = hand_res["keypoints"]
                finger_1 = keypoints[8]
                finger_2 = keypoints[4]
                finger_1_coord = (finger_1[:2].astype(int))
                finger_2_coord = (finger_2[:2].astype(int))
                if np.linalg.norm(finger_1_coord-finger_2_coord) < 35:
                    draw_steps += 1
                    if draw_steps > 2:
                        to_draw = cv2.circle(to_draw, ((finger_1_coord+finger_2_coord)/2).astype(int),
                                             radius=25, color=(255, 255, 255), thickness=-1)
                else:
                    draw_steps = 1
            result = (result.astype(int) + to_draw.astype(int)).clip(0, 255).astype(np.uint8)


            # show the results
            result = vis_pose_result(
                                    pose_model,
                                    result,
                                    pose_results,
                                    dataset=dataset,
                                    dataset_info=dataset_info,
                                    kpt_score_thr=args.kpt_thr,
                                    radius=args.radius,
                                    thickness=args.thickness)
            # if args.show:
            #     cv2.imshow('result', result)
            #     c = cv2.waitKey(1)
            out.write(result)
        else:
            cap.release()
            cv2.destroyAllWindows()
            break
    out.release()





if __name__ == '__main__':
    main()
