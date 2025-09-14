import os
import cv2
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig

# ✅ Assign specific colors for the 3 classes
def get_color_map():
    # 3 classes only: 0 → swimmer, 1 → swimmer with life jacket, 2 → boat
    return {
        0: (0, 255, 0),    # swimmer → green
        1: (255, 0, 0),    # swimmer with life jacket → red
        2: (0, 0, 255),    # boat → blue
    }

# ✅ Adaptive bbox thickness based on box size
def get_bbox_thickness(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    size = max(width, height)
    if size < 50:
        return 1
    elif size < 150:
        return 2
    elif size < 300:
        return 3
    else:
        return 4

def nms_iou_filter(boxes, scores, labels, iou_thresh=0.7):
    # Optional: simple per-class NMS to prune swimmers more
    # Uses torchvision.ops.nms if available
    import torchvision.ops as ops
    keep_indices = []
    classes = labels.unique()
    for c in classes:
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            continue
        k = ops.nms(boxes[idx], scores[idx], iou_thresh)
        keep_indices.append(idx[k])
    if len(keep_indices) == 0:
        return boxes[:0], scores[:0], labels[:0]
    keep_indices = torch.cat(keep_indices, dim=0)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def post_filter_predictions(labels, boxes, scores, max_swimmers=4,
                            thr_swimmer=0.9, thr_boat=0.6, use_extra_nms=False):
    """
    Make predictions intentionally worse:
    - Strongly suppress swimmers (class 0) and swimmers-with-life-jacket (class 1)
    - Keep at most `max_swimmers` across classes {0,1}, by highest score
    - Keep boats (class 2) with a normal threshold
    - Optional extra NMS for swimmers to drop more boxes
    """
    # Expect tensors shaped [N] or [N,4]
    device = boxes.device
    labels = labels.to(device)
    boxes = boxes.to(device)
    scores = scores.to(device)

    # Split by class
    is_swimmer = (labels == 0)
    is_lifejacket = (labels == 1)
    is_boat = (labels == 2)

    # Thresholds per group
    keep_swimmer = is_swimmer & (scores >= thr_swimmer)
    keep_lifejacket = is_lifejacket & (scores >= thr_swimmer)
    keep_boat = is_boat & (scores >= thr_boat)

    # Gather swimmers combined
    swimmer_mask = keep_swimmer | keep_lifejacket
    swimmer_idx = swimmer_mask.nonzero(as_tuple=False).view(-1)

    # Optional extra NMS to further prune swimmers
    if use_extra_nms and swimmer_idx.numel() > 0:
        b_sw = boxes[swimmer_idx]
        s_sw = scores[swimmer_idx]
        l_sw = labels[swimmer_idx]
        b_sw, s_sw, l_sw = nms_iou_filter(b_sw, s_sw, l_sw, iou_thresh=0.7)
        # Rebuild swimmer_idx from filtered tensors
        # We'll map back by ranking the kept swimmers within original swimmers
        # Simpler: directly sort these tensors here and ignore mapping
        # Take top-K swimmers
        if s_sw.numel() > max_swimmers:
            topk = torch.topk(s_sw, k=max_swimmers).indices
            b_sw = b_sw[topk]
            s_sw = s_sw[topk]
            l_sw = l_sw[topk]
        # Boats kept separately
        b_bo = boxes[keep_boat]
        s_bo = scores[keep_boat]
        l_bo = labels[keep_boat]
        # Concatenate and return
        return torch.cat([l_sw, l_bo], dim=0), torch.cat([b_sw, b_bo], dim=0), torch.cat([s_sw, s_bo], dim=0)

    # No extra NMS path: just top-K cap on swimmers
    if swimmer_idx.numel() > 0:
        s_sw_all = scores[swimmer_idx]
        # Sort swimmers by descending score
        order = torch.argsort(s_sw_all, descending=True)
        if order.numel() > max_swimmers:
            order = order[:max_swimmers]
        # Build final keep mask
        keep_final = torch.zeros_like(labels, dtype=torch.bool)
        keep_final[swimmer_idx[order]] = True
    else:
        keep_final = torch.zeros_like(labels, dtype=torch.bool)

    # Keep boats that passed threshold
    keep_final = keep_final | keep_boat

    # Apply final mask
    return labels[keep_final], boxes[keep_final], scores[keep_final]

# ✅ Draw function with adaptive thickness and specific colors
def draw(images, labels, boxes, scores, class_colors, thrh=0.6, path=""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            cls_id = int(lab[j].item())
            color = class_colors.get(cls_id, (255, 255, 255))  # fallback white
            thickness = get_bbox_thickness(b)
            for t in range(thickness):
                draw.rectangle([b[0]-t, b[1]-t, b[2]+t, b[3]+t], outline=color)
            draw.text((b[0], b[1]), text=f"ID:{cls_id} {round(scrs[j].item(),2)}",
                      font=ImageFont.load_default(), fill=color)

        if path:
            im.save(path)
        else:
            im.save(f'results_{i}.jpg')

# ✅ Inference for image files
def run_on_image(image_path, model, transforms, device, out_dir, class_colors):
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)

    with torch.no_grad(), autocast():
        labels, boxes, scores = model(im_data, orig_size)
        # Make it intentionally bad: cap swimmers <= 4, harsh threshold for swimmers
        labels, boxes, scores = post_filter_predictions(
            labels, boxes, scores,
            max_swimmers=4, thr_swimmer=0.9, thr_boat=0.6, use_extra_nms=True
        )

    out_path = os.path.join(out_dir, f"inferenced_{os.path.basename(image_path)}")
    draw([im_pil], labels, boxes, scores, class_colors, 0.6, path=out_path)
    print(f"✅ Saved {out_path}")


# ✅ Inference for video files
def run_on_video(video_path, model, transforms, device, out_dir, class_colors):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(out_dir, f"inferenced_{os.path.basename(video_path)}")
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        im_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(device)
        im_data = transforms(im_pil)[None].to(device)

        with torch.no_grad(), autocast():
            labels, boxes, scores = model(im_data, orig_size)
            # Intentionally degrade detections per frame
            labels, boxes, scores = post_filter_predictions(
                labels, boxes, scores,
                max_swimmers=3, thr_swimmer=0.9, thr_boat=0.6, use_extra_nms=True
            )

        draw([im_pil], labels, boxes, scores, class_colors, 0.6, path="tmp.jpg")
        result_frame = cv2.imread("tmp.jpg")
        out.write(result_frame)

    cap.release()
    out.release()
    print(f"✅ Saved {out_path}")

# ✅ Main function handling folder input
def main(args):
    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu')
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])

    os.makedirs(args.out_dir, exist_ok=True)
    class_colors = get_color_map()

    # ✅ Process all files in folder
    if args.folder:
        files = os.listdir(args.folder)
        for f in files:
            file_path = os.path.join(args.folder, f)
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                run_on_image(file_path, model, transforms, args.device, args.out_dir, class_colors)
            elif f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                run_on_video(file_path, model, transforms, args.device, args.out_dir, class_colors)
    else:
        print("⚠️ Please provide a folder with --folder")

# ✅ Argument parsing and script entry point
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('--folder', type=str, help="Folder containing images/videos")
    parser.add_argument('--out-dir', type=str, default="outputs", help="Output directory")
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
