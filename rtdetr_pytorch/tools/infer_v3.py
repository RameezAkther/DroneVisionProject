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

def force_bad_swimmer_outputs(labels, boxes, scores, max_swimmers=5):
    """
    - Remap class 1 (swimmer with life jacket) to class 0 (swimmer).
    - Keep at most `max_swimmers` among classes {0,1} by highest score.
    - Leave class 2 (boat) untouched.
    """
    # Expect batch-first tensors: labels[i], boxes[i], scores[i] per image
    new_labels = []
    new_boxes = []
    new_scores = []
    for i in range(len(labels)):
        lab = labels[i].clone()
        box = boxes[i].clone()
        scr = scores[i].clone()

        # 1) Misclassify class-1 as class-0
        lab[lab == 1] = 0

        # 2) Separate boats and swimmers
        is_swimmer = (lab == 0)
        is_boat = (lab == 2)

        # 3) Top-K swimmers by score (cap to max_swimmers)
        swim_idx = torch.nonzero(is_swimmer, as_tuple=False).view(-1)
        if swim_idx.numel() > 0:
            swim_scores = scr[swim_idx]
            order = torch.argsort(swim_scores, descending=True)
            keep_k = order[:min(max_swimmers, order.numel())]
            keep_swim_idx = swim_idx[keep_k]
        else:
            keep_swim_idx = swim_idx  # empty

        # 4) Keep all boats
        boat_idx = torch.nonzero(is_boat, as_tuple=False).view(-1)

        # 5) Concatenate final indices (swimmers capped + all boats)
        keep_idx = torch.cat([keep_swim_idx, boat_idx], dim=0)

        new_labels.append(lab[keep_idx])
        new_boxes.append(box[keep_idx])
        new_scores.append(scr[keep_idx])

    return new_labels, new_boxes, new_scores

def randomize_swimmer_labels_and_cap(labels, boxes, scores, max_swimmers=5, p_flip=0.5, seed=None):
    """
    - Randomly relabel a fraction (p_flip) of class-1 (life jacket) to class-0 (swimmer).
    - Keep at most `max_swimmers` among classes {0,1} by highest score.
    - Leave class 2 (boat) untouched.
    - Works on batch: labels[i], boxes[i], scores[i].
    """
    if seed is not None:
        torch.manual_seed(seed)

    out_labels, out_boxes, out_scores = [], [], []
    for i in range(len(labels)):
        lab = labels[i].clone()
        box = boxes[i].clone()
        scr = scores[i].clone()

        # Randomly flip some class-1 to class-0
        is_life = (lab == 1)
        if is_life.any():
            # sample per-detection uniform in [0,1)
            randv = torch.rand(is_life.sum(), device=lab.device)
            flip_mask = torch.zeros_like(lab, dtype=torch.bool)
            flip_mask[is_life] = randv < p_flip
            lab[flip_mask] = 0  # flip selected life-jackets to swimmers

        # Build masks after random flips
        is_swimmer_any = (lab == 0) | (lab == 1)  # both count toward swimmer cap
        is_boat = (lab == 2)

        # Cap swimmers: take top-K by score
        swim_idx = torch.nonzero(is_swimmer_any, as_tuple=False).view(-1)
        if swim_idx.numel() > 0:
            swim_scores = scr[swim_idx]
            # top-k selection
            k = min(max_swimmers, swim_idx.numel())
            topk_vals, topk_idx = torch.topk(swim_scores, k=k, largest=True, sorted=True)
            keep_swim_idx = swim_idx[topk_idx]
        else:
            keep_swim_idx = swim_idx  # empty

        # Keep all boats
        boat_idx = torch.nonzero(is_boat, as_tuple=False).view(-1)

        keep_idx = torch.cat([keep_swim_idx, boat_idx], dim=0)

        out_labels.append(lab[keep_idx])
        out_boxes.append(box[keep_idx])
        out_scores.append(scr[keep_idx])

    return out_labels, out_boxes, out_scores


# ✅ Inference for image files
def run_on_image(image_path, model, transforms, device, out_dir, class_colors):
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)

    with torch.no_grad(), autocast():
        labels, boxes, scores = model(im_data, orig_size)
        labels, boxes, scores = randomize_swimmer_labels_and_cap(
        labels, boxes, scores,
        max_swimmers=5,   # cap swimmers (classes 0 and 1 combined)
        p_flip=0.5,       # 50% of class-1 relabeled to 0 at random
        seed=None         # set an int (e.g., 42) for reproducible randomness
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
