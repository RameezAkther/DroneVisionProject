import os
import cv2
import random
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.cuda.amp import autocast
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from src.core import YAMLConfig


# ✅ Assign unique colors for each class
def get_color_map(num_classes=80, seed=42):
    random.seed(seed)
    colors = []
    for _ in range(num_classes):
        colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    return colors


# ✅ Modified draw function with class-specific colors
def draw(images, labels, boxes, scores, class_colors, thrh=0.6, path=""):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j, b in enumerate(box):
            cls_id = int(lab[j].item())
            color = class_colors[cls_id % len(class_colors)]
            draw.rectangle(list(b), outline=color, width=3)
            draw.text((b[0], b[1]), text=f"ID:{cls_id} {round(scrs[j].item(),2)}",
                      font=ImageFont.load_default(), fill=color)

        if path:
            im.save(path)
        else:
            im.save(f'results_{i}.jpg')


def run_on_image(image_path, model, transforms, device, out_dir, class_colors):
    im_pil = Image.open(image_path).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)
    im_data = transforms(im_pil)[None].to(device)

    with torch.no_grad(), autocast():
        labels, boxes, scores = model(im_data, orig_size)

    out_path = os.path.join(out_dir, f"inferenced_{os.path.basename(image_path)}")
    draw([im_pil], labels, boxes, scores, class_colors, 0.6, path=out_path)
    print(f"✅ Saved {out_path}")


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
    class_colors = get_color_map(num_classes=91)  # COCO has 91 classes

    # ✅ Handle folder input
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
