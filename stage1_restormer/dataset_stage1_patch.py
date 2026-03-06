import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class Stage1RainDatasetPatch(Dataset):
    def __init__(self, roots, patch_size=256, debug=False):
        self.samples = []
        self.patch = patch_size
        self.debug = debug

        for r in roots:
            domain = r["name"]
            root = r["path"]
            clean_name = r["clean_dir"]

            drop_dir = os.path.join(root, "Drop")
            blur_dir = os.path.join(root, "Blur")
            clean_dir = os.path.join(root, clean_name)

            if not os.path.exists(drop_dir):
                print(f"[Warning] Directory not found: {drop_dir}")
                continue

            scenes = sorted(os.listdir(drop_dir))
            for scene in scenes:
                drop_scene = os.path.join(drop_dir, scene)
                if not os.path.isdir(drop_scene):
                    continue

                frames = sorted(os.listdir(drop_scene))
                for frame in frames:
                    if not frame.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                        
                    self.samples.append({
                        "domain": domain,
                        "drop":  os.path.join(drop_dir, scene, frame),
                        "blur":  os.path.join(blur_dir, scene, frame),
                        "clean": os.path.join(clean_dir, scene, frame),
                    })

        if self.debug:
            print(f"[DEBUG] Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return img

    def _pad_if_needed(self, img, ps):
        c, h, w = img.shape
        pad_h = max(0, ps - h)
        pad_w = max(0, ps - w)
        
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect').squeeze(0)
        return img

    def _random_crop(self, drop, blur, clean):
        drop = self._pad_if_needed(drop, self.patch)
        blur = self._pad_if_needed(blur, self.patch)
        clean = self._pad_if_needed(clean, self.patch)

        _, H, W = drop.shape
        ps = self.patch

        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)

        drop  = drop[:,  top:top+ps, left:left+ps]
        blur  = blur[:,  top:top+ps, left:left+ps]
        clean = clean[:, top:top+ps, left:left+ps]

        return drop, blur, clean, top, left

    def __getitem__(self, idx):
        s = self.samples[idx]

        try:
            drop  = self._load_image(s["drop"])
            blur  = self._load_image(s["blur"])
            clean = self._load_image(s["clean"])

            drop, blur, clean, top, left = self._random_crop(drop, blur, clean)
            
            if random.random() > 0.5:
                drop = torch.flip(drop, dims=[2])
                blur = torch.flip(blur, dims=[2])
                clean = torch.flip(clean, dims=[2])

            return drop, blur, clean
        except Exception as e:
            print(f"Error loading index {idx}: {s['drop']} - {e}")
            raise e