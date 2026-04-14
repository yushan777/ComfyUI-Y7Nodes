# Stripped-down variant of 1038lab/ComfyUI-JoyCaption (AILab)
# https://github.com/1038lab/ComfyUI-JoyCaption

import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps



# Loads a batch of images from a directory and returns them as a list of image
# tensors and a matching list of their full file paths. Supports jpg, jpeg, png,
# and webp. Images are EXIF-transposed and converted to RGB float32 tensors.
# Pair with CaptionSaver — the IMAGE_PATH output tells CaptionSaver where to
# write each .txt file, and IMAGE feeds into JoyCaption for captioning.
class Y7Nodes_ImageBatchPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_dir": ("STRING", {"default": "", "multiline": True, "placeholder": "Input directory containing images"}),
            },
            "optional": {
                # 0 = load all images in the directory
                "batch_size": ("INT", {"default": 0, "min": 0, "step": 1, "tooltip": "Number of images to load (0 = all)"}),
                # 1-based index of the first image to load (useful for resuming)
                "start_from": ("INT", {"default": 1, "min": 1, "step": 1, "tooltip": "Start from Nth image (1 = first)"}),
                # random forces re-evaluation on every run (IS_CHANGED returns NaN)
                "sort_method": (["sequential", "reverse", "random"], {"default": "sequential"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "IMAGE_PATH")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_image_batch"
    CATEGORY = "Y7Nodes/CaptionTools"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get("sort_method") == "random":
            return float("NaN")
        return hash(frozenset(kwargs))

    def load_image_batch(self, image_dir, batch_size=0, start_from=1, sort_method="sequential"):
        image_dir = os.path.expanduser(image_dir)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"Directory '{image_dir}' cannot be found.")

        valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        image_files = [f for f in os.listdir(image_dir) if Path(f).suffix.lower() in valid_extensions]

        if not image_files:
            raise FileNotFoundError(f"No valid images found in '{image_dir}'.")

        if sort_method == "sequential":
            image_files.sort()
        elif sort_method == "reverse":
            image_files.sort(reverse=True)
        elif sort_method == "random":
            import random
            random.shuffle(image_files)

        start_index = min(start_from - 1, len(image_files) - 1)
        image_files = image_files[start_index:]
        if batch_size > 0:
            image_files = image_files[:batch_size]

        images, image_paths = [], []
        for filename in image_files:
            img_path = os.path.join(image_dir, filename)
            try:
                image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
                image = torch.from_numpy(np.array(image).astype(np.float32) / 255.0)[None,]
                images.append(image)
                image_paths.append(img_path)
            except Exception as e:
                print(f"[ImageBatchPath] Error loading {filename}: {e}")

        return (images, image_paths)


# Saves a caption string as a .txt file next to the source image, using the
# same filename stem (e.g. cat.jpg -> cat.txt). Designed to be paired with
# ImageBatchPath: connect IMAGE_PATH here and STRING from JoyCaption.
# The overwrite toggle controls whether existing .txt files are replaced or
# suffixed with a counter (cat_01.txt, cat_02.txt, etc.).
# Compatible with any VLM node that outputs a STRING or list of STRINGs —
# not just JoyCaption. Examples: Florence2, MiniCPM, LLaVA, Qwen-VL, etc.
class Y7Nodes_CaptionSaver:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # The caption text to write to the .txt file.
                "string": ("STRING", {"forceInput": True}),
                # String path to the source image (e.g. from ImageBatchPath). The .txt file
                # is written to the same directory with the same stem (e.g. cat.jpg -> cat.txt).
                "image_path": ("STRING", {"forceInput": True}),
                "overwrite": ("BOOLEAN", {"default": True, "tooltip": "If false, appends a number to avoid overwriting existing files"}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_caption"
    CATEGORY = "Y7Nodes/CaptionTools"
    OUTPUT_NODE = True

    def _unique_path(self, base_path: Path) -> Path:
        if not base_path.exists():
            return base_path
        counter = 1
        while True:
            candidate = base_path.parent / f"{base_path.stem}_{counter:02d}{base_path.suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def save_caption(self, string, image_path, overwrite=True):
        try:
            image_path = Path(image_path)
            txt_path = image_path.with_suffix(".txt")

            if not overwrite:
                txt_path = self._unique_path(txt_path)

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(string)
            print(f"[CaptionSaver] Saved: {txt_path}")

        except Exception as e:
            print(f"[CaptionSaver] Error: {e}")

        return ()
