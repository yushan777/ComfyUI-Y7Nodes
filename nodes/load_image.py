import fnmatch
import os
import hashlib
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
from comfy_api.latest import io

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class Y7Nodes_LoadImage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = folder_paths.get_input_directory()

        exclude_files = {"Thumbs.db", "*.DS_Store", "desktop.ini", "*.lock"}
        exclude_folders = {"clipspace", ".*"}

        file_list = []
        for root, dirs, files in os.walk(input_dir, followlinks=True):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, ex) for ex in exclude_folders)]
            files = [f for f in files if not any(fnmatch.fnmatch(f, ex) for ex in exclude_files)]
            files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
            for file in files:
                relpath = os.path.relpath(os.path.join(root, file), start=input_dir)
                relpath = relpath.replace("\\", "/")
                file_list.append(relpath)

        return io.Schema(
            node_id="Y7Nodes_LoadImage",
            display_name="Y7 Load Image (subfolders)",
            category="Y7Nodes/image",
            inputs=[
                io.Combo.Input(
                    "image",
                    options=sorted(file_list),
                    upload=io.UploadType.image,
                ),
            ],
            outputs=[
                io.Image.Output("image"),
                io.Mask.Output("mask"),
            ],
        )

    @classmethod
    def execute(cls, image: str) -> io.NodeOutput:
        image_path = folder_paths.get_annotated_filepath(image)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)

        rgb = img.convert("RGB")
        image_tensor = torch.from_numpy(np.array(rgb).astype(np.float32) / 255.0)[None,]

        if "A" in img.getbands():
            mask = 1.0 - torch.from_numpy(np.array(img.getchannel("A")).astype(np.float32) / 255.0)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        return io.NodeOutput(image_tensor, mask.unsqueeze(0))

    @classmethod
    def fingerprint_inputs(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def validate_inputs(cls, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
