
# Y7Nodes Image Paste Crop by Location

# A variant of WAS_Image_Paste_Crop_Location where 'right' and 'bottom' are
# offsets inward from the right and bottom edges of the image, rather than
# absolute pixel coordinates from the top-left.

# So the paste region is defined as:
#   x1 = left
#   y1 = top
#   x2 = img_width  - right
#   y2 = img_height - bottom

from PIL import Image, ImageDraw, ImageFilter
import torch
import numpy as np


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a ComfyUI IMAGE tensor (B,H,W,C) to a PIL Image (first frame)."""
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a ComfyUI IMAGE tensor (1,H,W,C)."""
    return torch.from_numpy(
        np.array(image).astype(np.float32) / 255.0
    ).unsqueeze(0)


class Y7Nodes_PasteCroppedImageBack:
    """
    Paste a crop image onto a base image at a region defined by edge-relative
    coordinates.  'right' and 'bottom' are pixel offsets measured inward from
    the right and bottom edges of the base image respectively.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_orig": ("IMAGE", {
                    "tooltip": "Base image to paste onto"
                }),
                "image_crop": ("IMAGE", {
                    "tooltip": "Image to paste into the defined region (will be resized to fit)"
                }),
                "left": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1,
                    "tooltip": "Pixels from the left edge to the left of the paste region"
                }),
                "top": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1,
                    "tooltip": "Pixels from the top edge to the top of the paste region"
                }),
                "right": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1,
                    "tooltip": "Pixels inward from the RIGHT edge to the right of the paste region"
                }),
                "bottom": ("INT", {
                    "default": 0, "min": 0, "max": 10000000, "step": 1,
                    "tooltip": "Pixels inward from the BOTTOM edge to the bottom of the paste region"
                }),
                "crop_blending": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Blending/feathering amount at the edges of the pasted region"
                }),
                "crop_sharpening": ("INT", {
                    "default": 0, "min": 0, "max": 3, "step": 1,
                    "tooltip": "Number of sharpening passes applied to the crop before pasting"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "MASK")
    FUNCTION = "image_paste_crop_location"
    CATEGORY = "Y7Nodes/Image"

    DESCRIPTION = """
Paste Cropped Image Back

Pastes a crop image onto a base image at a region defined by edge-relative
coordinates. Unlike the WAS equivalent, 'right' and 'bottom' are offsets
measured inward from the RIGHT and BOTTOM edges of the base image.

Example — to paste into the bottom-right 256×256 corner of any image:
  top=0  left=0  right=256  bottom=256  (set top/left to img_height-256 / img_width-256 to be precise)

Paste region:
  x1 = left
  y1 = top
  x2 = image_width  - right
  y2 = image_height - bottom
"""

    def image_paste_crop_location(self, image_orig, image_crop,
                                  top=0, left=0, right=0, bottom=0,
                                  crop_blending=0.25, crop_sharpening=0):
        result_image, result_mask = self._paste_image(
            tensor2pil(image_orig), tensor2pil(image_crop),
            top, left, right, bottom,
            crop_blending, crop_sharpening
        )
        return (result_image, result_mask)

    def _paste_image(self, image: Image.Image, crop_image: Image.Image,
                     top=0, left=0, right=0, bottom=0,
                     blend_amount=0.25, sharpen_amount=0):

        image = image.convert("RGBA")
        crop_image = crop_image.convert("RGBA")

        img_width, img_height = image.size

        # Convert edge-relative right/bottom to absolute coordinates
        abs_right  = img_width  - right
        abs_bottom = img_height - bottom

        # Clamp everything to image bounds
        x1 = min(max(left,       0), img_width)
        y1 = min(max(top,        0), img_height)
        x2 = min(max(abs_right,  0), img_width)
        y2 = min(max(abs_bottom, 0), img_height)

        # Guard against degenerate regions
        if x2 <= x1 or y2 <= y1:
            return (pil2tensor(image), pil2tensor(Image.new("L", image.size, 0).convert("RGB")))

        crop_size = (x2 - x1, y2 - y1)
        crop_img = crop_image.resize(crop_size)
        crop_img = crop_img.convert("RGBA")

        if sharpen_amount > 0:
            for _ in range(sharpen_amount):
                crop_img = crop_img.filter(ImageFilter.SHARPEN)

        blend_amount = max(0.0, min(1.0, blend_amount))
        blend_ratio = (max(crop_size) / 2) * float(blend_amount)

        def inset_border(img, border_width, border_color):
            w, h = img.size
            bordered = Image.new(img.mode, (w, h), border_color)
            bordered.paste(img, (0, 0))
            draw = ImageDraw.Draw(bordered)
            draw.rectangle((0, 0, w - 1, h - 1), outline=border_color, width=border_width)
            return bordered

        blend = image.copy()
        mask = Image.new("L", image.size, 0)

        mask_block = Image.new("L", crop_size, 255)
        mask_block = inset_border(mask_block, int(blend_ratio / 2), 0)

        Image.Image.paste(mask, mask_block, (x1, y1))
        blend.paste(crop_img, (x1, y1), crop_img)

        mask = mask.filter(ImageFilter.BoxBlur(radius=blend_ratio / 4))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_ratio / 4))

        blend.putalpha(mask)
        image = Image.alpha_composite(image, blend)

        return (pil2tensor(image), pil2tensor(mask.convert("RGB")))
