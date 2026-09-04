# All-in-one image-editing node for Flux.2 Klein workflows for 1 Reference Image.
#
# Normally, setting up a Klein edit means wiring together several separate nodes: load the
# source image, optionally downscale it, VAE-encode it for the reference latent, process the
# painted mask (binarize/expand/feather), VAE-encode a masked version of the image for
# inpaint-style concat conditioning, and finally patch reference_latents/concat_latent_image/
# concat_mask onto the positive and negative conditioning. This node compacts all of that into
# a single node: pick an image (with its mask painted via the built-in mask editor), tweak a
# few widgets, and get back a ready-to-sample reference latent plus patched conditioning.
#
# Model coverage: this should work with all flux.2-klein variants (base, distilled, 4B/8B text
# encoder, etc.). ComfyUI routes every Klein/Flux.2 checkpoint through the same Flux2 model class
# (comfy/model_base.py, comfy/supported_models.py), which reads reference_latents/
# concat_latent_image/concat_mask identically regardless of backbone size or distillation -
# those variants only differ in checkpoint weights and text encoder (Qwen3-4B/8B vs Mistral3-24B
# for full Flux.2), which ComfyUI auto-detects. Verified by reading that source and exercising
# this node's schema/execute logic with a mock VAE, not by running inference against every real
# checkpoint.
import fnmatch
import math
import os

import torch
import torch.nn.functional as F

import folder_paths
import node_helpers
import nodes
from comfy_api.latest import io

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class Y7Nodes_Flux2KleinEdit_Ref1(io.ComfyNode):
    """Loads an image directly on-node and prepares minimal Klein edit conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        input_dir = folder_paths.get_input_directory()
        exclude_files = {"Thumbs.db", "*.DS_Store", "desktop.ini", "*.lock"}
        exclude_folders = {"clipspace", ".*"}

        file_list = []
        for root, dirs, files in os.walk(input_dir, followlinks=True):
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pat) for pat in exclude_folders)]
            files = [f for f in files if not any(fnmatch.fnmatch(f, pat) for pat in exclude_files)]
            files = [f for f in files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
            for file in files:
                relpath = os.path.relpath(os.path.join(root, file), start=input_dir)
                file_list.append(relpath.replace("\\", "/"))

        return io.Schema(
            node_id="Y7Nodes_Flux2KleinEdit_Ref1",
            display_name="Y7 Flux.2 Klein Edit Ref 1",
            category="Y7Nodes/Klein",
            description="Loads an image directly on the node and prepares minimal Klein edit conditioning "
                         "(reference latent, plus optional mask-driven inpaint conditioning).",
            inputs=[
                # VAE used to encode the uploaded (and optionally masked) image into the reference/concat latents.
                io.Vae.Input("vae"),
                # Image to edit, picked from the ComfyUI input folder; shows an upload button + preview in the UI.
                # Any mask painted on it (via the built-in mask editor) drives the optional inpaint conditioning below.
                # Listed first so it is the top-most widget on the node.
                # This input MUST be named "image": on save, the mask editor writes a new clipspace-*.png and points
                # the node at it via ComfyApp.pasteFromClipspace, which locates the widget with
                # `widgets.findIndex(w => w.name === "image")` - by name, not position. Under any other name the
                # write-back silently no-ops, the widget keeps the previous filename, and execute() reads an image
                # whose painted mask is missing.
                io.Combo.Input("image", options=sorted(file_list), upload=io.UploadType.image),
                # Shrinks the image (and its mask) before encoding, clamped to 0.25-1.0. Useful to cut VRAM/latent
                # size for large source images; 1.0 leaves the image at its original resolution.
                io.Float.Input(
                    "downscale_factor", default=1.0, min=0.25, max=1.0, step=0.05,
                    tooltip="Downscale the source image before encoding (1.0 = no downscale).",
                ),
                # Flux.2 works best on dimensions that are multiples of 16; the image-edit pipeline silently
                # rounds odd sizes down anyway. When on, the image (and its mask) is centre-cropped down to the
                # nearest multiple of 16 before encoding, so what is encoded matches what the model sees.
                # No-op when the image is already aligned.
                io.Boolean.Input(
                    "crop_2_nearest_16px", default=True,
                    tooltip="Centre-crop the image (and mask) down to the nearest multiple of 16, which Flux.2 "
                            "prefers. No-op if the dimensions are already multiples of 16.",
                ),
                # Grows the mask outward by this many pixels (dilation) so the edit region covers a bit more
                # than what was painted. 0 disables expansion.
                io.Int.Input(
                    "expand_mask", default=0, min=0, max=256, step=1,
                    tooltip="Dilate (expand) the mask by this many pixels.",
                ),
                # Softens the mask edges with a Gaussian blur of this radius, for a smoother blend between the
                # edited and preserved regions. 0 disables feathering.
                io.Int.Input(
                    "feather_mask", default=0, min=0, max=256, step=1,
                    tooltip="Feather (Gaussian blur) the mask by this radius.",
                ),
                # Applied last, after expand/feather, so it always produces a hard-edged mask: everything at or
                # above 0.5 becomes fully opaque, everything below fully transparent. Feathering still shapes the
                # edge (rounding corners, smoothing jagged strokes) before the cut, but leaves no grey ramp.
                io.Boolean.Input(
                    "binary_mask", default=False,
                    tooltip="Hard-threshold the finished mask to pure black/white (cut at 0.5). Applied after "
                            "expand/feather, so the result is always crisp with no soft edge.",
                ),
                # Optional positive conditioning to extend with the reference latent / mask-based concat data.
                # Passed through untouched (as an empty list) if not connected.
                io.Conditioning.Input("positive", optional=True),
                # Optional negative conditioning, extended the same way as positive.
                io.Conditioning.Input("negative", optional=True),
            ],
            outputs=[
                # VAE-encoded latent of the (downscaled) source image, for use as an edit-model reference latent.
                io.Latent.Output(display_name="reference_latent"),
                # Positive conditioning with reference_latents/concat_latent_image (and concat_mask if a mask was used) set.
                io.Conditioning.Output(display_name="positive"),
                # Negative conditioning, extended the same way as positive.
                io.Conditioning.Output(display_name="negative"),
                # The (possibly downscaled) source image, for on-canvas preview.
                io.Image.Output(display_name="preview_image"),
                # The processed mask (after binary/expand/feather), for on-canvas preview. All-zero if no mask was painted.
                io.Mask.Output(display_name="preview_mask"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae,
        image,
        downscale_factor=1.0,
        crop_2_nearest_16px=True,
        expand_mask=0,
        feather_mask=0,
        binary_mask=False,
        positive=None,
        negative=None,
    ) -> io.NodeOutput:
        # `image` arrives as the selected filename and is rebound to the loaded tensor.
        image, mask = nodes.LoadImage().load_image(image)
        if downscale_factor is None:
            downscale_factor = 1.0
        try:
            downscale_factor = float(downscale_factor)
        except (TypeError, ValueError):
            downscale_factor = 1.0
        if not math.isfinite(downscale_factor):
            downscale_factor = 1.0
        downscale_factor = max(0.25, min(1.0, downscale_factor))

        # When cropping to multiples of 16 is requested, snap the downscale target to 16 as well, so the
        # crop below has nothing left to trim instead of shaving another 8 pixels off a just-resized image.
        align = 16 if crop_2_nearest_16px else 8
        if downscale_factor < 1.0:
            scaled_width = max(align, int((image.shape[2] * downscale_factor) // align) * align)
            scaled_height = max(align, int((image.shape[1] * downscale_factor) // align) * align)
            image = F.interpolate(
                image.movedim(-1, 1),
                size=(scaled_height, scaled_width),
                mode="bilinear",
                align_corners=False,
            ).movedim(1, -1)
            if mask is not None and torch.count_nonzero(mask) > 0:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(scaled_height, scaled_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        if crop_2_nearest_16px:
            image, mask = cls._crop_to_multiple(image, mask, 16)

        pixels = image.clone()

        latent = vae.encode(pixels[:, :, :, :3])
        result = {"samples": latent}

        conditioned_image = pixels[:, :, :, :3]
        mask_resized = None

        if mask is not None and torch.count_nonzero(mask) > 0:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = cls._process_mask(mask, expand_mask, feather_mask, binary_mask)
        else:
            mask = None

        if mask is not None:
            mask_image = F.interpolate(
                mask.unsqueeze(1),
                size=(pixels.shape[1], pixels.shape[2]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).clamp(0.0, 1.0)

            # For edit/inpaint conditioning, masked regions are replaced with neutral gray
            # so Klein gets the preserved context outside the editable area.
            conditioned_image = ((conditioned_image - 0.5) * (1.0 - mask_image.unsqueeze(-1))) + 0.5

            mask_resized = F.interpolate(
                mask.unsqueeze(1),
                size=(latent.shape[2], latent.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1).clamp(0.0, 1.0)
            result["noise_mask"] = mask_resized

        concat_latent_image = vae.encode(conditioned_image)

        if positive is not None:
            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image},
            )
            positive = node_helpers.conditioning_set_values(
                positive,
                {"reference_latents": [latent]},
                append=True,
            )
            if mask is not None:
                positive = node_helpers.conditioning_set_values(positive, {"concat_mask": mask_resized})

        if negative is not None:
            negative = node_helpers.conditioning_set_values(
                negative,
                {"concat_latent_image": concat_latent_image},
            )
            negative = node_helpers.conditioning_set_values(
                negative,
                {"reference_latents": [latent]},
                append=True,
            )
            if mask is not None:
                negative = node_helpers.conditioning_set_values(negative, {"concat_mask": mask_resized})

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        preview_mask = mask if mask is not None else torch.zeros((1, pixels.shape[1], pixels.shape[2]), dtype=torch.float32)

        return io.NodeOutput(result, positive, negative, image, preview_mask)

    @staticmethod
    def _crop_to_multiple(image, mask, multiple=16):
        """Centre-crop image (B,H,W,C) and its mask down to the nearest multiple of `multiple`."""
        height, width = image.shape[1], image.shape[2]
        new_height = (height // multiple) * multiple
        new_width = (width // multiple) * multiple

        # Already aligned, or too small to crop without losing the whole image: leave everything alone.
        if (new_height, new_width) == (height, width) or new_height < multiple or new_width < multiple:
            return image, mask

        top = (height - new_height) // 2
        left = (width - new_width) // 2
        image = image[:, top:top + new_height, left:left + new_width, :]

        # LoadImage hands back a 64x64 placeholder mask for images without alpha, so only crop a mask
        # that actually lines up with the image.
        if mask is not None and mask.shape[-2] == height and mask.shape[-1] == width:
            mask = mask[..., top:top + new_height, left:left + new_width]

        return image, mask

    @staticmethod
    def _process_mask(mask, expand_pixels, feather_radius, binary_mask):
        result = mask.clone()

        if expand_pixels > 0:
            kernel_size = expand_pixels * 2 + 1
            result = result.unsqueeze(1)
            result = F.max_pool2d(result, kernel_size=kernel_size, stride=1, padding=expand_pixels)
            result = result.squeeze(1)

        if feather_radius > 0:
            kernel_size = feather_radius * 2 + 1
            sigma = feather_radius / 3.0
            x = torch.arange(kernel_size, dtype=torch.float32, device=result.device) - feather_radius
            gauss_1d = torch.exp(-0.5 * (x / sigma) ** 2)
            gauss_1d = gauss_1d / gauss_1d.sum()

            result = result.unsqueeze(1)

            k_h = gauss_1d.view(1, 1, 1, kernel_size)
            result = F.pad(result, (feather_radius, feather_radius, 0, 0), mode='replicate')
            result = F.conv2d(result, k_h)

            k_v = gauss_1d.view(1, 1, kernel_size, 1)
            result = F.pad(result, (0, 0, feather_radius, feather_radius), mode='replicate')
            result = F.conv2d(result, k_v)

            result = result.squeeze(1)

        result = torch.clamp(result, 0.0, 1.0)

        # Last step, so the mask that comes out is genuinely hard-edged: a threshold applied before
        # feathering would just be blurred straight back into a grey ramp.
        if binary_mask:
            result = (result >= 0.5).to(result.dtype)

        return result

    @classmethod
    def fingerprint_inputs(
        cls,
        vae,
        image,
        downscale_factor=1.0,
        crop_2_nearest_16px=True,
        expand_mask=0,
        feather_mask=0,
        binary_mask=False,
        positive=None,
        negative=None,
    ):
        image_path = folder_paths.get_annotated_filepath(image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return ""

    @classmethod
    def validate_inputs(
        cls,
        vae,
        image,
        downscale_factor=1.0,
        crop_2_nearest_16px=True,
        expand_mask=0,
        feather_mask=0,
        binary_mask=False,
        positive=None,
        negative=None,
    ):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
