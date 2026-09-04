# All-in-one image-editing node for Flux.2 Klein workflows with multiple reference images.
#
# Same idea as the single-reference node (flux2_klein_edit_ref1.py): pick the image to edit on the
# node itself, paint a mask on it, and get back a ready-to-sample reference latent plus patched
# positive/negative conditioning. The difference is the extra reference images: Flux.2 / Klein accepts
# a *list* of reference latents on the conditioning, so extra images can be fed in as additional
# visual context (a character sheet, a style reference, a product shot...) alongside the image
# actually being edited.
#
# Layout of the reference list handed to the model:
#   reference_latents[0]  -> the on-node `image` (the one being edited, and the only one a mask applies to)
#   reference_latents[1:] -> `ref_image_2`, `ref_image_3`, ... in socket order
#
# Prompting: there is no special syntax for addressing the references. Klein taps the Qwen3-VL LM with
# the visual tower unused (comfy/sd.py: "Flux2 Klein reuses the Qwen3-VL LM ...; visual unused"), so the
# text encoder never sees the images at all - they only reach the transformer as latent tokens appended
# to the sequence, in list order. Refer to them in plain English by position, always paired with a noun:
#   "Have the man in Figure 1 put on the clothes from Figure 2, and change the background to a savannah"
# `Figure N` is the wording in ComfyUI's own multi-reference Klein template; `image 1` / `the first image`
# is the same kind of plain positional reference. Neither is a real token, so both are worth trying if a
# prompt is not binding to the reference you meant.
#
# The extra references are IMAGE sockets rather than on-node file pickers, for two reasons:
#   - the mask editor is hard-wired to the widget literally named "image" (it looks the widget up by
#     name in ComfyApp.pasteFromClipspace), so only one on-node picker can ever carry a painted mask;
#   - as sockets they can come from anywhere - Load Image, an upscaler, another Y7 node, a batch.
#
# The sockets are declared with io.Autogrow (see docs/v3_migration.md), so only one empty
# `ref_image_2` slot is shown to begin with and a new slot appears each time one is connected,
# up to ref_image_8.
#
# Model coverage: as with the single-reference node, this works with all flux.2-klein variants
# (base, distilled, 4B/8B text encoder). ComfyUI routes every Klein/Flux.2 checkpoint through the
# same Flux2 model class (comfy/model_base.py), which reads reference_latents/concat_latent_image/
# concat_mask identically regardless of backbone size or distillation.
import fnmatch
import math
import os

import torch
import torch.nn.functional as F

import comfy.utils
import folder_paths
import node_helpers
import nodes
from comfy_api.latest import io

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Names for the growable reference sockets. The edited on-node image is conceptually "ref image 1",
# so the extra sockets start at 2. First entry is always shown (min=1); the rest appear as the
# preceding one is connected.
REF_IMAGE_NAMES = [f"ref_image_{i}" for i in range(2, 9)]


class Y7Nodes_Flux2KleinEdit_MultiRef(io.ComfyNode):
    """Loads the edit image on-node, takes extra reference images as sockets, and prepares Klein edit conditioning."""

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

        # Each generated socket is an optional IMAGE input, so the node still runs with nothing but
        # the on-node image connected. min=1 keeps a single empty `ref_image_2` slot visible.
        ref_images_template = io.Autogrow.TemplateNames(
            input=io.Image.Input(
                "ref_image",
                optional=True,
                tooltip="Extra reference image, VAE-encoded and appended to the conditioning's reference latents.",
            ),
            names=REF_IMAGE_NAMES,
            min=1,
        )

        return io.Schema(
            node_id="Y7Nodes_Flux2KleinEdit_MultiRef",
            display_name="Y7 Flux.2 Klein Edit Multi-Ref",
            category="Y7Nodes/Klein",
            description="Loads the edit image on the node, accepts additional reference images on growable "
                        "sockets, and prepares Klein edit conditioning (reference latents, plus optional "
                        "mask-driven inpaint conditioning).",
            inputs=[
                # VAE used to encode the uploaded (and optionally masked) image plus every reference image.
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
                # Resamples the image (and its mask) to this pixel budget before encoding - the same maths as
                # ImageScaleToTotalPixels, and the reason it is a target rather than a relative factor: Klein is
                # trained around 1 MP, and a straight multiplier cannot bring an 8 MP photo and a 0.2 MP reference
                # to the same scale. Applied independently to every image, so each one lands on the budget
                # whether it has to shrink or grow. 0 leaves every image at its original resolution.
                io.Float.Input(
                    "target_megapixels", default=1.0, min=0.0, max=16.0, step=0.05,
                    tooltip="Resample the source image and every reference image to roughly this many megapixels "
                            "before encoding (0 = leave at original resolution). Flux.2 Klein is trained around "
                            "1.0 MP; sampling far above that degrades quality badly at low step counts.",
                ),
                # Flux.2 works best on dimensions that are multiples of 16; the image-edit pipeline silently
                # rounds odd sizes down anyway. When on, the image (and its mask), plus every reference image,
                # is centre-cropped down to the nearest multiple of 16 before encoding, so what is encoded
                # matches what the model sees. No-op when the image is already aligned.
                io.Boolean.Input(
                    "crop_2_nearest_16px", default=True,
                    tooltip="Centre-crop the image (and mask) and every reference image down to the nearest "
                            "multiple of 16, which Flux.2 prefers. No-op if the dimensions are already multiples of 16.",
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
                # Growable IMAGE sockets (ref_image_2 ... ref_image_8). Each connected image is encoded and
                # appended, in socket order, after the on-node image's latent in `reference_latents`.
                io.Autogrow.Input(
                    "ref_images", template=ref_images_template,
                    tooltip="Additional reference images. Each connected socket is VAE-encoded and appended to "
                            "the conditioning's reference latents, after the on-node image.",
                ),
                # Optional positive conditioning to extend with the reference latents / mask-based concat data.
                # Passed through untouched (as an empty list) if not connected.
                io.Conditioning.Input("positive", optional=True),
                # Optional negative conditioning, extended the same way as positive.
                io.Conditioning.Input("negative", optional=True),
            ],
            outputs=[
                # VAE-encoded latent of the (resampled) source image, for use as an edit-model reference latent.
                # Only the edited image - the extra references go onto the conditioning, not into this latent.
                io.Latent.Output(display_name="reference_latent"),
                # Positive conditioning with reference_latents/concat_latent_image (and concat_mask if a mask was used) set.
                io.Conditioning.Output(display_name="positive"),
                # Negative conditioning, extended the same way as positive.
                io.Conditioning.Output(display_name="negative"),
                # The (possibly resampled) source image, for on-canvas preview.
                io.Image.Output(display_name="preview_image"),
                # The processed mask (after binary/expand/feather), for on-canvas preview. All-zero if no mask was painted.
                io.Mask.Output(display_name="preview_mask"),
                # How many reference latents ended up on the conditioning (the edited image plus every extra
                # reference, batched references counted per-frame). Handy sanity check when wiring several sockets.
                io.Int.Output(display_name="ref_count"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae,
        image,
        target_megapixels=1.0,
        crop_2_nearest_16px=True,
        expand_mask=0,
        feather_mask=0,
        binary_mask=False,
        ref_images=None,
        positive=None,
        negative=None,
    ) -> io.NodeOutput:
        # `image` arrives as the selected filename and is rebound to the loaded tensor.
        image, mask = nodes.LoadImage().load_image(image)
        target_megapixels = cls._sanitize_megapixels(target_megapixels)

        # When cropping to multiples of 16 is requested, snap the resize target to 16 as well, so the
        # crop below has nothing left to trim instead of shaving another 8 pixels off a just-resized image.
        align = 16 if crop_2_nearest_16px else 8
        if target_megapixels > 0.0:
            scaled_height, scaled_width = cls._target_size(image, target_megapixels, align)
            image = cls._resample(image, scaled_width, scaled_height)
            if mask is not None and torch.count_nonzero(mask) > 0:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                # The mask stays on bilinear rather than the image's lanczos: lanczos round-trips through
                # 8-bit PIL, and can overshoot past 0/1 on a hard edge.
                mask = F.interpolate(
                    mask.unsqueeze(1),
                    size=(scaled_height, scaled_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1).clamp(0.0, 1.0)

        if crop_2_nearest_16px:
            image, mask = cls._crop_to_multiple(image, mask, 16)

        pixels = image.clone()

        latent = vae.encode(pixels[:, :, :, :3])
        result = {"samples": latent}

        # Extra references, encoded one latent per source image and kept in socket order.
        ref_latents = cls._encode_reference_images(vae, ref_images, target_megapixels, crop_2_nearest_16px)
        # The edited image always leads the list; Flux.2 reads the whole list as visual context.
        all_reference_latents = [latent] + ref_latents

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

        # Only the edited image feeds concat conditioning: concat_latent_image has to line up pixel-for-pixel
        # with the latent being denoised, which the extra references (any size, any subject) do not.
        concat_latent_image = vae.encode(conditioned_image)

        if positive is not None:
            positive = node_helpers.conditioning_set_values(
                positive,
                {"concat_latent_image": concat_latent_image},
            )
            positive = node_helpers.conditioning_set_values(
                positive,
                {"reference_latents": all_reference_latents},
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
                {"reference_latents": all_reference_latents},
                append=True,
            )
            if mask is not None:
                negative = node_helpers.conditioning_set_values(negative, {"concat_mask": mask_resized})

        if positive is None:
            positive = []
        if negative is None:
            negative = []

        preview_mask = mask if mask is not None else torch.zeros((1, pixels.shape[1], pixels.shape[2]), dtype=torch.float32)

        return io.NodeOutput(result, positive, negative, image, preview_mask, len(all_reference_latents))

    @staticmethod
    def _sanitize_megapixels(target_megapixels):
        """Coerce the megapixel widget to 0.0 (off) or a usable float up to 16.0, falling back to 1.0."""
        if target_megapixels is None:
            return 1.0
        try:
            target_megapixels = float(target_megapixels)
        except (TypeError, ValueError):
            return 1.0
        if not math.isfinite(target_megapixels) or target_megapixels <= 0.0:
            return 0.0
        return min(16.0, target_megapixels)

    @staticmethod
    def _target_size(image, target_megapixels, align):
        """Dimensions that put a (B,H,W,C) image on the megapixel budget, rounded to a multiple of `align`.

        Megapixels are counted as 1024*1024, matching ImageScaleToTotalPixels, so a value copied from that
        node lands on the same size. Rounding is to the *nearest* multiple rather than down, so the result
        keeps the aspect ratio and needs no further cropping.
        """
        height, width = image.shape[1], image.shape[2]
        scale_by = math.sqrt((target_megapixels * 1024 * 1024) / (width * height))
        new_width = max(align, round(width * scale_by / align) * align)
        new_height = max(align, round(height * scale_by / align) * align)
        return new_height, new_width

    @staticmethod
    def _resample(image, width, height):
        """Resize a (B,H,W,C) image, skipping the work when it is already the right size."""
        if (image.shape[1], image.shape[2]) == (height, width):
            return image
        # lanczos both ways: these images are usually being shrunk by a large factor, where bilinear
        # aliases badly, and the reference images are sometimes being grown instead.
        return comfy.utils.common_upscale(
            image.movedim(-1, 1), width, height, "lanczos", "disabled",
        ).movedim(1, -1)

    @classmethod
    def _encode_reference_images(cls, vae, ref_images, target_megapixels, crop_2_nearest_16px):
        """VAE-encode every connected reference socket into its own latent, in socket order.

        `ref_images` is the Autogrow dict ({"ref_image_2": tensor, ...}); only connected sockets are
        present, and dict order follows the declared socket order. A socket carrying a batch of images
        is split into one reference latent per image, since each entry of `reference_latents` is
        treated as a separate reference by the model rather than as a batch.
        """
        latents = []
        if not ref_images:
            return latents

        for ref_image in ref_images.values():
            if ref_image is None:
                continue
            ref_image = cls._prepare_reference_image(ref_image, target_megapixels, crop_2_nearest_16px)
            encoded = vae.encode(ref_image[:, :, :, :3])
            for i in range(encoded.shape[0]):
                latents.append(encoded[i:i + 1])

        return latents

    @classmethod
    def _prepare_reference_image(cls, ref_image, target_megapixels, crop_2_nearest_16px):
        """Put a reference image on the same megapixel budget / 16px alignment as the edited image.

        The budget is applied per image, not as a shared multiplier, so a reference smaller than the
        target is scaled *up* to it - a 0.2 MP reference next to a 1 MP edit image contributes almost
        nothing at its native size.
        """
        align = 16 if crop_2_nearest_16px else 8

        if target_megapixels > 0.0:
            scaled_height, scaled_width = cls._target_size(ref_image, target_megapixels, align)
            ref_image = cls._resample(ref_image, scaled_width, scaled_height)

        if crop_2_nearest_16px:
            ref_image, _ = cls._crop_to_multiple(ref_image, None, 16)

        return ref_image

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

    # The dynamic ref sockets mean the kwargs this node is validated/fingerprinted with vary with the
    # workflow, so both hooks take **kwargs and look only at the on-node image.
    @classmethod
    def fingerprint_inputs(cls, image=None, **kwargs):
        if image is None:
            return ""
        image_path = folder_paths.get_annotated_filepath(image)
        if os.path.exists(image_path):
            return os.path.getmtime(image_path)
        return ""

    @classmethod
    def validate_inputs(cls, image=None, **kwargs):
        if image is None or not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
