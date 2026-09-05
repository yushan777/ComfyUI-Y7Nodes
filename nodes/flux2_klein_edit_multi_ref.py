# All-in-one image-editing node for Flux.2 Klein workflows with multiple reference images.
#
# Pick the image to edit on the node itself, mask it (paint one on the node, or feed the optional
# `external_mask` socket from anywhere else), and get back a ready-to-sample
# reference latent plus patched positive/negative conditioning. On top of that, Flux.2 / Klein
# accepts a *list* of reference latents on the conditioning, so extra images can be wired in as
# additional visual context (a character sheet, a style reference, a product shot...) alongside
# the image actually being edited.
#
# Layout of the reference list handed to the model:
#   reference_latents[0]  -> the on-node `image` (the one being edited, and the only one a mask applies to)
#
# Masking applies to that primary image and nothing else, whichever route it arrives by - painted in the
# MaskEditor or wired into `external_mask`. The `ref_image_*` sockets are IMAGE-only, have no mask of
# their own, and drop any alpha channel before encoding. That is forced by what a mask is for here: it
# ends up as the latent's noise_mask, which has to line up cell-for-cell with the latent being denoised.
# The extra references are arbitrary images at arbitrary sizes, so a mask on one would have nothing to
# align to.
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
# Model coverage: this works with all flux.2-klein variants (base, distilled, 4B/8B text encoder).
# ComfyUI routes every Klein/Flux.2 checkpoint through the same Flux2 model class
# (comfy/model_base.py), which reads reference_latents/concat_latent_image/concat_mask identically
# regardless of backbone size or distillation.
import fnmatch
import logging
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
                tooltip="An extra picture for the model to look at. It is not edited, and it never appears "
                        "in your result directly.",
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
                        "an optional mask - painted on the node or fed into external_mask - that limits "
                        "which part of the picture is redrawn). The mask only ever applies to the image "
                        "being edited, never to the extra reference images.",
            inputs=[
                # VAE used to encode the uploaded (and optionally masked) image plus every reference image.
                io.Vae.Input("vae"),
                # Image to edit, picked from the ComfyUI input folder; shows an upload button + preview in the UI.
                # Any mask painted on it (via the built-in mask editor) confines the edit to that area, via the
                # latent's noise_mask; the mask options below shape it.
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
                # to the same scale. This one governs the edited image, which is the latent actually being
                # denoised, so it also sets the output resolution. 0 leaves it at its original resolution.
                io.Float.Input(
                    "target_megapixels", default=1.0, min=0.0, max=16.0, step=0.05,
                    tooltip="How big to make the picture you are editing. A megapixel is a million pixels, "
                            "so 1.0 resizes it to about a million pixels while keeping its shape - nothing "
                            "gets stretched or squashed. This decides how big your finished picture comes out. "
                            "0 = leave it at its original size. Klein is built for around 1.0; going much "
                            "higher usually makes things worse rather than better.",
                ),
                # Separate budget for the reference sockets, because they are not the same job as the edited
                # image: references only supply visual context, never output pixels, so they rarely need parity
                # with it. Splitting them also makes the total context size steerable - the budget is per image,
                # so eight sockets at 1 MP is 9 MP of latent tokens on the conditioning, not 1.
                # Deliberately the same range, default and rule as target_megapixels (0 = leave alone, above
                # 0 = that budget) rather than a sentinel for "follow the other widget": two widgets that read
                # identically are easier to reason about than one with an extra hidden mode, and the coupling
                # that sentinel bought back is the thing this widget exists to break.
                io.Float.Input(
                    "ref_megapixels", default=1.0, min=0.0, max=16.0, step=0.05,
                    tooltip="How big to make each reference picture. Works exactly like target_megapixels, "
                            "but these pictures are only there to be looked at, so this does not change how "
                            "big your finished picture comes out. 0 = leave them at their original size. "
                            "It applies to each picture separately, so they add up: one picture plus six "
                            "references, all at 1.0, is seven megapixels of work rather than one, and that is "
                            "what uses your video memory and your time. If you run slow or run out of memory, "
                            "turn this down first - try 0.5. A face or a style still comes through fine at "
                            "half the size, and your result stays just as big.",
                ),
                # Flux.2 works best on dimensions that are multiples of 16; the image-edit pipeline silently
                # rounds odd sizes down anyway. When on, the image (and its mask), plus every reference image,
                # is centre-cropped down to the nearest multiple of 16 before encoding, so what is encoded
                # matches what the model sees. No-op when the image is already aligned.
                io.Boolean.Input(
                    "crop_2_nearest_16px", default=True,
                    tooltip="Trims a few pixels off the edges so the width and height divide by 16, which "
                            "Flux.2 works best with. Applies to the picture you are editing, its mask, and "
                            "every reference picture. Does nothing if the sizes already divide by 16.",
                ),
                # Grows the mask outward by this many pixels (dilation) so the edit region covers a bit more
                # than what was painted. 0 disables expansion.
                # Defaults to 16 - one full latent cell. The mask is resized to latent resolution before use and
                # the Flux.2 VAE is 16x, so a tightly painted edge lands as a fractional latent cell and the edit
                # region is effectively clipped inward from what was painted.
                io.Int.Input(
                    "expand_mask", default=16, min=0, max=256, step=1,
                    tooltip="Grows the area you painted outwards by this many pixels, so the change covers "
                            "a little more than you painted. 16 is a good starting point: the mask is used "
                            "at a much smaller size than your picture, so a tight edge tends to lose a sliver "
                            "of what you painted. Try 24-32 for hair, fur or other soft edges, or when the "
                            "new thing needs more room than the old one. 0 turns it off. Like every "
                            "mask setting here, it only affects the picture being edited.",
                ),
                # Softens the mask edges with a Gaussian blur of this radius. Stays at 0 by default because the
                # mask never reaches the model: Klein has no mask input (in_channels == out_channels, so
                # Flux.concat_cond drops concat_mask), and the mask only drives the sampler's latent composite,
                # which runs at 1/16 resolution. A feather under ~16px is erased by that downsample, and a wider
                # one crossfades partially-denoised latents - which decodes as a smeared seam, not a blend.
                # Soft edges belong in pixel space, after decoding (see Y7 Paste Cropped Image Back).
                io.Int.Input(
                    "feather_mask", default=0, min=0, max=256, step=1,
                    tooltip="Softens the edge of the area you painted. Best left at 0: the mask is used at a "
                            "much smaller size than your picture, so a small softening vanishes entirely and "
                            "a large one smears the edge instead of blending it. To blend the edited part "
                            "into the rest smoothly, do it after the picture is made (the Y7 Paste Cropped "
                            "Image Back node). Above about 32 this acts as a loose, fuzzy edit area rather "
                            "than a tidy edge.",
                ),
                # Applied last, after expand/feather, so it always produces a hard-edged mask: everything at or
                # above 0.5 becomes fully opaque, everything below fully transparent. Feathering still shapes the
                # edge (rounding corners, smoothing jagged strokes) before the cut, but leaves no grey ramp.
                # On by default: the mask editor's brush carries hardness and opacity settings, so painted masks
                # are frequently soft-edged whether or not that was intended, and part-strength mask values only
                # buy a muddy latent-space blend.
                io.Boolean.Input(
                    "binary_mask", default=True,
                    tooltip="Makes the mask edge hard, so every part you painted is either fully changed or "
                            "left alone - nothing in between. On by default, because the mask brush can lay "
                            "down soft, half-strength edges without you noticing, and half-strength areas "
                            "come out muddy rather than blended. Turn it off only if you deliberately want "
                            "an edit applied at part strength.",
                ),
                # A mask made anywhere else - LoadImageMask on a mask file, a segmentation node, a
                # threshold - as an alternative to painting one on the node. Takes over from the painted
                # mask when it carries anything, and is resized to the loaded image before the expand/
                # feather/binary options above are applied, so both routes behave identically from there.
                io.Mask.Input(
                    "external_mask", optional=True,
                    tooltip="Use a mask from somewhere else instead of painting one on this node - a "
                            "black-and-white mask file loaded with Load Image (as Mask), or a mask from "
                            "any other node. White marks the part to change. If you use Load Image (as "
                            "Mask) with a plain black-and-white file, set its channel to red: on alpha "
                            "(the setting it starts on) that node hands back an empty mask and nothing "
                            "here gets masked. It is stretched to match "
                            "the picture being edited, so give it the same shape to avoid a blurred or "
                            "shifted edge. When this is connected and has anything in it, it replaces "
                            "whatever was painted on the node; leave it unconnected to paint instead. "
                            "Like a painted mask, it only ever applies to the picture being edited on "
                            "this node - the extra reference pictures are never masked.",
                ),
                # Growable IMAGE sockets (ref_image_2 ... ref_image_8). Each connected image is encoded and
                # appended, in socket order, after the on-node image's latent in `reference_latents`.
                io.Autogrow.Input(
                    "ref_images", template=ref_images_template,
                    tooltip="Extra pictures for the model to look at, on top of the one you are editing. "
                            "Connect one and another empty socket appears, up to seven in total. Refer to "
                            "them in your prompt by number - the one on the node is 1, ref_image_2 is 2, and "
                            "so on.",
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
                # The processed mask (after expand/feather/binary), for on-canvas preview. All-zero if no mask was painted.
                io.Mask.Output(display_name="preview_mask"),
                # Human-readable report of what this node actually did: sizes in and out, mask state,
                # every reference socket, and the token cost of the whole reference list. Wire it into
                # a text preview (Y7 Show Anything, or any string display) when something looks wrong.
                io.String.Output(display_name="debug"),
            ],
        )

    @classmethod
    def execute(
        cls,
        vae,
        image,
        target_megapixels=1.0,
        ref_megapixels=1.0,
        crop_2_nearest_16px=True,
        expand_mask=16,
        feather_mask=0,
        binary_mask=True,
        external_mask=None,
        ref_images=None,
        positive=None,
        negative=None,
    ) -> io.NodeOutput:
        # `image` arrives as the selected filename and is rebound to the loaded tensor, so keep the
        # name for the debug report first.
        image_name = image
        image, mask = nodes.LoadImage().load_image(image)

        # An external mask takes over from anything painted on the node. Applied here, immediately after
        # loading and before the resize/crop below, so from this point on it travels exactly the same
        # path as a painted mask - one code path, one set of expand/feather/binary semantics.
        mask_source = "painted on the node"
        external_mask_size = None
        if external_mask is not None:
            # A MASK is meant to be (B, H, W), but plenty of nodes hand back (H, W) or a channel-first
            # (B, 1, H, W); normalise both so everything below can assume three dimensions.
            if external_mask.dim() == 2:
                external_mask = external_mask.unsqueeze(0)
            elif external_mask.dim() == 4 and external_mask.shape[1] == 1:
                external_mask = external_mask.squeeze(1)
            external_mask_size = (external_mask.shape[-2], external_mask.shape[-1])
            if torch.count_nonzero(external_mask) > 0:
                # One mask per run: the node edits a single image, so a batch of masks has nothing to
                # pair up with beyond the first entry. Moved onto the loaded image's device/dtype so a
                # mask built on the CPU (most mask nodes) can multiply against a GPU-resident image.
                external_mask = external_mask[:1].to(device=image.device, dtype=image.dtype)
                if external_mask_size != (image.shape[1], image.shape[2]):
                    external_mask = F.interpolate(
                        external_mask.unsqueeze(1),
                        size=(image.shape[1], image.shape[2]),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).clamp(0.0, 1.0)
                mask = external_mask
                mask_source = "external_mask input"
            else:
                # Connected but blank. Falling back to the painted mask beats silently editing nothing;
                # the debug report says which one ended up being used either way.
                #
                # Almost always the "Load Image (as Mask)" trap: its `channel` widget defaults to alpha,
                # and a plain black-and-white mask file has no alpha channel, so that node returns a
                # 64x64 block of zeros. Loud in the console as well as in the debug output, because from
                # the canvas an ignored mask looks exactly like a mask that did nothing.
                mask_source = "external_mask input (empty - fell back to the painted mask)"
                logging.warning(
                    "[Y7 Flux.2 Klein Edit Multi-Ref] external_mask is connected but completely empty "
                    "(%dx%d, all black) - it was ignored. If it came from \"Load Image (as Mask)\", set "
                    "its `channel` to red instead of alpha: a black-and-white mask file has no alpha "
                    "channel, so that node hands back an empty 64x64 mask.",
                    external_mask_size[1], external_mask_size[0],
                )

        target_megapixels = cls._sanitize_megapixels(target_megapixels)
        ref_megapixels = cls._sanitize_megapixels(ref_megapixels)

        loaded_size = (image.shape[1], image.shape[2])

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

        resized_size = (image.shape[1], image.shape[2])

        if crop_2_nearest_16px:
            image, mask = cls._crop_to_multiple(image, mask, 16)

        final_size = (image.shape[1], image.shape[2])

        pixels = image.clone()

        latent = vae.encode(pixels[:, :, :, :3])
        result = {"samples": latent}

        # Extra references, encoded one latent per source image and kept in socket order.
        # `ref_report` is per-socket bookkeeping for the debug output only; it never affects sampling.
        ref_latents, ref_report = cls._encode_reference_images(
            vae, ref_images, ref_megapixels, crop_2_nearest_16px,
        )
        # The edited image always leads the list; Flux.2 reads the whole list as visual context.
        all_reference_latents = [latent] + ref_latents

        conditioned_image = pixels[:, :, :, :3]
        mask_resized = None
        # Size of whatever LoadImage handed back, kept before `mask` is dropped, so the debug output can
        # tell "no mask painted" (a 64x64 placeholder) apart from "mask painted but empty".
        raw_mask_size = None if mask is None else (mask.shape[-2], mask.shape[-1])

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

            # Masked regions are replaced with neutral gray so an edit model gets the preserved context
            # outside the editable area. Note that Klein itself ignores this: it has no mask channel
            # (in_channels == out_channels), so Flux.concat_cond drops concat_latent_image/concat_mask
            # entirely and only the noise_mask below has any effect. Kept for edit checkpoints that do
            # take fill-style conditioning.
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

        # Noted before the None -> [] coercion below, since an unconnected socket and a passed-through
        # empty list are indistinguishable afterwards, and "you forgot to wire the conditioning" is
        # exactly the kind of thing the debug output is for.
        positive_connected = positive is not None
        negative_connected = negative is not None

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

        debug = cls._build_debug_report(
            image_name=image_name,
            loaded_size=loaded_size,
            resized_size=resized_size,
            final_size=final_size,
            target_megapixels=target_megapixels,
            ref_megapixels=ref_megapixels,
            crop_2_nearest_16px=crop_2_nearest_16px,
            latent=latent,
            mask=mask,
            raw_mask_size=raw_mask_size,
            mask_resized=mask_resized,
            expand_mask=expand_mask,
            feather_mask=feather_mask,
            binary_mask=binary_mask,
            mask_source=mask_source,
            external_mask_size=external_mask_size,
            ref_report=ref_report,
            all_reference_latents=all_reference_latents,
            positive_connected=positive_connected,
            negative_connected=negative_connected,
        )

        return io.NodeOutput(result, positive, negative, image, preview_mask, debug)

    @staticmethod
    def _sanitize_megapixels(target_megapixels):
        """Coerce either megapixel widget to 0.0 (off) or a usable float up to 16.0, falling back to 1.0.

        Shared by target_megapixels and ref_megapixels: the two widgets have the same range and the same
        meaning, so they get the same coercion.
        """
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
    def _encode_reference_images(cls, vae, ref_images, ref_megapixels, crop_2_nearest_16px):
        """VAE-encode every connected reference socket into its own latent, in socket order.

        `ref_images` is the Autogrow dict ({"ref_image_2": tensor, ...}); only connected sockets are
        present, and dict order follows the declared socket order. A socket carrying a batch of images
        is split into one reference latent per image, since each entry of `reference_latents` is
        treated as a separate reference by the model rather than as a batch.

        Returns (latents, report), where `report` is one dict per connected socket describing what came
        in and what went out. It exists only to be printed on the `debug` output.
        """
        latents = []
        report = []
        if not ref_images:
            return latents, report

        for name, ref_image in ref_images.items():
            if ref_image is None:
                continue
            source_size = (ref_image.shape[1], ref_image.shape[2])
            ref_image = cls._prepare_reference_image(ref_image, ref_megapixels, crop_2_nearest_16px)
            encoded = vae.encode(ref_image[:, :, :, :3])
            for i in range(encoded.shape[0]):
                latents.append(encoded[i:i + 1])
            report.append({
                "name": name,
                "source_size": source_size,
                "final_size": (ref_image.shape[1], ref_image.shape[2]),
                "batch": int(encoded.shape[0]),
                "latent_size": (encoded.shape[2], encoded.shape[3]),
            })

        return latents, report

    @classmethod
    def _prepare_reference_image(cls, ref_image, ref_megapixels, crop_2_nearest_16px):
        """Put a reference image on the `ref_megapixels` budget and the same 16px alignment as the edited image.

        The budget is applied per image, not as a shared multiplier, so a reference smaller than it is
        scaled *up* - a 0.2 MP reference next to a 1 MP edit image contributes almost nothing at its
        native size. That upscale is not free, though: it is interpolated detail costing real latent
        tokens, and lanczos rings on the hard edges of logos and line art, which is a reason to give
        references a smaller budget than the edited image rather than blanket parity.
        """
        align = 16 if crop_2_nearest_16px else 8

        if ref_megapixels > 0.0:
            scaled_height, scaled_width = cls._target_size(ref_image, ref_megapixels, align)
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

    @staticmethod
    def _fmt_size(size):
        """(H, W) tuple -> "W x H", the order people read image sizes in."""
        if size is None:
            return "-"
        return "{} x {}".format(size[1], size[0])

    @staticmethod
    def _megapixels(size):
        """Megapixels of an (H, W) tuple, counted as 1024*1024 to match `_target_size`."""
        return (size[0] * size[1]) / (1024 * 1024)

    @classmethod
    def _build_debug_report(
        cls,
        image_name,
        loaded_size,
        resized_size,
        final_size,
        target_megapixels,
        ref_megapixels,
        crop_2_nearest_16px,
        latent,
        mask,
        raw_mask_size,
        mask_resized,
        expand_mask,
        feather_mask,
        binary_mask,
        mask_source,
        external_mask_size,
        ref_report,
        all_reference_latents,
        positive_connected,
        negative_connected,
    ):
        """Plain-text summary of this run, for the `debug` output.

        Purely descriptive - it reads state that has already been computed and never changes any of it.
        Token counts are latent cells: Flux.2 uses patch_size 1 (comfy/model_detection.py), so every
        latent cell is one token in the transformer sequence, and the VAE is 16x, which makes a
        1024x1024 image 64x64 = 4096 tokens. The whole reference list rides on the conditioning at
        once, so the total is what actually has to fit in memory.
        """
        lines = []
        warnings = []

        def row(label, value):
            lines.append("  {:<14}{}".format(label, value))

        latent_size = (latent.shape[2], latent.shape[3])
        edit_tokens = latent_size[0] * latent_size[1]

        lines.append("[ image being edited ]")
        row("file", image_name)
        row("loaded", "{}  ({:.2f} MP)".format(cls._fmt_size(loaded_size), cls._megapixels(loaded_size)))
        if target_megapixels > 0.0:
            row("resized", "{}  ({:.2f} MP, asked for {:.2f})".format(
                cls._fmt_size(resized_size), cls._megapixels(resized_size), target_megapixels))
        else:
            row("resized", "off (target_megapixels 0) - kept at its original size")
        if not crop_2_nearest_16px:
            row("crop to /16", "off")
        elif final_size == resized_size:
            row("crop to /16", "nothing to trim")
        else:
            row("crop to /16", "{}  (trimmed {} x {} px)".format(
                cls._fmt_size(final_size),
                resized_size[1] - final_size[1],
                resized_size[0] - final_size[0]))
        row("encoded", "{}  ({:.2f} MP)".format(cls._fmt_size(final_size), cls._megapixels(final_size)))
        row("latent", "{}  x {} channels".format(cls._fmt_size(latent_size), latent.shape[1]))
        row("tokens", "{}".format(edit_tokens))
        row("output size", "{} - this is how big your result comes out".format(cls._fmt_size(final_size)))

        lines.append("")
        lines.append("[ mask ]")
        if mask is None:
            row("in use", "no - the whole picture is up for editing")
            if external_mask_size is not None:
                row("external", "{} (empty - ignored)".format(cls._fmt_size(external_mask_size)))
                warnings.append(
                    "external_mask is connected but completely black, so it was ignored. If it comes "
                    "from \"Load Image (as Mask)\", set that node's channel to red - on alpha it returns "
                    "an empty 64 x 64 mask for any file without an alpha channel")
            if raw_mask_size is not None:
                row("loaded mask", "{} (empty)".format(cls._fmt_size(raw_mask_size)))
        else:
            mask_size = (mask.shape[-2], mask.shape[-1])
            row("in use", "yes")
            row("source", mask_source)
            if external_mask_size is not None:
                if mask_source != "external_mask input":
                    note = "  (empty - ignored, the painted mask was used instead)"
                    warnings.append(
                        "external_mask is connected but completely black, so the mask painted on the "
                        "node was used instead. If it comes from \"Load Image (as Mask)\", set that "
                        "node's channel to red - on alpha it returns an empty 64 x 64 mask for any "
                        "file without an alpha channel")
                elif external_mask_size != final_size:
                    note = "  (stretched to fit the picture)"
                else:
                    note = ""
                row("external", "{}{}".format(cls._fmt_size(external_mask_size), note))
            row("size", cls._fmt_size(mask_size))
            row("coverage", "{:.1f}% of the picture".format(float(mask.mean()) * 100.0))
            row("expand", "{} px".format(expand_mask) if expand_mask > 0 else "off")
            row("feather", "{} px".format(feather_mask) if feather_mask > 0 else "off")
            row("hard edge", "on" if binary_mask else "off")
            if mask_resized is not None:
                row("latent mask", "{} (concat_mask + noise_mask set)".format(
                    cls._fmt_size((mask_resized.shape[-2], mask_resized.shape[-1]))))
            if external_mask_size is not None and mask_source == "external_mask input":
                ext_aspect = external_mask_size[1] / max(external_mask_size[0], 1)
                img_aspect = final_size[1] / max(final_size[0], 1)
                if abs(ext_aspect - img_aspect) > 0.01:
                    warnings.append(
                        "external_mask is {} but the picture is {} - a different shape, so the mask was "
                        "squashed to fit and no longer lines up with what you masked".format(
                            cls._fmt_size(external_mask_size), cls._fmt_size(final_size)))
            if mask_size != final_size:
                warnings.append(
                    "mask is {} but the picture is {} - it was stretched to fit, which can blur the edge".format(
                        cls._fmt_size(mask_size), cls._fmt_size(final_size)))

        lines.append("")
        lines.append("[ reference pictures ]")
        row("budget", "{:.2f} MP each".format(ref_megapixels) if ref_megapixels > 0.0
            else "off (ref_megapixels 0) - kept at their original sizes")
        lines.append("  1  (the picture being edited)  {}  ->  {} tokens".format(
            cls._fmt_size(final_size), edit_tokens))

        slot = 2
        if not ref_report:
            lines.append("  (no extra reference sockets connected)")
        for entry in ref_report:
            tokens_each = entry["latent_size"][0] * entry["latent_size"][1]
            resize_note = "" if entry["final_size"] == entry["source_size"] else "  (was {})".format(
                cls._fmt_size(entry["source_size"]))
            for i in range(entry["batch"]):
                batch_note = "" if entry["batch"] == 1 else "  [batch image {} of {}]".format(i + 1, entry["batch"])
                lines.append("  {}  {:<16}{}{}  ->  {} tokens{}".format(
                    slot, entry["name"], cls._fmt_size(entry["final_size"]), resize_note,
                    tokens_each, batch_note))
                slot += 1
            if entry["batch"] > 1:
                warnings.append(
                    "{} is carrying a batch of {} pictures, so it counts as {} references, not 1".format(
                        entry["name"], entry["batch"], entry["batch"]))
            src_pixels = entry["source_size"][0] * entry["source_size"][1]
            final_pixels = entry["final_size"][0] * entry["final_size"][1]
            if final_pixels > src_pixels * 1.5:
                warnings.append(
                    "{} was enlarged from {} to {} - it costs full price in memory but adds no real detail".format(
                        entry["name"], cls._fmt_size(entry["source_size"]), cls._fmt_size(entry["final_size"])))

        total_tokens = sum(lat.shape[2] * lat.shape[3] for lat in all_reference_latents)

        lines.append("")
        lines.append("[ totals ]")
        row("references", "{} (the one being edited + {} extra)".format(
            len(all_reference_latents), len(all_reference_latents) - 1))
        row("tokens", "{} across every reference - this is what costs memory and time".format(total_tokens))
        row("positive", "connected" if positive_connected else "NOT connected")
        row("negative", "connected" if negative_connected else "NOT connected")

        if not positive_connected:
            warnings.append(
                "nothing is plugged into `positive`, so the positive output is empty and the model gets "
                "no prompt and no reference pictures")
        if not negative_connected:
            warnings.append(
                "nothing is plugged into `negative`, so the negative output is empty")
        if total_tokens > 20000:
            warnings.append(
                "{} tokens is a lot of context - if you run out of memory or it crawls, lower "
                "ref_megapixels first (0.5 is usually plenty for a face or a style)".format(total_tokens))

        if warnings:
            lines.append("")
            lines.append("[ worth knowing ]")
            for warning in warnings:
                lines.append("  ! {}".format(warning))

        return "\n".join(lines)

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
