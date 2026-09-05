import torch
import torch.nn.functional as F

MAX_RESOLUTION = 16384

FILL_MODES = ["grey", "edge replicate", "mirror", "blurred edge", "noise"]

def snap_to_step(value, step):
    """Round to the nearest multiple of step, ties rounding up."""
    if step <= 1:
        return value
    quotient, remainder = divmod(value, step)
    if remainder * 2 >= step:
        quotient += 1
    return quotient * step


def resolve_padding(left, top, right, bottom, step):
    """Turn the raw widget values into the final padding by snapping each to `step`."""
    return (
        snap_to_step(left, step),
        snap_to_step(top, step),
        snap_to_step(right, step),
        snap_to_step(bottom, step),
    )


def _replicate_index(length, before, after):
    """Indices into a row/column of `length` px that clamp to the edge outside it."""
    return torch.arange(-before, length + after).clamp(0, length - 1)


def _mirror_index(length, before, after):
    """Indices into a row/column of `length` px that reflect back inside it."""
    idx = torch.arange(-before, length + after)
    if length == 1:
        return torch.zeros_like(idx)
    period = 2 * length - 2
    idx = idx.abs() % period
    return torch.where(idx >= length, period - idx, idx)


def _fill_canvas(image, left, top, right, bottom, fill):
    """
    Build the padded canvas and fill the new region according to `fill`.
    The original image is not pasted in yet - the caller does that.
    """
    d1, d2, d3, d4 = image.size()
    new_h, new_w = d2 + top + bottom, d3 + left + right

    if fill == "grey":
        return torch.ones((d1, new_h, new_w, d4), dtype=torch.float32) * 0.5

    if fill == "noise":
        mean = image.mean(dim=(1, 2), keepdim=True)
        std = image.std(dim=(1, 2), keepdim=True)
        noise = torch.randn((d1, new_h, new_w, d4), dtype=torch.float32)
        return (noise * std * 0.5 + mean).clamp(0.0, 1.0)

    index = _mirror_index if fill == "mirror" else _replicate_index
    rows = index(d2, top, bottom)
    cols = index(d3, left, right)
    canvas = image[:, rows][:, :, cols].float()

    if fill == "blurred edge":
        # Heavy downscale then bilinear back up - a cheap, very soft blur that
        # smears the edge colours outwards instead of leaving hard streaks.
        scale = max(2, max(new_h, new_w) // 48)
        small = F.interpolate(
            canvas.permute(0, 3, 1, 2),
            size=(max(1, new_h // scale), max(1, new_w // scale)),
            mode="area",
        )
        canvas = F.interpolate(
            small, size=(new_h, new_w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)

    return canvas.contiguous()


def _feather_mask(d2, d3, left, top, right, bottom, feathering):
    """
    The soft ramp painted over the original image region, matching the built-in node's
    falloff but computed with tensor ops instead of a per-pixel Python loop.
    """
    t = torch.zeros((d2, d3), dtype=torch.float32)

    # The built-in silently skips feathering once it no longer fits; clamp instead so
    # a large value still gives the widest ramp the image can hold.
    f = min(feathering, (min(d2, d3) - 1) // 2)
    if f <= 0:
        return t

    ii = torch.arange(d2, dtype=torch.float32).view(-1, 1).expand(d2, d3)
    jj = torch.arange(d3, dtype=torch.float32).view(1, -1).expand(d2, d3)

    # A side with no padding contributes no falloff, so it is given a distance
    # large enough to never win the min().
    big_v = torch.full((d2, d3), float(d2))
    big_h = torch.full((d2, d3), float(d3))

    dt = ii if top != 0 else big_v
    db = (d2 - ii) if bottom != 0 else big_v
    dl = jj if left != 0 else big_h
    dr = (d3 - jj) if right != 0 else big_h

    d = torch.minimum(torch.minimum(dt, db), torch.minimum(dl, dr))
    v = ((f - d) / f).clamp(min=0.0)
    return v * v


class Y7Nodes_ImagePadForOutpaint:
    """
    Pad an image ready for outpainting.

    A rework of ComfyUI's built-in ImagePadForOutpaint with the fiddly parts smoothed
    out: padding snaps to a chosen step, the new area can be filled with something more
    useful than flat grey, and the feathering is vectorised rather than looping over
    every pixel in Python.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "step": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Padding is rounded to a multiple of this number, so the "
                               "final image stays a size the model is happy with. Also "
                               "sets how far the +/- arrows move each side.",
                }),
                "fill": (FILL_MODES, {
                    "default": "grey",
                    "tooltip": "What to put in the new area before outpainting. "
                               "grey: flat 50% grey, same as the built-in node. "
                               "edge replicate: smears the outermost pixels outwards. "
                               "mirror: reflects the image back on itself. "
                               "blurred edge: a soft blur of the nearby colours. "
                               "noise: random noise in the image's own colours. "
                               "It all gets painted over anyway, but a plausible starting "
                               "colour usually gives a better outpaint than flat grey.",
                }),
                "feathering": ("INT", {
                    "default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                    "advanced": True,
                    "tooltip": "Width in pixels of the soft fade at the join between the "
                               "original image and the new area. 0 gives a hard edge.",
                }),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                 "tooltip": "Pixels to add to the left edge."}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                "tooltip": "Pixels to add to the top edge."}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                  "tooltip": "Pixels to add to the right edge."}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1,
                                   "tooltip": "Pixels to add to the bottom edge."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image (original)", "image (padded)", "mask", "width", "height")
    FUNCTION = "expand_image"
    OUTPUT_NODE = True

    CATEGORY = "Y7Nodes/Image"

    def expand_image(self, image, step, fill, feathering, left, top, right, bottom):
        d1, d2, d3, d4 = image.size()

        left, top, right, bottom = resolve_padding(left, top, right, bottom, step)

        new_image = _fill_canvas(image, left, top, right, bottom, fill)
        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones((d2 + top + bottom, d3 + left + right), dtype=torch.float32)
        mask[top:top + d2, left:left + d3] = _feather_mask(
            d2, d3, left, top, right, bottom, feathering
        )

        new_w, new_h = d3 + left + right, d2 + top + bottom

        return {
            "ui": {
                "text": [
                    f"{d3} x {d2}  ->  {new_w} x {new_h}",
                    f"L {left}   T {top}   R {right}   B {bottom}",
                ],
                "src": [d3, d2],
            },
            "result": (image, new_image, mask.unsqueeze(0), new_w, new_h),
        }
