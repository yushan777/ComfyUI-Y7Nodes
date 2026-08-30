import torch

MAX_RESOLUTION = 16384


def snap_to_step(value, step):
    if step <= 1:
        return value
    quotient, remainder = divmod(value, step)
    if remainder * 2 >= step:
        quotient += 1
    return quotient * step


class Y7Nodes_ImagePadForOutpaint:
    """
    Clone of ComfyUI's built-in ImagePadForOutpaint, with an added `step` input:
    each of left/top/right/bottom is independently snapped to the nearest multiple
    of `step` (ties round up) before padding is applied.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "top": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "right": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "feathering": ("INT", {"default": 40, "min": 0, "max": MAX_RESOLUTION, "step": 1, "advanced": True}),
                "step": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "Each of left/top/right/bottom is snapped to the nearest multiple of this value (ties round up) before padding is applied.",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "expand_image"

    CATEGORY = "Y7Nodes/Image"

    def expand_image(self, image, left, top, right, bottom, feathering, step):
        d1, d2, d3, d4 = image.size()

        left = snap_to_step(left, step)
        top = snap_to_step(top, step)
        right = snap_to_step(right, step)
        bottom = snap_to_step(bottom, step)

        new_image = torch.ones(
            (d1, d2 + top + bottom, d3 + left + right, d4),
            dtype=torch.float32,
        ) * 0.5

        new_image[:, top:top + d2, left:left + d3, :] = image

        mask = torch.ones(
            (d2 + top + bottom, d3 + left + right),
            dtype=torch.float32,
        )

        t = torch.zeros(
            (d2, d3),
            dtype=torch.float32
        )

        if feathering > 0 and feathering * 2 < d2 and feathering * 2 < d3:

            for i in range(d2):
                for j in range(d3):
                    dt = i if top != 0 else d2
                    db = d2 - i if bottom != 0 else d2

                    dl = j if left != 0 else d3
                    dr = d3 - j if right != 0 else d3

                    d = min(dt, db, dl, dr)

                    if d >= feathering:
                        continue

                    v = (feathering - d) / feathering

                    t[i, j] = v * v

        mask[top:top + d2, left:left + d3] = t

        return (new_image, mask.unsqueeze(0))
