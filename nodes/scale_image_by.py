import comfy.utils


class Y7Nodes_ScaleImageBy:

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_method": (cls.UPSCALE_METHODS, {"default": "lanczos"}),
                "scale_by": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.01,
                    "max": 8.0,
                    "step": 0.01,
                }),
            },
            "optional": {
                "resolution_steps": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "execute"
    CATEGORY = "Y7Nodes/image"
    OUTPUT_NODE = True

    def execute(self, image, upscale_method, scale_by, resolution_steps=8):
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return {"ui": {"text": [f"{width} x {height}"]}, "result": (s, width, height)}
