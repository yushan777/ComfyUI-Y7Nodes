import math
import comfy.utils
from comfy_api.latest import io


class Y7Nodes_ScaleImageToTotalPixels(io.ComfyNode):

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_ScaleImageToTotalPixels",
            display_name="Y7 Scale Image to Total Pixels",
            category="Y7Nodes/Image",
            description="Scale an image to a target megapixel count, snapping the result to a step size.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS, default="lanczos"),
                io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                io.Int.Input("resolution_steps", optional=True, default=8, min=1, max=256),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, megapixels, resolution_steps=8) -> io.NodeOutput:
        samples = image.movedim(-1, 1)
        total = megapixels * 1024 * 1024

        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps

        s = comfy.utils.common_upscale(samples, int(width), int(height), upscale_method, "disabled")
        s = s.movedim(1, -1)
        return io.NodeOutput(s, int(width), int(height), ui={"text": [f"{int(width)} x {int(height)}"]})
