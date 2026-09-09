import comfy.utils
from comfy_api.latest import io


class Y7Nodes_ScaleImageBy(io.ComfyNode):

    UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_ScaleImageBy",
            display_name="Y7 Scale Image By",
            category="Y7Nodes/Image",
            description="Scale an image by a factor, snapping the result to a step size.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("upscale_method", options=cls.UPSCALE_METHODS, default="lanczos"),
                io.Float.Input("scale_by", default=1.0, min=0.01, max=8.0, step=0.01),
                io.Int.Input("resolution_steps", optional=True, default=8, min=1, max=256),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
            ],
        )

    @classmethod
    def execute(cls, image, upscale_method, scale_by, resolution_steps=8) -> io.NodeOutput:
        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by / resolution_steps) * resolution_steps
        height = round(samples.shape[2] * scale_by / resolution_steps) * resolution_steps
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return io.NodeOutput(s, width, height, ui={"text": [f"{width} x {height}"]})
