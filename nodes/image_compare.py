# ==========================================================================
# Y7 Image Compare
# ==========================================================================
# 
# Compares two images. using a draggable slider for interactive side-by-side 
# comparison with 2 blend modes (normal and difference).
#
#   - The node displays a live preview of the connected images, updating as
#     the slider is moved or the blend mode is changed.
#   - Previews are delivered through ComfyUI's standard node-output channel
#     (saved to the temp folder and returned under "ui"), so they persist
#     across workflow-tab switches instead of being lost like a one-shot event.
#   - Automatic resizing of the node to match the aspect ratio of the input images.
#   - State serialization: Slider position and blend mode are saved with the workflow.
#
# ==========================================================================


from comfy_api.latest import io
from nodes import PreviewImage  # type: ignore

# Main class --------------

class Y7Nodes_ImageCompare(io.ComfyNode):
    """
    A custom node to compare two images with a
    draggable slider and selectable blend modes.

    Previews are returned through ComfyUI's standard output channel
    (image files under "ui"), which ComfyUI persists per workflow tab,
    so switching tabs and returning keeps the preview visible.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_ImageCompare",
            display_name="Y7 Image Compare",
            category="Y7Nodes/Image",
            description="Compare two images with a draggable slider and selectable blend modes.",
            is_output_node=True,
            inputs=[
                io.Image.Input("image_a"),
                io.Image.Input("image_b", optional=True),
            ],
            outputs=[],
            hidden=[
                io.Hidden.prompt,
                io.Hidden.extra_pnginfo,
                io.Hidden.unique_id,
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b=None) -> io.NodeOutput:
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        # Save previews to the temp folder and hand them back to the frontend
        # through the standard "ui" channel. ComfyUI stores these in
        # app.nodeOutputs[node_id] and restores them when the workflow tab is
        # re-activated, so the preview survives tab switches.
        a_images = []
        b_images = []

        # PreviewImage is only used here as a helper for writing previews to the temp
        # folder; it keeps the original filename scheme and compression settings.
        previewer = PreviewImage()

        if image_a is not None:
            a_images = previewer.save_images(image_a, "y7_compare", prompt, extra_pnginfo)["ui"]["images"]

        if image_b is not None:
            b_images = previewer.save_images(image_b, "y7_compare", prompt, extra_pnginfo)["ui"]["images"]

        return io.NodeOutput(ui={"a_images": a_images, "b_images": b_images})
