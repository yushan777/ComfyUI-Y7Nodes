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


from nodes import PreviewImage  # type: ignore

# Main class --------------

class Y7Nodes_ImageCompare(PreviewImage):
    """
    A custom node to compare two images with a
    draggable slider and selectable blend modes.

    Previews are returned through ComfyUI's standard output channel
    (image files under "ui"), which ComfyUI persists per workflow tab,
    so switching tabs and returning keeps the preview visible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        blend_modes = ["normal", "difference"]
        return {
            "required": {
                "image_a": ("IMAGE",),
            },
            "optional": {
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
                "blend_mode": (blend_modes, {"default": "normal"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "Y7Nodes/Image"

    def execute(self, image_a, image_b=None, prompt=None, extra_pnginfo=None, unique_id=None, blend_mode="normal"):
        # Save previews to the temp folder and hand them back to the frontend
        # through the standard "ui" channel. ComfyUI stores these in
        # app.nodeOutputs[node_id] and restores them when the workflow tab is
        # re-activated, so the preview survives tab switches.
        a_images = []
        b_images = []

        if image_a is not None:
            a_images = self.save_images(image_a, "y7_compare", prompt, extra_pnginfo)["ui"]["images"]

        if image_b is not None:
            b_images = self.save_images(image_b, "y7_compare", prompt, extra_pnginfo)["ui"]["images"]

        return {
            "ui": {"a_images": a_images, "b_images": b_images},
        }
