import torch
import torch.nn.functional as F

from comfy_api.latest import io


class Y7Nodes_ImageStitcher(io.ComfyNode):
    """Stitches 2–8 images together. The image_count widget controls how many input sockets are shown."""

    MAX_IMAGES = 8

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_ImageStitcher",
            display_name="Y7 Image Stitcher",
            category="Y7Nodes/Image",
            description="Stitches 2-8 images together, horizontally or vertically.",
            inputs=[
                io.Int.Input(
                    "image_count",
                    default=2, min=2, max=cls.MAX_IMAGES, step=1,
                    tooltip="Number of image inputs to show",
                ),
                io.Combo.Input(
                    "orientation",
                    options=["Side-by-Side (Horizontal)", "Top-and-Bottom (Vertical)"],
                    default="Side-by-Side (Horizontal)",
                ),
                *[io.Image.Input(f"image{i}", optional=True) for i in range(1, cls.MAX_IMAGES + 1)],
            ],
            outputs=[
                io.Image.Output(display_name="stitched_image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        image_count,
        orientation,
        image1=None, image2=None, image3=None, image4=None,
        image5=None, image6=None, image7=None, image8=None,
    ) -> io.NodeOutput:
        all_slots = [image1, image2, image3, image4, image5, image6, image7, image8]
        images = [img for img in all_slots[:image_count] if img is not None]

        if not images:
            raise ValueError("Y7 Image Stitcher: no images connected.")
        if len(images) == 1:
            return io.NodeOutput(images[0])

        # (B, H, W, C) → (B, C, H, W) for interpolation
        imgs_p = [img.permute(0, 3, 1, 2) for img in images]

        # Normalise batch dimension
        max_b = max(img.shape[0] for img in imgs_p)
        imgs_p = [
            img.repeat(max_b, 1, 1, 1) if img.shape[0] < max_b else img
            for img in imgs_p
        ]

        if orientation == "Side-by-Side (Horizontal)":
            ref_h = imgs_p[0].shape[2]
            resized = []
            for img in imgs_p:
                h, w = img.shape[2], img.shape[3]
                if h != ref_h:
                    new_w = max(1, int(w * ref_h / h))
                    img = F.interpolate(img, size=(ref_h, new_w), mode="bilinear", align_corners=False)
                resized.append(img)
            stitched = torch.cat(resized, dim=3)
        else:  # Top-and-Bottom (Vertical)
            ref_w = imgs_p[0].shape[3]
            resized = []
            for img in imgs_p:
                h, w = img.shape[2], img.shape[3]
                if w != ref_w:
                    new_h = max(1, int(h * ref_w / w))
                    img = F.interpolate(img, size=(new_h, ref_w), mode="bilinear", align_corners=False)
                resized.append(img)
            stitched = torch.cat(resized, dim=2)

        return io.NodeOutput(stitched.permute(0, 2, 3, 1))
