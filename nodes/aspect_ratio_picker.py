class Y7Nodes_AspectRatioPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Xi": ("INT", {"default": 512, "min": 0, "max": 8192}),
                "Yi": ("INT", {"default": 512, "min": 0, "max": 8192}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("X","Y",)

    FUNCTION = "main"
    CATEGORY = 'Y7Nodes/Utils'

    def main(self, Xi, Yi):
        return (Xi, Yi,)
