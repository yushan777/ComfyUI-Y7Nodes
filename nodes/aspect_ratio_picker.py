from comfy_api.latest import io


class Y7Nodes_AspectRatioPicker(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_AspectRatioPicker",
            display_name="Y7 Aspect Ratio Picker",
            category="Y7Nodes/Utils",
            inputs=[
                io.Int.Input("Xi", default=512, min=0, max=8192),
                io.Int.Input("Yi", default=512, min=0, max=8192),
            ],
            outputs=[
                io.Int.Output(display_name="X"),
                io.Int.Output(display_name="Y"),
            ],
        )

    @classmethod
    def execute(cls, Xi, Yi) -> io.NodeOutput:
        return io.NodeOutput(Xi, Yi)
