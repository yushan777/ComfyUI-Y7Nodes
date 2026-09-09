from comfy_api.latest import io


# ===========================================================
class Y7Nodes_Text(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_Text",
            display_name="Y7 Text",
            category="Y7Nodes/Utils",
            description="A multiline text box whose contents are passed through to the output.",
            inputs=[
                io.String.Input(
                    "text",
                    optional=True,
                    default="",
                    multiline=True,
                    tooltip="Text input that will be passed through to the output",
                ),
            ],
            outputs=[
                io.String.Output(display_name="text_out"),
            ],
        )

    @classmethod
    def execute(cls, text=None) -> io.NodeOutput:
        # Return values must match the schema's output order
        return io.NodeOutput(text)
