import sys

from comfy_api.latest import io


# The increment/decrement itself happens in the browser (web/js/float.js), which steps the
# `value` widget when a prompt is queued, exactly as the native control_after_generate widget
# does for seeds. The backend only ever sees, and passes through, the value it was sent.
class Y7Nodes_Float(io.ComfyNode):

    CONTROL_MODES = ["fixed", "increment", "decrement"]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_Float",
            display_name="Y7 Float",
            category="Y7Nodes/Utils",
            description="A float value that can step itself up or down by a set amount after each run.",
            inputs=[
                io.Float.Input(
                    "value",
                    default=0.0,
                    min=-sys.maxsize,
                    max=sys.maxsize,
                    step=0.01,
                    tooltip="The value sent to the output",
                ),
                io.Combo.Input(
                    "control_after_generate",
                    options=cls.CONTROL_MODES,
                    default="fixed",
                    tooltip="fixed leaves the value alone. increment / decrement move it by step each time a prompt is queued.",
                ),
                io.Float.Input(
                    "step",
                    default=0.1,
                    min=0.0,
                    max=sys.maxsize,
                    step=0.01,
                    tooltip="How much increment / decrement moves the value by. Ignored when control_after_generate is fixed.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="value"),
            ],
        )

    @classmethod
    def execute(cls, value, control_after_generate="fixed", step=0.1) -> io.NodeOutput:
        return io.NodeOutput(value)
