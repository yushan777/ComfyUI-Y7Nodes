# A MODEL-only take on Impact Pack's "Switch (Any)" (ImpactSwitch): several MODEL sockets, a
# `select` number, and only the selected model comes out.
#
# The sockets are declared with io.Autogrow, so the node starts with one empty `model_1` slot and a
# new one appears each time the last is connected, up to model_10. Impact does the same thing with
# its own JS and a validation bypass; Autogrow gets it from ComfyUI directly.
#
# Every socket is lazy, and check_lazy_status only asks for the selected one. That's the point of
# the node: the unselected loaders upstream (Load Diffusion Model, Load Checkpoint, a LoRA chain...)
# never run, so their weights are never read from disk.
#
# How ComfyUI hands Autogrow inputs to check_lazy_status (comfy_api/latest/_io.py,
# build_nested_inputs with create_dynamic_tuple): `models` arrives as
#   {"model_1": (value, "models.model_1"), "model_2": (value, "models.model_2"), ...}
# holding only the connected sockets, with value None until that socket has been evaluated. The
# name to return is the second element, the full dotted key, not the bare "model_N". With no socket
# connected at all, `models` is instead the empty-dict default wrapped the same way: ({}, "models").
from comfy_api.latest import io

from ..utils.logger import logger


MAX_MODELS = 10
MODEL_NAMES = [f"model_{i}" for i in range(1, MAX_MODELS + 1)]


class Y7Nodes_SwitchModel(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        models_template = io.Autogrow.TemplateNames(
            input=io.Model.Input(
                "model",
                optional=True,
                lazy=True,
                tooltip="A model to choose from. Only the selected one is loaded.",
            ),
            names=MODEL_NAMES,
            min=1,
        )

        return io.Schema(
            node_id="Y7Nodes_SwitchModel",
            display_name="Switch (model)",
            category="Y7Nodes/Utils",
            description="Pick one of several connected models by number. Only the selected model is loaded.",
            inputs=[
                io.Int.Input(
                    "select",
                    default=1,
                    min=1,
                    max=MAX_MODELS,
                    step=1,
                    tooltip="Which model_N socket to pass through. If that one isn't connected, the nearest connected socket below it is used.",
                ),
                io.Autogrow.Input(
                    "models",
                    template=models_template,
                    tooltip="Connect one and another empty socket appears, up to ten in total.",
                ),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Int.Output(display_name="selected_index", tooltip="The socket number actually used, after clamping."),
            ],
        )

    @classmethod
    def check_lazy_status(cls, select, models=None):
        # With nothing connected, `models` is the tuple ({}, "models") rather than a dict
        if not isinstance(models, dict):
            return []
        index = _resolve_index(select, models)
        if index is None:
            # Nothing connected - execute raises the error
            return []
        value, key = models[f"model_{index}"]
        return [key] if value is None else []

    @classmethod
    def execute(cls, select, models: io.Autogrow.Type = None) -> io.NodeOutput:
        models = models or {}
        index = _resolve_index(select, models)
        if index is None:
            raise ValueError("Switch (model): no model is connected")
        if index != select:
            logger.info(f"Switch (model): model_{select} is not connected, using model_{index}")
        return io.NodeOutput(models[f"model_{index}"], index)


def _resolve_index(select, models):
    """The socket number to use: `select` if connected, else the highest connected socket below it,
    else the lowest connected one. None when nothing is connected.

    A `select` past the last socket clamps to it, and one pointing at a gap left by disconnecting a
    middle socket falls back to the one before. check_lazy_status and execute both go through here so
    the socket evaluated is always the one returned.
    """
    connected = [i for i in range(1, MAX_MODELS + 1) if f"model_{i}" in models]
    if not connected:
        return None
    at_or_below = [i for i in connected if i <= select]
    return at_or_below[-1] if at_or_below else connected[0]
