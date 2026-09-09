import comfy
from comfy_api.latest import io


# The output socket has to be typed as the sampler list itself, not as a plain "COMBO".
# ComfyUI validates links by comparing the upstream RETURN_TYPES entry against the
# downstream input type, and KSampler declares sampler_name as the raw SAMPLERS list.
# A stock io.Combo.Output would report "COMBO" and fail that comparison, so declare a
# custom type whose io_type *is* the list, exactly as the V1 RETURN_TYPES did.
@io.comfytype(io_type=comfy.samplers.KSampler.SAMPLERS)
class SamplerNameType(io.ComfyTypeIO):
    Type = str


# Exposes the sampler name as a linkable node output.
#
# ComfyUI's built-in KSampler and similar nodes define sampler_name as a "COMBO"
# input (a hardcoded dropdown list), not a STRING. COMBO inputs are rendered as
# widgets and cannot accept incoming node connections — they have no input socket.
# Since no built-in node outputs a SAMPLER name as a plain STRING either, there's
# no standard way to wire a sampler choice dynamically from another node.
#
# This node works around that by declaring its output type as KSampler.SAMPLERS
# (the same combo list), which ComfyUI recognises as a compatible type for any
# node that accepts sampler_name. That gives it a linkable output socket.
class SamplerSelect_Name(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SamplerSelect_Name",
            display_name="Y7 Sampler Select (Name)",
            category="Y7Nodes",
            description="Exposes the sampler name as a linkable node output.",
            inputs=[
                # dropdown of available samplers
                io.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
            ],
            outputs=[
                SamplerNameType.Output(display_name="sampler_name"),
            ],
        )

    @classmethod
    def execute(cls, sampler_name) -> io.NodeOutput:
        return io.NodeOutput(sampler_name)
