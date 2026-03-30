import comfy


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
class SamplerSelect_Name:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),  # dropdown of available samplers
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler_name",)
    FUNCTION = "select_sampler"
    CATEGORY = "Y7Nodes"

    def select_sampler(self, sampler_name):
        return (sampler_name,)
