# All-in-one Flux.2 sampler node.
#
# Normally, sampling a Flux.2 (including Klein) model means wiring together five separate nodes:
# RandomNoise (seed -> NOISE), KSamplerSelect (sampler_name -> SAMPLER), Flux2Scheduler
# (steps/width/height -> SIGMAS), CFGGuider (model/positive/negative/cfg -> GUIDER), and finally
# SamplerCustomAdvanced (noise/guider/sampler/sigmas/latent_image -> LATENT). This node compacts
# all of that into a single node with one input side (model/positive/negative/latent_image) and
# one output (the denoised latent, ready for VAE decode). Flux2Scheduler's width/height widgets are
# dropped: the sigma schedule's seq_len is derived from latent_image's own dimensions instead, so it
# can't drift out of sync with the latent being sampled.
import math

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy_api.latest import io


# Flux.2 sigma schedule math, vendored from comfy_extras/nodes_flux.py's Flux2Scheduler so this
# node doesn't depend on that internal (non-API) module. seq_len is the token count implied by
# width/height at Flux.2's 16x16-per-token latent packing.
def _flux2_generalized_time_snr_shift(t, mu: float, sigma: float):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def _flux2_compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


def _flux2_sigma_schedule(num_steps: int, image_seq_len: int) -> torch.Tensor:
    mu = _flux2_compute_empirical_mu(image_seq_len, num_steps)
    timesteps = torch.linspace(1, 0, num_steps + 1)
    timesteps = _flux2_generalized_time_snr_shift(timesteps, mu, 1.0)
    return timesteps


class Y7Nodes_Flux2Sampler(io.ComfyNode):
    """Compacts RandomNoise + KSamplerSelect + Flux2Scheduler + CFGGuider + SamplerCustomAdvanced into one node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Y7Nodes_Flux2Sampler",
            display_name="Y7 Flux.2 Sampler",
            category="Y7Nodes/Klein",
            description="All-in-one Flux.2 sampler: combines RandomNoise, KSamplerSelect, Flux2Scheduler, "
                         "CFGGuider and SamplerCustomAdvanced into a single node. There are no width/height "
                         "widgets - the sigma schedule's resolution is read from latent_image, so it always "
                         "matches the latent being sampled.",
            inputs=[
                # Diffusion model to sample with.
                io.Model.Input("model"),
                # Starting latent to denoise, e.g. from an Empty Flux.2 Latent Image or the Klein Edit node.
                # Its dimensions also set the resolution used for the Flux.2 sigma schedule (see execute()),
                # which is why this node has no width/height widgets.
                io.Latent.Input(
                    "latent_image",
                    tooltip="Latent to denoise. Its width/height also determine the Flux.2 sigma schedule, "
                            "so no separate width/height widgets are needed.",
                ),
                # Positive conditioning (what to steer generation towards).
                io.Conditioning.Input("positive"),
                # Negative conditioning (what to steer generation away from); ignored by the guider when cfg == 1.0.
                io.Conditioning.Input("negative"),
                # Seed for the initial noise, paired with the standard randomize/increment/decrement/fixed control.
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.randomize),
                # Classifier-free guidance scale. Klein checkpoints are typically guidance-distilled, so 1.0
                # (negative conditioning effectively skipped) is the usual default.
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                # Which k-diffusion sampler algorithm to step with.
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                # Number of sampling steps. Distilled Klein checkpoints commonly only need ~4.
                io.Int.Input("steps", default=4, min=1, max=4096),
            ],
            outputs=[
                # Denoised latent, ready for VAE decode.
                io.Latent.Output(display_name="output"),
            ],
        )

    @classmethod
    def execute(cls, model, latent_image, positive, negative, seed, cfg, sampler_name, steps) -> io.NodeOutput:
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)

        sampler = comfy.samplers.sampler_object(sampler_name)

        latent = latent_image.copy()
        samples = latent["samples"]
        samples = comfy.sample.fix_empty_latent_channels(
            model, samples, latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples

        # Flux.2 latents are 16x-downscaled ([batch, 128, height // 16, width // 16]), so the latent's own
        # spatial dims *are* the token grid the sigma schedule needs - no width/height widgets to keep in sync.
        seq_len = samples.shape[-2] * samples.shape[-1]
        sigmas = _flux2_sigma_schedule(steps, seq_len)

        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index", None))
        noise_mask = latent.get("noise_mask", None)

        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        result = guider.sample(noise, samples, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        result = result.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = result

        return io.NodeOutput(out)
