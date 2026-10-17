"""
Y7Nodes Krea 2 Two-Stage / Three-Stage Sampler

Sigma-locked multi-model samplers. Stage 1 runs the first part of its own schedule, then hands the
partly denoised latent to stage 2 at the exact sigma it stopped at; stage 2's schedule is cut to
start at its nearest matching step. The usual pairing is a raw/base model for stage 1 (better
seed-to-seed variation) and a turbo/distilled model or LoRA for stage 2 (speed, finish quality).

Stage 2 can optionally run at a higher resolution: stage 1 is then sampled down to a clean estimate,
the latent is resized, and stage 2 adds fresh noise at the handoff sigma. Without a resize, stage 1's
leftover noise is carried straight into stage 2.

The three-stage variant adds a final pass that reuses all of stage 1's settings (model, steps, cfg,
sampler, scheduler) from a second handoff point onward.

Cloned from krea2-twostage-sampler by Auryg (MIT licence), ported to the V3 schema.
"""

import math

import torch

import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import nodes
from comfy_api.latest import io

from ..utils.logger import logger


UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


# ---------------------------------------------------------------------------
# Sigma schedules
# ---------------------------------------------------------------------------

def _sigma_schedule(model, steps, sampler_name, scheduler):
    device = getattr(model, "load_device", comfy.model_management.get_torch_device())
    sampler = comfy.samplers.KSampler(
        model,
        steps=steps,
        device=device,
        sampler=sampler_name,
        scheduler=scheduler,
        denoise=1.0,
        model_options=model.model_options,
    )
    return sampler.sigmas.detach().cpu()


def _percent_schedule_index(steps, sigmas, handoff_percent, stage_name):
    max_index = len(sigmas) - 2
    if max_index < 1:
        raise ValueError(f"{stage_name} must produce at least one non-terminal handoff sigma.")
    requested = round(steps * handoff_percent / 100.0)
    return max(1, min(max_index, requested))


def _nearest_schedule_index(sigmas, boundary_sigma, minimum_index, stage_name):
    max_index = len(sigmas) - 2
    if max_index < minimum_index:
        raise ValueError(f"{stage_name} does not have enough non-terminal sigmas for this handoff.")

    boundary = float(boundary_sigma)
    if boundary <= 0.0:
        raise ValueError("The handoff sigma must be greater than zero.")
    if boundary >= float(sigmas[0]):
        raise ValueError(
            f"The handoff sigma is outside the {stage_name} schedule. "
            "Use compatible model sampling settings or move the handoff later."
        )

    candidates = [idx for idx in range(minimum_index, max_index + 1) if float(sigmas[idx - 1]) > boundary]
    if not candidates:
        raise ValueError(
            f"Could not find a valid {stage_name} handoff step. "
            "Use compatible model sampling settings or move the handoffs farther apart."
        )

    return min(candidates, key=lambda idx: abs(float(sigmas[idx]) - boundary))


def _build_sigma_pair(stage1_model, stage2_model, stage1_steps, stage2_steps, stage1_sampler, stage1_scheduler,
                      stage2_sampler, stage2_scheduler, handoff_percent):
    stage1_sigmas = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler, stage1_scheduler)
    stage2_sigmas = _sigma_schedule(stage2_model, stage2_steps, stage2_sampler, stage2_scheduler)

    stage1_end = _percent_schedule_index(stage1_steps, stage1_sigmas, handoff_percent, "stage 1")
    boundary_sigma = stage1_sigmas[stage1_end].clone()
    stage2_start = _nearest_schedule_index(stage2_sigmas, boundary_sigma, 1, "stage 2")

    stage1_custom = stage1_sigmas[:stage1_end + 1].clone()
    stage2_custom = stage2_sigmas[stage2_start:].clone()
    stage2_custom[0] = boundary_sigma

    return stage1_custom, stage2_custom, stage1_end, stage2_start, float(boundary_sigma)


def _build_sigma_triplet(stage1_model, stage2_model, stage1_steps, stage2_steps, stage1_sampler, stage1_scheduler,
                         stage2_sampler, stage2_scheduler, handoff_percent, stage3_handoff_percent):
    stage1_full = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler, stage1_scheduler)
    stage2_full = _sigma_schedule(stage2_model, stage2_steps, stage2_sampler, stage2_scheduler)

    stage3_start = _percent_schedule_index(stage1_steps, stage1_full, stage3_handoff_percent, "stage 1")
    stage2_end_sigma = stage1_full[stage3_start].clone()

    stage1_custom = None
    stage1_end = None
    stage1_end_sigma = None
    if handoff_percent <= 0.0:
        stage2_start = 0
    else:
        stage1_end = _percent_schedule_index(stage1_steps, stage1_full, handoff_percent, "stage 1")
        stage1_end_sigma = stage1_full[stage1_end].clone()
        stage1_custom = stage1_full[:stage1_end + 1].clone()
        stage2_start = _nearest_schedule_index(stage2_full, stage1_end_sigma, 1, "stage 2")

        if float(stage2_end_sigma) >= float(stage1_end_sigma):
            raise ValueError(
                "stage3_handoff_percent must produce a later denoising sigma than handoff_percent. "
                "Move the handoffs farther apart or use compatible model sampling settings."
            )

    stage2_end = _nearest_schedule_index(stage2_full, stage2_end_sigma, stage2_start + 1, "stage 2")
    stage2_custom = torch.cat((stage2_full[stage2_start:stage2_end], stage2_end_sigma.reshape(1)))
    if stage1_end_sigma is not None:
        stage2_custom[0] = stage1_end_sigma

    stage3_custom = stage1_full[stage3_start:].clone()
    stage3_custom[0] = stage2_end_sigma

    return (
        stage1_custom,
        stage2_custom,
        stage3_custom,
        stage1_end,
        stage2_start,
        stage2_end,
        stage3_start,
        None if stage1_end_sigma is None else float(stage1_end_sigma),
        float(stage2_end_sigma),
    )


# ---------------------------------------------------------------------------
# Latent resize between stages
# ---------------------------------------------------------------------------

def _latent_spatial_downscale(model):
    latent_format = model.get_model_object("latent_format")
    return getattr(latent_format, "spacial_downscale_ratio", 8)


def _target_latent_size(latent, model, final_width, final_height):
    """Target latent W/H. A 0 dimension is derived from the other, keeping the current aspect ratio."""
    samples = latent["samples"]
    downscale = _latent_spatial_downscale(model)
    current_width = samples.shape[-1] * downscale
    current_height = samples.shape[-2] * downscale

    if final_width == 0:
        final_width = max(64, round(current_width * final_height / current_height))
    elif final_height == 0:
        final_height = max(64, round(current_height * final_width / current_width))

    final_width = max(64, int(final_width))
    final_height = max(64, int(final_height))

    return max(1, round(final_width / downscale)), max(1, round(final_height / downscale))


def _will_upscale_latent(latent, model, final_width, final_height):
    if final_width == 0 and final_height == 0:
        return False

    target_latent_width, target_latent_height = _target_latent_size(latent, model, final_width, final_height)
    samples = latent["samples"]
    return target_latent_width != samples.shape[-1] or target_latent_height != samples.shape[-2]


def _upscale_latent_if_needed(latent, model, final_width, final_height, upscale_method):
    """Returns (latent, resized). Untouched when both final dims are 0 or already match."""
    if final_width == 0 and final_height == 0:
        return latent, False

    samples = latent["samples"]
    target_latent_width, target_latent_height = _target_latent_size(latent, model, final_width, final_height)

    if target_latent_width == samples.shape[-1] and target_latent_height == samples.shape[-2]:
        return latent, False

    out = latent.copy()
    out["samples"] = comfy.utils.common_upscale(samples, target_latent_width, target_latent_height, upscale_method, "disabled")
    return out, True


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _sample_with_sigmas(model, seed, cfg, sampler_name, scheduler, positive, negative, latent, sigmas, disable_noise):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(
        model,
        latent_image,
        latent.get("downscale_ratio_spacial", None),
        latent.get("downscale_ratio_temporal", None),
    )

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = latent.get("noise_mask", None)
    steps = max(1, len(sigmas) - 1)
    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    device = getattr(model, "load_device", comfy.model_management.get_torch_device())

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise=1.0,
        disable_noise=disable_noise,
        force_full_denoise=False,
        noise_mask=noise_mask,
        sigmas=sigmas.to(device),
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    out = latent.copy()
    out.pop("downscale_ratio_spacial", None)
    out.pop("downscale_ratio_temporal", None)
    out["samples"] = samples
    return out


def _zero_out_conditioning(conditioning):
    out = []
    for item in conditioning:
        metadata = item[1].copy()
        pooled_output = metadata.get("pooled_output", None)
        if pooled_output is not None:
            metadata["pooled_output"] = torch.zeros_like(pooled_output)
        conditioning_lyrics = metadata.get("conditioning_lyrics", None)
        if conditioning_lyrics is not None:
            metadata["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
        out.append([torch.zeros_like(item[0]), metadata])
    return out


def _stage2_negative(negative, cfg):
    """Stage 2's negative, zeroed when its cfg is 1."""
    return _zero_out_conditioning(negative) if math.isclose(cfg, 1.0, rel_tol=0.0, abs_tol=1e-6) else negative


def _run_two_stage(stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
                   stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
                   stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
                   final_width, final_height, upscale_method, stage2_positive, stage2_negative):
    if handoff_percent <= 0.0:
        stage2_sigmas = _sigma_schedule(stage2_model, stage2_steps, stage2_sampler_name, stage2_scheduler)
        stage2_input, did_upscale = _upscale_latent_if_needed(latent_image, stage2_model, final_width, final_height, upscale_method)
        logger.info(
            "[Y7 Krea 2 Two-Stage Sampler] "
            f"stage2_only=true, stage2_steps={len(stage2_sigmas) - 1}, resized_input={str(did_upscale).lower()}"
        )
        return _sample_with_sigmas(
            stage2_model, seed, stage2_cfg, stage2_sampler_name, stage2_scheduler,
            stage2_positive, _stage2_negative(stage2_negative, stage2_cfg),
            stage2_input, stage2_sigmas, disable_noise=False,
        )

    if handoff_percent >= 100.0:
        stage1_sigmas = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler_name, stage1_scheduler)
        logger.info(f"[Y7 Krea 2 Two-Stage Sampler] stage1_only=true, stage1_steps={len(stage1_sigmas) - 1}")
        return _sample_with_sigmas(
            stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            positive, negative, latent_image, stage1_sigmas, disable_noise=False,
        )

    stage1_sigmas, stage2_sigmas, stage1_end, stage2_start, boundary_sigma = _build_sigma_pair(
        stage1_model, stage2_model, stage1_steps, stage2_steps,
        stage1_sampler_name, stage1_scheduler, stage2_sampler_name, stage2_scheduler,
        handoff_percent,
    )
    # When resizing, run stage 1 to a clean estimate (final sigma 0) so the resize doesn't smear noise;
    # stage 2 then re-noises at the boundary sigma
    upscale_requested = _will_upscale_latent(latent_image, stage2_model, final_width, final_height)
    stage1_run_sigmas = stage1_sigmas.clone()
    if upscale_requested:
        stage1_run_sigmas[-1] = 0.0

    logger.info(
        "[Y7 Krea 2 Two-Stage Sampler] "
        f"stage1_end={stage1_end}, stage2_start={stage2_start}, boundary_sigma={boundary_sigma:.8f}, "
        f"noise_mode={'fresh_high_res' if upscale_requested else 'carry_leftover'}"
    )

    stage1 = _sample_with_sigmas(
        stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
        positive, negative, latent_image, stage1_run_sigmas, disable_noise=False,
    )
    stage1, did_upscale = _upscale_latent_if_needed(stage1, stage2_model, final_width, final_height, upscale_method)

    return _sample_with_sigmas(
        stage2_model, seed, stage2_cfg, stage2_sampler_name, stage2_scheduler,
        stage2_positive, _stage2_negative(stage2_negative, stage2_cfg),
        stage1, stage2_sigmas, disable_noise=not did_upscale,
    )


def _run_three_stage(stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
                     stage3_handoff_percent, stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
                     stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
                     final_width, final_height, upscale_method, stage2_positive, stage2_negative):
    if stage3_handoff_percent < handoff_percent:
        raise ValueError("stage3_handoff_percent must be greater than or equal to handoff_percent.")

    # No stage 3 at all: identical to the two-stage sampler
    if stage3_handoff_percent >= 100.0:
        return _run_two_stage(
            stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
            stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
            final_width, final_height, upscale_method, stage2_positive, stage2_negative,
        )

    # Both handoffs at the same point: stage 2 is skipped, stage 1 settings run either side of it
    if math.isclose(stage3_handoff_percent, handoff_percent, rel_tol=0.0, abs_tol=1e-9):
        if stage3_handoff_percent <= 0.0:
            stage3_sigmas = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler_name, stage1_scheduler)
            stage3_input, did_upscale = _upscale_latent_if_needed(latent_image, stage1_model, final_width, final_height, upscale_method)
            logger.info(
                "[Y7 Krea 2 Three-Stage Sampler] "
                f"stage3_only=true, stage3_steps={len(stage3_sigmas) - 1}, resized_input={str(did_upscale).lower()}"
            )
            return _sample_with_sigmas(
                stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
                positive, negative, stage3_input, stage3_sigmas, disable_noise=False,
            )

        stage1_full = _sigma_schedule(stage1_model, stage1_steps, stage1_sampler_name, stage1_scheduler)
        stage3_start = _percent_schedule_index(stage1_steps, stage1_full, stage3_handoff_percent, "stage 1")
        boundary_sigma = stage1_full[stage3_start].clone()
        stage1_sigmas = stage1_full[:stage3_start + 1].clone()
        stage3_sigmas = stage1_full[stage3_start:].clone()
        upscale_requested = _will_upscale_latent(latent_image, stage1_model, final_width, final_height)
        stage1_run_sigmas = stage1_sigmas.clone()
        if upscale_requested:
            stage1_run_sigmas[-1] = 0.0

        logger.info(
            "[Y7 Krea 2 Three-Stage Sampler] "
            f"stage2_skipped=true, stage1_end={stage3_start}, "
            f"stage3_start={stage3_start}, boundary_sigma={float(boundary_sigma):.8f}, "
            f"noise_mode={'fresh_high_res' if upscale_requested else 'carry_leftover'}"
        )
        stage1 = _sample_with_sigmas(
            stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            positive, negative, latent_image, stage1_run_sigmas, disable_noise=False,
        )
        stage3_input, did_upscale = _upscale_latent_if_needed(stage1, stage1_model, final_width, final_height, upscale_method)
        return _sample_with_sigmas(
            stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            positive, negative, stage3_input, stage3_sigmas, disable_noise=not did_upscale,
        )

    (
        stage1_sigmas,
        stage2_sigmas,
        stage3_sigmas,
        stage1_end,
        stage2_start,
        stage2_end,
        stage3_start,
        stage1_boundary_sigma,
        stage2_boundary_sigma,
    ) = _build_sigma_triplet(
        stage1_model, stage2_model, stage1_steps, stage2_steps,
        stage1_sampler_name, stage1_scheduler, stage2_sampler_name, stage2_scheduler,
        handoff_percent, stage3_handoff_percent,
    )

    if stage1_sigmas is None:
        stage2_input, did_upscale = _upscale_latent_if_needed(latent_image, stage2_model, final_width, final_height, upscale_method)
        stage2_disable_noise = False
        stage1_log = "stage1_skipped=true"
    else:
        upscale_requested = _will_upscale_latent(latent_image, stage2_model, final_width, final_height)
        stage1_run_sigmas = stage1_sigmas.clone()
        if upscale_requested:
            stage1_run_sigmas[-1] = 0.0
        stage1 = _sample_with_sigmas(
            stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            positive, negative, latent_image, stage1_run_sigmas, disable_noise=False,
        )
        stage2_input, did_upscale = _upscale_latent_if_needed(stage1, stage2_model, final_width, final_height, upscale_method)
        stage2_disable_noise = not did_upscale
        stage1_log = f"stage1_end={stage1_end}, stage1_boundary_sigma={stage1_boundary_sigma:.8f}"

    logger.info(
        "[Y7 Krea 2 Three-Stage Sampler] "
        f"{stage1_log}, stage2_start={stage2_start}, stage2_end={stage2_end}, "
        f"stage3_start={stage3_start}, "
        f"stage2_boundary_sigma={stage2_boundary_sigma:.8f}, "
        f"noise_mode={'fresh_high_res' if not stage2_disable_noise else 'carry_leftover'}"
    )
    stage2 = _sample_with_sigmas(
        stage2_model, seed, stage2_cfg, stage2_sampler_name, stage2_scheduler,
        stage2_positive, _stage2_negative(stage2_negative, stage2_cfg),
        stage2_input, stage2_sigmas, disable_noise=stage2_disable_noise,
    )
    return _sample_with_sigmas(
        stage1_model, seed, stage1_cfg, stage1_sampler_name, stage1_scheduler,
        positive, negative, stage2, stage3_sigmas, disable_noise=True,
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def _handoff_input():
    return io.Float.Input(
        "handoff_percent",
        default=12.0, min=0.0, max=100.0, step=0.01, round=0.01,
        tooltip="Point in the denoising process where stage 1 hands off to stage 2. "
                "0% uses only stage 2; 100% uses only stage 1.",
    )


def _stage3_handoff_input():
    return io.Float.Input(
        "stage3_handoff_percent",
        default=83.33, min=0.0, max=100.0, step=0.01, round=0.01,
        tooltip="Point in the denoising process where stage 2 hands off to stage 3. Stage 3 reuses all "
                "stage 1 settings. Must be greater than or equal to handoff_percent.",
    )


def _inputs(with_stage3):
    """Shared input list; the three-stage node inserts stage3_handoff_percent after handoff_percent."""
    handoffs = [_handoff_input(), _stage3_handoff_input()] if with_stage3 else [_handoff_input()]
    return [
        io.Model.Input("stage1_model"),
        io.Model.Input("stage2_model"),
        io.Conditioning.Input("positive"),
        io.Conditioning.Input("negative"),
        io.Latent.Input("latent_image"),
        io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.randomize),
        *handoffs,
        io.Int.Input("stage1_steps", default=52, min=2, max=200),
        io.Float.Input("stage1_cfg", default=4.0, min=0.0, max=100.0, step=0.1, round=0.01),
        io.Combo.Input("stage1_sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
        io.Combo.Input("stage1_scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
        io.Int.Input("stage2_steps", default=8, min=2, max=200),
        io.Float.Input("stage2_cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
        io.Combo.Input("stage2_sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
        io.Combo.Input("stage2_scheduler", options=comfy.samplers.KSampler.SCHEDULERS),
        io.Int.Input(
            "final_width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8,
            tooltip="Resolution for stage 2 onward. 0 = derive from final_height (or no resize if both are 0).",
        ),
        io.Int.Input(
            "final_height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=8,
            tooltip="Resolution for stage 2 onward. 0 = derive from final_width (or no resize if both are 0).",
        ),
        io.Combo.Input(
            "upscale_method", options=UPSCALE_METHODS, default="bislerp", advanced=True,
            tooltip="How the latent is resized at the handoff.",
        ),
        io.Conditioning.Input(
            "stage2_positive_optional", optional=True,
            tooltip="Stage 2 positive conditioning. Uses positive when unconnected.",
        ),
        io.Conditioning.Input(
            "stage2_negative_optional", optional=True,
            tooltip="Stage 2 negative conditioning. Uses negative when unconnected; zeroed at stage 2 cfg 1.",
        ),
    ]


class Y7Nodes_Krea2_TwoStageSampler(io.ComfyNode):
    """Sigma-locked two-model sampler with an optional resolution change at the handoff."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_Krea2_TwoStageSampler",
            display_name="Y7 Krea 2 Two-Stage Sampler",
            category="Y7Nodes/Sampling",
            description="Runs stage 1 with one model up to handoff_percent, then finishes with a second model "
                        "from the matching sigma, optionally at a higher resolution.",
            inputs=_inputs(with_stage3=False),
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
                stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
                stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
                final_width, final_height, upscale_method="bislerp",
                stage2_positive_optional=None, stage2_negative_optional=None) -> io.NodeOutput:
        return io.NodeOutput(_run_two_stage(
            stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
            stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
            final_width, final_height, upscale_method,
            positive if stage2_positive_optional is None else stage2_positive_optional,
            negative if stage2_negative_optional is None else stage2_negative_optional,
        ))


class Y7Nodes_Krea2_ThreeStageSampler(io.ComfyNode):
    """Two-stage sampler plus a final pass that reuses stage 1's settings."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_Krea2_ThreeStageSampler",
            display_name="Y7 Krea 2 Three-Stage Sampler",
            category="Y7Nodes/Sampling",
            description="Stage 1 model up to handoff_percent, stage 2 model up to stage3_handoff_percent, "
                        "then stage 1's model and settings again to finish.",
            inputs=_inputs(with_stage3=True),
            outputs=[io.Latent.Output(display_name="latent")],
        )

    @classmethod
    def execute(cls, stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
                stage3_handoff_percent, stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
                stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
                final_width, final_height, upscale_method="bislerp",
                stage2_positive_optional=None, stage2_negative_optional=None) -> io.NodeOutput:
        return io.NodeOutput(_run_three_stage(
            stage1_model, stage2_model, positive, negative, latent_image, seed, handoff_percent,
            stage3_handoff_percent, stage1_steps, stage1_cfg, stage1_sampler_name, stage1_scheduler,
            stage2_steps, stage2_cfg, stage2_sampler_name, stage2_scheduler,
            final_width, final_height, upscale_method,
            positive if stage2_positive_optional is None else stage2_positive_optional,
            negative if stage2_negative_optional is None else stage2_negative_optional,
        ))
