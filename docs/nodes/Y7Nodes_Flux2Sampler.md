# Flux.2 Sampler

All-in-one Flux.2 sampler: RandomNoise + KSamplerSelect + Flux2Scheduler + CFGGuider + SamplerCustomAdvanced in a single node.

Replaces the five-node chain normally needed to sample a Flux.2 or Klein model. Feed it a model, a starting latent and conditioning, and it returns the denoised latent ready for VAE decode.

There are deliberately no `width`/`height` widgets: the Flux.2 sigma schedule is derived from `latent_image`'s own dimensions, so it can never drift out of sync with the latent actually being sampled.

Inputs:

  - `model`: Diffusion model to sample with
  - `latent_image`: Starting latent to denoise — e.g. from an Empty Flux.2 Latent Image, or the `reference_latent` output of the Klein Edit node. Its dimensions also set the resolution used for the sigma schedule
  - `positive`: Positive conditioning (what to steer generation towards)
  - `negative`: Negative conditioning (what to steer away from). Effectively ignored when `cfg` is `1.0`
  - `seed`: Seed for the initial noise, with the standard randomize/increment/decrement/fixed control
  - `cfg`: Classifier-free guidance scale. Klein checkpoints are usually guidance-distilled, so the default `1.0` (negative conditioning skipped) is normally what you want. Raise it only on models that expect real CFG
  - `sampler_name`: Which k-diffusion sampler algorithm to step with. `euler` is the default
  - `steps`: Number of sampling steps. Distilled Klein checkpoints commonly need only ~4; non-distilled Flux.2 wants considerably more

Outputs:

  - `output`: The denoised latent, ready for VAE decode

Notes:

  - Any `noise_mask` carried on the incoming latent (as set by the Klein Edit node when a mask is painted) is honoured, so inpaint-style edits work without extra wiring
  - The Flux.2 sigma schedule math is vendored from `Flux2Scheduler`, so this node doesn't depend on ComfyUI's internal `comfy_extras` module
