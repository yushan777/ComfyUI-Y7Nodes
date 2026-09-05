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
  - `denoise`: How much of the latent to redo. `1.0` (the default) starts from pure noise, which is what you want for normal generation; `0` hands the latent back untouched. Lowering it trims the front off the schedule, which shortens the run as well as the starting noise level. It is rarely the dial you want on Flux.2 — see the note below — so leave it at `1.0` unless you have a specific reason

Outputs:

  - `output`: The denoised latent, ready for VAE decode

Notes:

  - Any `noise_mask` carried on the incoming latent (as set by the Klein Edit node when a mask is painted) is honoured, so inpaint-style edits work without extra wiring
  - `denoise` here does exactly what wiring a `SplitSigmasDenoise` node between `Flux2Scheduler` and the sampler and taking its `low_sigmas` output does — deliberately, so this node stays a faithful drop-in for that chain
  - Do not expect it to work as an image-to-image strength dial. Flux.2 shifts its schedule hard toward the high-noise end (that shift is what lets distilled Klein checkpoints work in 4 steps), so at 1024x1024 / 4 steps the whole schedule is `1.000, 0.967, 0.908, 0.767, 0.000`. Trimming the front barely lowers where you start: `denoise 0.75` still begins at sigma `0.967` and keeps only ~3% of the image, while also cutting you to 3 steps. At `steps 4` there are only five reachable settings in total
  - To preserve an existing image with a Flux.2 edit model, use the `Klein Edit` node's conditioning instead and leave `denoise` at `1.0`. Edit conditioning sets `concat_latent_image`, which the model reads on every step — that is what holds your source image, not the noise level. The two mechanisms fight each other, and the conditioning wins
  - The Flux.2 sigma schedule math is vendored from `Flux2Scheduler`, so this node doesn't depend on ComfyUI's internal `comfy_extras` module
