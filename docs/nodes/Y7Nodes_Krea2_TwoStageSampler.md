# Krea 2 Two-Stage Sampler

Samples with one model up to `handoff_percent`, then finishes with a second model from the exact sigma where the first stopped.

A typical pairing is a raw/base model for stage 1 (better variation between seeds) and a turbo model or LoRA for stage 2 (speed and finish). Each stage keeps its own steps, cfg, sampler and scheduler; stage 2's schedule is cut at its step nearest the handoff sigma, so the two can use different step counts.

## Inputs

  - `stage1_model` / `stage2_model`: The model for each stage. Can be the same model with different settings
  - `positive` / `negative`: Conditioning for stage 1 (and stage 2 unless overridden)
  - `latent_image`: Starting latent, at the base resolution
  - `handoff_percent`: Where stage 1 hands off. 25% = stage 1 does the first quarter of denoising. 0% runs only stage 2; 100% runs only stage 1
  - `stage1_*` / `stage2_*`: Steps, cfg, sampler and scheduler for each stage. A stage's steps cover its whole schedule, not just its share
  - `final_width` / `final_height`: Resize the latent at the handoff so stage 2 runs at this size. 0 for one dimension keeps the aspect ratio; 0 for both disables the resize
  - `upscale_method` (advanced): Latent resize method used at the handoff
  - `stage2_positive_optional` / `stage2_negative_optional`: Separate conditioning for stage 2. Unconnected inputs reuse `positive` / `negative`

## Outputs

  - `latent`: The finished latent, ready for VAE decode

## Notes [collapsed]

  - Without a resize, stage 1's leftover noise carries straight into stage 2
  - With a resize, stage 1 runs to a clean estimate, the latent is resized, and stage 2 adds fresh noise at the handoff sigma
  - At stage 2 cfg 1, the stage 2 negative is zeroed
  - Both models need compatible model sampling (shift) settings, or the handoff sigma may fall outside stage 2's schedule and raise an error


