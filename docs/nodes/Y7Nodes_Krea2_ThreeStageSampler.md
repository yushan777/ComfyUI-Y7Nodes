# Krea 2 Three-Stage Sampler

The Krea 2 Two-Stage Sampler plus a final pass that reuses all of stage 1's settings from `stage3_handoff_percent` onward.

Stage 1 runs up to `handoff_percent`, stage 2 up to `stage3_handoff_percent`, then stage 1's model, steps, cfg, sampler and scheduler finish the job. Useful when you want negative conditioning (stage 1 cfg) at the start and end of a generation but a fast cfg-1 model in the middle.

## Inputs

  - `handoff_percent`: Stage 1 to stage 2 handoff. 0% skips stage 1
  - `stage3_handoff_percent`: Stage 2 to stage 3 handoff. Must be at or after `handoff_percent`. 100% means no stage 3 (identical to the Krea 2 Two-Stage Sampler); equal to `handoff_percent` skips stage 2
  - Everything else is as on the Krea 2 Two-Stage Sampler
  - Stages 1 and 3 always use `positive` / `negative`; the `stage2_*_optional` inputs only affect stage 2

## Outputs

  - `latent`: The finished latent, ready for VAE decode

## Notes [collapsed]

  - Any resize from `final_width` / `final_height` happens at the first handoff; later stages continue at the final size
  - The two handoffs must land on different sigmas. If they round to the same step, move them farther apart


