# Klein & Qwen Edit Align

Snaps an edited image back onto the original's geometry and colour-matches it to the original's unedited background.

Flux.2 Klein and Qwen-Image-Edit often return the whole frame slightly shifted, rescaled or tinted. This node corrects that with a single SIFT homography (translation, scale, rotation, mild perspective) followed by a Reinhard colour transfer in LAB. It does not correct uneven local warping, and it does no compositing.

## Inputs

  - `edited_image`: The edit model's output
  - `original_image`: The image that was edited. Resized to the edited image's size if they differ
  - `color_match_strength`: 0.0 = no colour match, 1.0 = full transfer
  - `two_pass`: Re-align using only unedited background so the edit can't bias the solve. Slower but noticeably more accurate
  - `edit_mask` (optional): White = edited region. Overrides the automatic detection used for pass 2 and the colour statistics

## Outputs

  - `aligned_image`: The edited image, aligned and colour-matched
  - `valid_mask`: 0 where the warp pulled from outside the frame. Those pixels are filled by reflection, so use this mask when compositing
  - `report`: Inliers per pass, the homography determinant, and whether colour match ran

## Reading the report [collapsed]

  - `determinant` near 1.0 means little scaling was needed
  - `Pass 2: skipped` or `failed` means you are getting pass-1 accuracy, usually because the edit covers most of the frame
  - `Color match: skipped` means fewer than 100 usable background pixels, or strength 0
  - `Alignment: FAILED` means too few matching features; the image is returned unaligned

Requires `opencv-python-headless` (included in `requirements.txt`).
