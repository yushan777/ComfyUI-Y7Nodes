"""
Y7Nodes Klein & Qwen Edit Align

Flux.2 Klein and Qwen-Image-Edit often hand back the whole frame slightly shifted, rescaled or
tinted. This node snaps the edited image back onto the original's geometry with a single SIFT
homography, then Reinhard colour-matches it to the original using the unedited background only.

Extracted from the alignment and colour-match stages of comfyui-klein-edit-composite. The change
detection, optical flow, morphology and compositing are left out; the only change detection kept
is a coarse mask used for two things:
  - pass 2 re-runs SIFT on unedited background only, so the edit can't drag the solve off
  - colour-match statistics are sampled from that same background, so the edit isn't pulled
    back toward the original's colours
"""

import math

import numpy as np
import torch

from comfy_api.latest import io


# ---------------------------------------------------------------------------
# sRGB <-> CIELAB (D65). True L in 0-100 and signed a/b, which the Reinhard maths assumes.
# Inputs must be 3-channel and within [0, 1]: both np.where branches are evaluated, so a negative
# value warns from the fractional powers even though that branch is discarded.
# ---------------------------------------------------------------------------

_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)

_XYZ_TO_RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
], dtype=np.float32)

_D65 = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
_DELTA = 6.0 / 29.0


def _rgb_to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = np.dot(lin, _RGB_TO_XYZ.T) / _D65

    f = np.where(xyz > _DELTA ** 3, xyz ** (1 / 3), xyz / (3 * _DELTA ** 2) + 4 / 29)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]

    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1).astype(np.float32)


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0

    f = np.stack([fx, fy, fz], axis=-1)
    xyz = np.where(f > _DELTA, f ** 3, 3.0 * _DELTA ** 2 * (f - 4.0 / 29.0)) * _D65

    # Clip negative linear values so the fractional power below can't produce NaN
    lin = np.clip(np.dot(xyz, _XYZ_TO_RGB.T), 0.0, None)
    rgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1.0 / 2.4) - 0.055)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour match
# ---------------------------------------------------------------------------

def _apply_color_match(orig_rgb, gen_rgb, edit_mask, valid_mask, strength):
    """Reinhard transfer in LAB, fitted on background pixels, applied to the whole image.

    Returns (image, applied). Skips when strength is 0 or there are fewer than 100 background
    pixels, since a transform fitted to that few is noise.
    """
    if strength <= 0.0:
        return gen_rgb, False

    bg = (edit_mask < 0.05) & (valid_mask > 0.5)
    if bg.sum() < 100:
        return gen_rgb, False

    orig_lab = _rgb_to_lab(orig_rgb)
    gen_lab = _rgb_to_lab(gen_rgb)

    orig_mean, orig_std = orig_lab[bg].mean(axis=0), orig_lab[bg].std(axis=0) + 1e-5
    gen_mean, gen_std = gen_lab[bg].mean(axis=0), gen_lab[bg].std(axis=0) + 1e-5

    matched = _lab_to_rgb((gen_lab - gen_mean) / gen_std * orig_std + orig_mean)
    return np.clip(matched * strength + gen_rgb * (1.0 - strength), 0.0, 1.0), True


# ---------------------------------------------------------------------------
# Coarse edit detection (only used to pick background for pass 2 and colour stats)
# ---------------------------------------------------------------------------

def _blur_kernel(H, W):
    k = max(3, int(round(math.sqrt(H * H + W * W) / 724.0 * 3)))
    if k % 2 == 0:
        k += 1
    return (k, k)


def _compute_diff_map(cv2, orig_np, gen_np, blur_kernel):
    """Delta E (lightness down-weighted) plus a gradient-magnitude term for shape changes."""
    o_blur = cv2.GaussianBlur(orig_np, blur_kernel, 0)
    g_blur = cv2.GaussianBlur(gen_np, blur_kernel, 0)

    diff_lab = _rgb_to_lab(o_blur) - _rgb_to_lab(g_blur)
    diff_lab *= np.array([0.5, 1.2, 1.2], dtype=np.float32)  # ignore most global lighting shift
    color_diff = np.sqrt(np.sum(diff_lab ** 2, axis=-1))

    def grad_mag(img):
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return np.sqrt(gx ** 2 + gy ** 2)

    # Scaled to put structural change on roughly the same scale as Delta E
    struct_diff = np.abs(grad_mag(o_blur) - grad_mag(g_blur)) * 40.0

    return color_diff + struct_diff


def _auto_threshold_mad(diff_map, valid_mask, k=6.0, min_t=3.0, max_t=60.0):
    """median + k * MAD over valid pixels, clamped to [min_t, max_t]."""
    sample = diff_map[valid_mask > 0.5] if valid_mask.sum() > 100 else diff_map.ravel()

    if len(sample) > 50000:
        # Fixed seed so the same inputs always give the same mask (and the same output)
        sample = np.random.default_rng(0).choice(sample, 50000, replace=False)

    med = float(np.median(sample))
    mad = max(float(np.median(np.abs(sample - med))), 0.5)  # noise floor

    return float(np.clip(med + k * mad, min_t, max_t))


# ---------------------------------------------------------------------------
# SIFT homography
# ---------------------------------------------------------------------------

def _sift_prealign(cv2, gray_orig, gray_gen, gen_rgb, orig_mask=None):
    """Warp gen_rgb onto the original's frame with one SIFT + MAGSAC homography.

    orig_mask limits which original keypoints are used (pass 2: background only).

    Returns (aligned, homography, inliers, determinant, valid_mask). On any failure the
    homography is None and gen_rgb comes back unchanged with an all-ones valid mask.
    """
    H, W = gray_orig.shape[:2]
    fallback = (gen_rgb, None, 0, None, np.ones((H, W), dtype=np.float32))

    # CLAHE keeps SIFT finding features after the model has changed global lighting or style
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq_orig = clahe.apply(gray_orig)
    eq_gen = clahe.apply(gray_gen)

    # Deliberately greedy detection and a loose ratio test; MAGSAC rejects the outliers
    sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.03)
    kp1, des1 = sift.detectAndCompute(eq_orig, mask=orig_mask)
    kp2, des2 = sift.detectAndCompute(eq_gen, mask=None)

    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        return fallback

    # FLANN's randomised trees and MAGSAC both draw from OpenCV's RNG; without a fixed seed the
    # same inputs give a slightly different homography on every run
    cv2.setRNGSeed(0)

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=80))
    good = [
        pair[0] for pair in flann.knnMatch(des1, des2, k=2)
        if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
    ]
    if len(good) < 10:
        return fallback

    pts_orig = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_gen = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Solve generated -> original, so the original is never moved
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H_mat, inlier_mask = cv2.findHomography(pts_gen, pts_orig, method, 4.0)
    if H_mat is None:
        return fallback

    # Reject a valid but absurd solve (extreme scale or a flip) from a handful of bad matches
    det = float(H_mat[0, 0] * H_mat[1, 1] - H_mat[0, 1] * H_mat[1, 0])
    if not (0.2 < det < 5.0):
        return fallback

    aligned = cv2.warpPerspective(
        gen_rgb, H_mat, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
    # Zero where the warp pulled from outside the frame (filled by reflection in `aligned`)
    valid = cv2.warpPerspective(
        np.ones((H, W), dtype=np.float32), H_mat, (W, H),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    return aligned, H_mat, int(inlier_mask.sum()), det, valid


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _align_and_match(cv2, orig_np, gen_np, color_match_strength, two_pass, edit_mask=None):
    H, W = orig_np.shape[:2]
    gray_orig = cv2.cvtColor((orig_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_gen = cv2.cvtColor((gen_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Pass 1: blind, keypoints from the whole frame (edit included)
    aligned, H_1, inliers_1, det_1, valid = _sift_prealign(cv2, gray_orig, gray_gen, gen_np)
    det = det_1
    locked = H_1 is not None

    # Edit region: supplied, or thresholded from the pass-1 residual
    if edit_mask is not None:
        coarse = np.clip(edit_mask, 0.0, 1.0) * valid
    else:
        diff = _compute_diff_map(cv2, orig_np, np.clip(aligned, 0, 1), _blur_kernel(H, W))
        coarse = (diff > _auto_threshold_mad(diff, valid)).astype(np.float32) * valid

    # Pass 2: keypoints from unedited background only, eroded inward for safety
    pass2 = "off"
    if two_pass:
        bg = cv2.erode(
            (coarse < 0.1).astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
        )
        if (bg > 0).sum() <= H * W * 0.05:
            pass2 = "skipped (background under 5% of frame)"
        else:
            aligned_2, H_2, inliers_2, det_2, valid_2 = _sift_prealign(
                cv2, gray_orig, gray_gen, gen_np, orig_mask=bg
            )
            if H_2 is None:
                pass2 = "failed, using pass 1"
            else:
                aligned, valid, det = aligned_2, valid_2, det_2
                locked = True
                pass2 = f"{inliers_2} inliers"

    # Align first, then colour match: the statistics compare pixels at the same positions
    aligned, matched = _apply_color_match(orig_np, aligned, coarse, valid, color_match_strength)

    alignment = f"determinant {det:.4f}" if locked else "FAILED - image returned unaligned"

    report = "\n".join([
        f"Alignment:      {alignment}",
        f"Pass 1:         {f'{inliers_1} inliers' if H_1 is not None else 'failed'}",
        f"Pass 2:         {pass2}",
        f"Valid coverage: {100 * valid.mean():.1f}%",
        f"Edit region:    {100 * (coarse > 0.5).mean():.1f}% of frame"
        + (" (from edit_mask)" if edit_mask is not None else ""),
        f"Color match:    {'applied' if matched else 'skipped'}",
    ])

    return np.clip(aligned, 0, 1).astype(np.float32), valid.astype(np.float32), report


class Y7Nodes_KleinQwenEditAlign(io.ComfyNode):
    """Align an edited image back onto its original and match its colour."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Y7Nodes_KleinQwenEditAlign",
            display_name="Y7 Klein & Qwen Edit Align",
            category="Y7Nodes/Image",
            description=(
                "Snaps an edited image (Flux.2 Klein, Qwen-Image-Edit) back onto the original's "
                "geometry with a SIFT homography, then colour-matches it to the original using "
                "the unedited background."
            ),
            inputs=[
                io.Image.Input(
                    "edited_image",
                    tooltip="The edit model's output, to be aligned and colour-matched.",
                ),
                io.Image.Input(
                    "original_image",
                    tooltip="The image that was edited. Resized to the edited image's size if they differ.",
                ),
                io.Float.Input(
                    "color_match_strength",
                    default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="0 = no colour match, 1 = full Reinhard transfer from the original's background.",
                ),
                io.Boolean.Input(
                    "two_pass",
                    default=True,
                    tooltip="Re-align using only unedited background so the edit can't bias the solve. Slower, more accurate.",
                ),
                io.Mask.Input(
                    "edit_mask",
                    optional=True,
                    tooltip="White = edited region. Overrides the automatic detection for pass 2 and colour statistics.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="aligned_image"),
                io.Mask.Output(
                    display_name="valid_mask",
                    tooltip="0 where the warp pulled from outside the frame (filled by reflection).",
                ),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, edited_image, original_image, color_match_strength, two_pass, edit_mask=None) -> io.NodeOutput:
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV not found. Please install it with: pip install opencv-python-headless"
            )

        batch = edited_image.shape[0]
        for name, t in (("original_image", original_image), ("edit_mask", edit_mask)):
            if t is not None and t.shape[0] not in (1, batch):
                raise ValueError(
                    f"KleinQwenEditAlign: {name} must be a single item or match the edited_image batch ({batch})."
                )

        H, W = edited_image.shape[1:3]
        images, valids, reports = [], [], []

        for i in range(batch):
            # RGB only (drop alpha); contiguous so OpenCV can wrap it
            gen_np = np.ascontiguousarray(edited_image[i, ..., :3].cpu().float().numpy())
            orig_np = np.ascontiguousarray(
                original_image[min(i, original_image.shape[0] - 1), ..., :3].cpu().float().numpy()
            )

            if orig_np.shape[:2] != (H, W):
                # Lanczos rings outside [0, 1]; the clip below brings it back
                orig_np = cv2.resize(orig_np, (W, H), interpolation=cv2.INTER_LANCZOS4)

            gen_np = np.clip(gen_np, 0, 1)
            orig_np = np.clip(orig_np, 0, 1)

            mask_np = None
            if edit_mask is not None:
                mask_np = edit_mask[min(i, edit_mask.shape[0] - 1)].cpu().float().numpy()
                if mask_np.shape != (H, W):
                    mask_np = cv2.resize(mask_np, (W, H), interpolation=cv2.INTER_LINEAR)

            aligned, valid, report = _align_and_match(
                cv2, orig_np, gen_np, color_match_strength, two_pass, mask_np
            )
            images.append(torch.from_numpy(aligned))
            valids.append(torch.from_numpy(valid))
            reports.append(report if batch == 1 else f"[Image {i + 1}]\n{report}")

        return io.NodeOutput(torch.stack(images), torch.stack(valids), "\n\n".join(reports))
