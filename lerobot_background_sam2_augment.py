#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create SAM2-mask based background-augmented LeRobot dataset copies.

The script copies a LeRobot dataset directory, decodes videos under
``videos/**/*.mp4``, uses a first-frame SAM2 prompt to track foreground masks,
applies visual changes only to the background, and re-encodes videos with
LeRobot's ``encode_video_frames`` helper. Non-video LeRobot metadata/action/state
files are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_DATA_ROOT = Path("/mnt/c/05-fuxi/datasets")
LOCAL_DATA_ROOT = Path("/mnt/c/05-fuxi/datasets")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
LEROBOT_MAIN_SOURCE_ROOT = WORKSPACE_ROOT / "lerobot_main" / "src"


@dataclass(frozen=True)
class BackgroundSam2Spec:
    name: str
    blur_kernel: int = 9
    noise_std: float = 10.0
    desaturate: float = 0.35
    brightness: float = -8.0
    contrast: float = 0.95
    texture_strength: float = 10.0
    foreground_dilation: int = 7
    mask_feather: int = 13
    background_color: tuple[int, int, int] | None = None
    background_color_alpha: float = 0.0


@dataclass(frozen=True)
class PromptSpec:
    box: list[float] | None = None
    points: list[list[float]] | None = None
    labels: list[int] | None = None


PRESETS: dict[str, BackgroundSam2Spec] = {
    "sam2_bg_blur_only": BackgroundSam2Spec(
        name="sam2_bg_blur_only",
        blur_kernel=15,
        noise_std=0,
        desaturate=0,
        brightness=0,
        contrast=1.0,
        texture_strength=0,
    ),
    "sam2_bg_desaturate": BackgroundSam2Spec(
        name="sam2_bg_desaturate",
        blur_kernel=5,
        noise_std=0,
        desaturate=0.55,
        brightness=-4,
        contrast=0.92,
        texture_strength=0,
    ),
    "sam2_bg_texture_noise": BackgroundSam2Spec(
        name="sam2_bg_texture_noise",
        blur_kernel=3,
        noise_std=18,
        desaturate=0.15,
        brightness=0,
        contrast=0.98,
        texture_strength=16,
    ),
    "sam2_bg_color_gray": BackgroundSam2Spec(
        name="sam2_bg_color_gray",
        blur_kernel=0,
        noise_std=0,
        desaturate=0,
        brightness=0,
        contrast=1.0,
        texture_strength=0,
        background_color=(128, 128, 128),
        background_color_alpha=0.35,
    ),
    "sam2_bg_color_blue": BackgroundSam2Spec(
        name="sam2_bg_color_blue",
        blur_kernel=0,
        noise_std=0,
        desaturate=0,
        brightness=0,
        contrast=1.0,
        texture_strength=0,
        background_color=(70, 110, 170),
        background_color_alpha=0.35,
    ),
    "sam2_bg_color_green": BackgroundSam2Spec(
        name="sam2_bg_color_green",
        blur_kernel=0,
        noise_std=0,
        desaturate=0,
        brightness=0,
        contrast=1.0,
        texture_strength=0,
        background_color=(70, 145, 95),
        background_color_alpha=0.35,
    ),
    "sam2_bg_mixed": BackgroundSam2Spec(name="sam2_bg_mixed"),
}


def default_data_root() -> Path:
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    return LOCAL_DATA_ROOT


def build_output_dataset_name(
    dataset_name: str, preset_name: str, variant_index: int
) -> str:
    return f"{dataset_name}_sam2bg_{preset_name}_{variant_index:02d}"


def resolve_dataset_paths(
    data_root: Path,
    dataset_name: str,
    preset_name: str,
    variant_index: int,
    output_name: str | None = None,
) -> tuple[Path, Path]:
    root = data_root.expanduser().resolve()
    if Path(dataset_name).is_absolute():
        raise ValueError(
            "dataset_name must be a directory name under data_root, not an absolute path"
        )

    source = (root / dataset_name).resolve()
    if source != root and root not in source.parents:
        raise ValueError("dataset_name must stay under data_root")
    if not source.is_dir():
        raise FileNotFoundError(f"dataset not found: {source}")

    target_name = output_name or build_output_dataset_name(
        dataset_name, preset_name, variant_index
    )
    if Path(target_name).is_absolute():
        raise ValueError(
            "output_name must be a directory name under data_root, not an absolute path"
        )
    target = (root / target_name).resolve()
    if target != root and root not in target.parents:
        raise ValueError("output_name must stay under data_root")
    if target == source:
        raise ValueError("output dataset must not equal source dataset")
    return source, target


def video_files(dataset_dir: Path, include_video: str | None = None) -> list[Path]:
    videos = sorted(dataset_dir.glob("videos/**/*.mp4"))
    if include_video:
        needles = [item.strip() for item in include_video.split(",") if item.strip()]
        videos = [
            video
            for video in videos
            if any(needle in video.as_posix() for needle in needles)
        ]
    return videos


def parse_float_list(value: str, expected: int, name: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values")
    return [float(item) for item in items]


def parse_points(value: str) -> list[list[float]]:
    points = []
    for pair in value.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        points.append(parse_float_list(pair, 2, "point"))
    if not points:
        raise ValueError("--points must contain at least one x,y point")
    return points


def parse_labels(value: str) -> list[int]:
    labels = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not labels:
        raise ValueError("--labels must contain at least one label")
    return labels


def parse_color(value: str) -> tuple[int, int, int]:
    text = value.strip()
    if text.startswith("#"):
        hex_value = text[1:]
        if len(hex_value) != 6:
            raise ValueError("hex color must use #RRGGBB")
        return (
            int(hex_value[0:2], 16),
            int(hex_value[2:4], 16),
            int(hex_value[4:6], 16),
        )
    items = [item.strip() for item in text.split(",") if item.strip()]
    if len(items) != 3:
        raise ValueError("color must be #RRGGBB or r,g,b")
    color = tuple(int(item) for item in items)
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("RGB color channels must be in [0, 255]")
    return color  # type: ignore[return-value]


def override_spec(
    spec: BackgroundSam2Spec,
    foreground_dilation: int | None = None,
    mask_feather: int | None = None,
    background_color: str | None = None,
    background_color_alpha: float | None = None,
    texture_strength: float | None = None,
    blur_kernel: int | None = None,
) -> BackgroundSam2Spec:
    return BackgroundSam2Spec(
        name=spec.name,
        blur_kernel=spec.blur_kernel if blur_kernel is None else max(0, blur_kernel),
        noise_std=spec.noise_std,
        desaturate=spec.desaturate,
        brightness=spec.brightness,
        contrast=spec.contrast,
        texture_strength=(
            spec.texture_strength
            if texture_strength is None
            else max(0.0, texture_strength)
        ),
        foreground_dilation=(
            spec.foreground_dilation
            if foreground_dilation is None
            else max(0, foreground_dilation)
        ),
        mask_feather=spec.mask_feather if mask_feather is None else max(0, mask_feather),
        background_color=(
            spec.background_color if background_color is None else parse_color(background_color)
        ),
        background_color_alpha=(
            spec.background_color_alpha
            if background_color_alpha is None
            else float(np.clip(background_color_alpha, 0, 1))
        ),
    )


def coerce_prompt_spec(payload: dict[str, object]) -> PromptSpec:
    box = payload.get("box")
    points = payload.get("points")
    labels = payload.get("labels")
    if box is not None:
        box = [float(item) for item in box]  # type: ignore[assignment]
        if len(box) != 4:
            raise ValueError("prompt box must be [x1, y1, x2, y2]")
    if points is not None:
        points = [
            [float(point[0]), float(point[1])] for point in points  # type: ignore[index]
        ]
    if labels is not None:
        labels = [int(item) for item in labels]  # type: ignore[assignment]
    if points is not None and labels is None:
        labels = [1 for _ in points]
    if points is not None and len(points) != len(labels or []):
        raise ValueError("prompt points and labels must have the same length")
    if box is None and points is None:
        raise ValueError("prompt must provide box or points")
    return PromptSpec(box=box, points=points, labels=labels)  # type: ignore[arg-type]


def load_prompt_specs(
    prompt_json: Path | None,
    box: str | None,
    points: str | None,
    labels: str | None = None,
) -> dict[str, PromptSpec]:
    prompts: dict[str, PromptSpec] = {}
    if prompt_json:
        payload = json.loads(prompt_json.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--prompt-json must be a JSON object")
        for key, value in payload.items():
            if not isinstance(value, dict):
                raise ValueError(f"prompt entry {key!r} must be an object")
            prompts[str(key)] = coerce_prompt_spec(value)

    inline: dict[str, object] = {}
    if box:
        inline["box"] = parse_float_list(box, 4, "--box")
    if points:
        inline["points"] = parse_points(points)
        inline["labels"] = parse_labels(labels) if labels else [1 for _ in inline["points"]]  # type: ignore[index]
    elif labels:
        raise ValueError("--labels requires --points")
    if inline:
        prompts["default"] = coerce_prompt_spec(inline)
    if not prompts:
        raise ValueError("provide --box, --points, or --prompt-json for SAM2 prompts")
    return prompts


def prompt_for_video(relative_video: Path, prompts: dict[str, PromptSpec]) -> PromptSpec:
    path = relative_video.as_posix()
    matches = [
        (key, prompt)
        for key, prompt in prompts.items()
        if key != "default" and key in path
    ]
    if matches:
        matches.sort(key=lambda item: len(item[0]), reverse=True)
        return matches[0][1]
    if "default" in prompts:
        return prompts["default"]
    raise KeyError(f"no prompt matched video path: {path}")


def normalize_odd_kernel(value: int) -> int:
    if value <= 1:
        return 0
    return value if value % 2 else value + 1


def box_blur_image(img: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = normalize_odd_kernel(kernel_size)
    if not kernel_size:
        return img
    pad = kernel_size // 2
    padded = np.pad(
        img.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)), mode="edge"
    )
    output = np.zeros_like(img, dtype=np.float32)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            output += padded[dy : dy + img.shape[0], dx : dx + img.shape[1], :]
    return np.clip(output / float(kernel_size * kernel_size), 0, 255).astype(np.uint8)


def box_blur_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = normalize_odd_kernel(kernel_size)
    if not kernel_size:
        return mask.astype(np.float32)
    pad = kernel_size // 2
    source = mask.astype(np.float32)
    padded = np.pad(source, ((pad, pad), (pad, pad)), mode="edge")
    output = np.zeros_like(source, dtype=np.float32)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            output += padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
    return np.clip(output / float(kernel_size * kernel_size), 0, 1)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    source = mask.astype(bool)
    padded = np.pad(source, ((radius, radius), (radius, radius)), mode="constant")
    output = np.zeros_like(source, dtype=bool)
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            output |= padded[dy : dy + source.shape[0], dx : dx + source.shape[1]]
    return output


def background_mask_from_foreground(
    foreground_mask: np.ndarray, dilation: int = 7, feather: int = 13
) -> np.ndarray:
    if foreground_mask.ndim != 2:
        raise ValueError("foreground_mask must be a 2D boolean array")
    height, width = foreground_mask.shape
    effective_dilation = min(dilation, max(0, min(height, width) // 8))
    protected = dilate_mask(foreground_mask, effective_dilation)
    background = (~protected).astype(np.float32)
    background = box_blur_mask(background, feather)
    background[foreground_mask.astype(bool)] = 0.0
    return background


def merge_prompt_box_mask(
    foreground_mask: np.ndarray,
    prompt: PromptSpec,
    protect_prompt_box: bool = False,
) -> np.ndarray:
    merged = np.array(foreground_mask, dtype=bool, copy=True)
    if not protect_prompt_box or prompt.box is None:
        return merged
    height, width = merged.shape
    x1, y1, x2, y2 = prompt.box
    left = int(np.clip(np.floor(min(x1, x2)), 0, width))
    right = int(np.clip(np.ceil(max(x1, x2)), 0, width))
    top = int(np.clip(np.floor(min(y1, y2)), 0, height))
    bottom = int(np.clip(np.ceil(max(y1, y2)), 0, height))
    if right > left and bottom > top:
        merged[top:bottom, left:right] = True
    return merged


def desaturate_image(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return img
    amount = float(np.clip(amount, 0, 1))
    gray = np.dot(img[:, :, :3].astype(np.float32), [0.114, 0.587, 0.299])
    mixed = img.astype(np.float32) * (1.0 - amount) + gray[:, :, None] * amount
    return np.clip(mixed, 0, 255).astype(np.uint8)


def add_texture_noise(
    img: np.ndarray, rng: np.random.Generator, strength: float
) -> np.ndarray:
    if strength <= 0:
        return img
    height, width = img.shape[:2]
    coarse_h = max(2, height // 24)
    coarse_w = max(2, width // 24)
    coarse = rng.normal(0, strength, (coarse_h, coarse_w, 1)).astype(np.float32)
    repeat_y = int(np.ceil(height / coarse_h))
    repeat_x = int(np.ceil(width / coarse_w))
    texture = np.kron(coarse, np.ones((repeat_y, repeat_x, 1), dtype=np.float32))
    texture = texture[:height, :width, :]
    return np.clip(img.astype(np.float32) + texture, 0, 255).astype(np.uint8)


def augment_background_image(
    frame: np.ndarray, spec: BackgroundSam2Spec, rng: np.random.Generator
) -> np.ndarray:
    bg = np.array(frame, dtype=np.uint8, copy=True)
    if spec.blur_kernel:
        bg = box_blur_image(bg, spec.blur_kernel)
    bg = bg.astype(np.float32) * spec.contrast + spec.brightness
    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bg = desaturate_image(bg, spec.desaturate)
    if spec.noise_std > 0:
        bg = np.clip(
            bg.astype(np.float32) + rng.normal(0, spec.noise_std, bg.shape),
            0,
            255,
        ).astype(np.uint8)
    bg = add_texture_noise(bg, rng, spec.texture_strength)
    if spec.background_color is not None and spec.background_color_alpha > 0:
        color = np.array(spec.background_color, dtype=np.float32)
        alpha = float(np.clip(spec.background_color_alpha, 0, 1))
        bg = np.clip(bg.astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    return bg


def apply_background_augmentation(
    frame: np.ndarray,
    foreground_mask: np.ndarray,
    spec: BackgroundSam2Spec,
    rng: np.random.Generator,
    protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    img = np.asarray(frame)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = np.array(img, dtype=np.uint8, copy=True)
    height, width = img.shape[:2]
    if foreground_mask.shape != (height, width):
        raise ValueError(
            f"foreground_mask shape must be {(height, width)}, got {foreground_mask.shape}"
        )
    if protect_mask is not None:
        if protect_mask.shape != (height, width):
            raise ValueError(
                f"protect_mask shape must be {(height, width)}, got {protect_mask.shape}"
            )
        foreground_mask = np.asarray(foreground_mask, dtype=bool) | np.asarray(
            protect_mask, dtype=bool
        )

    background_alpha = background_mask_from_foreground(
        foreground_mask,
        dilation=spec.foreground_dilation,
        feather=spec.mask_feather,
    )[:, :, None]
    background = augment_background_image(img, spec, rng).astype(np.float32)
    composed = img.astype(np.float32) * (1.0 - background_alpha) + background * background_alpha
    return np.clip(composed, 0, 255).astype(np.uint8)


def variant_spec(
    base: BackgroundSam2Spec, variant_index: int, rng: np.random.Generator
) -> BackgroundSam2Spec:
    return BackgroundSam2Spec(
        name=base.name,
        blur_kernel=max(0, int(round(base.blur_kernel + rng.normal(0, 2)))),
        noise_std=max(0.0, base.noise_std + float(rng.normal(0, 2))),
        desaturate=float(np.clip(base.desaturate + rng.normal(0, 0.06), 0, 0.8)),
        brightness=base.brightness + float(rng.normal(0, 5)),
        contrast=max(0.65, base.contrast + float(rng.normal(0, 0.04))),
        texture_strength=max(0.0, base.texture_strength + float(rng.normal(0, 2))),
        foreground_dilation=max(0, int(round(base.foreground_dilation + rng.normal(0, 1)))),
        mask_feather=max(0, int(round(base.mask_feather + rng.normal(0, 2)))),
        background_color=base.background_color,
        background_color_alpha=base.background_color_alpha,
    )


def parse_frame_indices(value: str | None) -> set[int]:
    if not value:
        return {0}
    indices: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.add(int(item))
    return indices or {0}


def safe_debug_video_dir(relative_video: Path) -> Path:
    parts = list(relative_video.parts)
    if parts and parts[0] == "videos":
        parts = parts[1:]
    if parts and parts[-1].endswith(".mp4"):
        parts[-1] = Path(parts[-1]).stem
    return Path(*parts)


def save_debug_frames(
    debug_dir: Path,
    frame_index: int,
    frame: np.ndarray,
    foreground_mask: np.ndarray,
    background_alpha: np.ndarray,
) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    frame = np.asarray(frame, dtype=np.uint8)
    mask_img = np.asarray(foreground_mask, dtype=np.uint8) * 255
    alpha_img = np.clip(background_alpha.astype(np.float32) * 255, 0, 255).astype(np.uint8)
    overlay = frame.astype(np.float32).copy()
    red = np.array([255, 30, 30], dtype=np.float32)
    overlay[foreground_mask.astype(bool)] = (
        overlay[foreground_mask.astype(bool)] * 0.55 + red * 0.45
    )
    write_debug_image(debug_dir / f"frame_{frame_index:06d}.jpg", frame)
    write_debug_image(debug_dir / f"mask_{frame_index:06d}.png", mask_img)
    write_debug_image(
        debug_dir / f"overlay_{frame_index:06d}.jpg",
        np.clip(overlay, 0, 255).astype(np.uint8),
    )
    write_debug_image(debug_dir / f"background_alpha_{frame_index:06d}.png", alpha_img)


def write_debug_image(path: Path, image: np.ndarray) -> None:
    try:
        import imageio.v2 as imageio

        imageio.imwrite(path, image)
        return
    except ImportError:
        pass

    try:
        import cv2

        output = image
        if output.ndim == 3 and output.shape[2] == 3:
            output = output[:, :, ::-1]
        cv2.imwrite(str(path), output)
        return
    except ImportError:
        pass

    # Last-resort fallback for dependency-light tests. The extension may not
    # match the PNM payload, but the debug artifact still preserves pixels.
    if image.ndim == 2:
        header = f"P5\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
        path.write_bytes(header + image.astype(np.uint8).tobytes())
    else:
        header = f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii")
        path.write_bytes(header + image[:, :, :3].astype(np.uint8).tobytes())


def mask_logits_to_numpy(mask_logits, height: int, width: int) -> np.ndarray:
    values = mask_logits.detach().cpu().numpy()
    values = np.asarray(values)
    values = np.squeeze(values)
    if values.ndim == 3:
        values = np.max(values, axis=0)
    if values.ndim != 2:
        raise ValueError(f"expected 2D SAM2 mask logits after squeeze, got {values.shape}")
    mask = values > 0
    if mask.shape == (height, width):
        return mask

    source_h, source_w = mask.shape
    y_idx = np.minimum(
        (np.arange(height) * source_h / height).astype(int), source_h - 1
    )
    x_idx = np.minimum(
        (np.arange(width) * source_w / width).astype(int), source_w - 1
    )
    return mask[y_idx[:, None], x_idx[None, :]]


def run_sam2_on_frames(
    predictor,
    frames_dir: Path,
    prompt: PromptSpec,
    height: int,
    width: int,
    obj_id: int = 1,
) -> dict[int, np.ndarray]:
    inference_state = predictor.init_state(str(frames_dir))
    if hasattr(predictor, "reset_state"):
        predictor.reset_state(inference_state)

    box_array = np.array(prompt.box, dtype=np.float32) if prompt.box is not None else None
    points_array = (
        np.array(prompt.points, dtype=np.float32) if prompt.points is not None else None
    )
    labels_array = (
        np.array(prompt.labels, dtype=np.int32) if prompt.labels is not None else None
    )
    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        points=points_array,
        labels=labels_array,
        box=box_array,
    )

    masks: dict[int, np.ndarray] = {}
    for frame_idx, _object_ids, mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        masks[int(frame_idx)] = mask_logits_to_numpy(mask_logits, height, width)
    return masks


def copy_dataset_tree(source: Path, target: Path, overwrite: bool = False) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"output dataset already exists: {target}")
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("*.mp4"))


def ensure_lerobot_import_path() -> None:
    package_dir = LEROBOT_MAIN_SOURCE_ROOT / "lerobot"
    if package_dir.is_dir():
        source_root = str(LEROBOT_MAIN_SOURCE_ROOT)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)


def import_encode_video_frames():
    from lerobot.datasets.video_utils import encode_video_frames

    return encode_video_frames


def is_depth_video(video_path: Path) -> bool:
    return any("depth" in part for part in video_path.parts)


def remove_partial_video(target_video: Path) -> None:
    try:
        target_video.unlink()
    except FileNotFoundError:
        pass


def encode_augmented_frames(frames_dir: Path, target_video: Path, fps: int) -> None:
    ensure_lerobot_import_path()
    encode_video_frames = import_encode_video_frames()
    target_video.parent.mkdir(parents=True, exist_ok=True)
    try:
        if is_depth_video(target_video):
            encode_video_frames(
                frames_dir,
                target_video,
                fps,
                vcodec="h264",
                pix_fmt="yuv444p",
                crf=20,
                overwrite=True,
            )
        else:
            encode_video_frames(frames_dir, target_video, fps, overwrite=True)
    except Exception:
        remove_partial_video(target_video)
        raise


def ensure_sam2_import_path(sam2_root: Path | None) -> None:
    if sam2_root:
        root = str(sam2_root.expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def load_sam2_predictor(
    sam2_root: Path | None, checkpoint: Path, model_cfg: str, device: str
):
    ensure_sam2_import_path(sam2_root)
    if os.environ.get("OMP_NUM_THREADS") in {"", "0"}:
        os.environ["OMP_NUM_THREADS"] = "8"
    from sam2.build_sam import build_sam2_video_predictor

    cwd = Path.cwd()
    try:
        if sam2_root:
            os.chdir(sam2_root.expanduser().resolve())
        return build_sam2_video_predictor(model_cfg, str(checkpoint), device=device)
    finally:
        os.chdir(cwd)


def decode_video_to_sam2_frames(
    source_video: Path, sam2_frames_dir: Path
) -> dict[str, object]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(source_video))
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps") or 30)
    source_frame_count = metadata.get("nframes")
    if not isinstance(source_frame_count, int) or source_frame_count <= 0:
        source_frame_count = None

    sam2_frames_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    width = 0
    height = 0
    try:
        for frame in reader:
            frame = np.asarray(frame)
            if frame.ndim == 2:
                frame = np.repeat(frame[:, :, None], 3, axis=2)
            if frame.shape[2] > 3:
                frame = frame[:, :, :3]
            frame = np.array(frame, dtype=np.uint8, copy=False)
            height, width = frame.shape[:2]
            imageio.imwrite(sam2_frames_dir / f"{written:06d}.jpg", frame)
            written += 1
            if written == 1 or written % 100 == 0:
                progress = f"{written}/{source_frame_count}" if source_frame_count else str(written)
                print(f"    decoded: {progress}", end="\r", flush=True)
    finally:
        reader.close()
    if written == 0:
        raise RuntimeError(f"no frames were decoded from video: {source_video}")
    print(f"    decoded: {written}/{source_frame_count}" if source_frame_count else f"    decoded: {written}")
    return {
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": source_frame_count,
        "decoded_frame_count": written,
        "reader": "imageio",
    }


def write_augmented_frames(
    sam2_frames_dir: Path,
    output_frames_dir: Path,
    masks: dict[int, np.ndarray],
    prompt: PromptSpec,
    spec: BackgroundSam2Spec,
    seed: int,
    protect_prompt_box: bool = False,
    debug_dir: Path | None = None,
    debug_frame_indices: set[int] | None = None,
) -> int:
    import imageio.v2 as imageio

    output_frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frame_paths = sorted(sam2_frames_dir.glob("*.jpg"))
    written = 0
    fallback_count = 0
    empty_mask = None
    for index, frame_path in enumerate(frame_paths):
        frame = np.asarray(imageio.imread(frame_path))
        if empty_mask is None:
            empty_mask = np.zeros(frame.shape[:2], dtype=bool)
        foreground = masks.get(index)
        if foreground is None:
            foreground = empty_mask
            fallback_count += 1
        foreground = merge_prompt_box_mask(
            foreground,
            prompt,
            protect_prompt_box=protect_prompt_box,
        )
        background_alpha = background_mask_from_foreground(
            foreground,
            dilation=spec.foreground_dilation,
            feather=spec.mask_feather,
        )
        if (
            debug_dir is not None
            and debug_frame_indices is not None
            and index in debug_frame_indices
        ):
            save_debug_frames(
                debug_dir,
                frame_index=index,
                frame=frame,
                foreground_mask=foreground,
                background_alpha=background_alpha,
            )
        augmented = apply_background_augmentation(frame, foreground, spec, rng)
        imageio.imwrite(output_frames_dir / f"frame-{index:06d}.png", augmented)
        written += 1
        if written == 1 or written % 100 == 0:
            print(f"    augmented: {written}/{len(frame_paths)}", end="\r", flush=True)
    print(f"    augmented: {written}/{len(frame_paths)}")
    if fallback_count:
        print(f"    warning: {fallback_count} frames had no SAM2 mask; treated as background")
    return written


def augment_video(
    source_video: Path,
    target_video: Path,
    predictor,
    prompt: PromptSpec,
    spec: BackgroundSam2Spec,
    seed: int,
    protect_prompt_box: bool = False,
    debug_dir: Path | None = None,
    debug_frame_indices: set[int] | None = None,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="lerobot_sam2_frames_") as sam2_tmpdir:
        with tempfile.TemporaryDirectory(prefix="lerobot_sam2_aug_frames_") as out_tmpdir:
            sam2_frames_dir = Path(sam2_tmpdir)
            output_frames_dir = Path(out_tmpdir)
            report = decode_video_to_sam2_frames(source_video, sam2_frames_dir)
            masks = run_sam2_on_frames(
                predictor,
                sam2_frames_dir,
                prompt,
                height=int(report["height"]),
                width=int(report["width"]),
            )
            written = write_augmented_frames(
                sam2_frames_dir,
                output_frames_dir,
                masks,
                prompt,
                spec,
                seed,
                protect_prompt_box=protect_prompt_box,
                debug_dir=debug_dir,
                debug_frame_indices=debug_frame_indices,
            )
            encode_augmented_frames(
                output_frames_dir, target_video, int(round(report["fps"]))
            )
            report["target"] = str(target_video)
            report["written_frame_count"] = written
            report["sam2_mask_count"] = len(masks)
            report["writer"] = "lerobot.encode_video_frames"
            return report


def augment_dataset(
    data_root: Path,
    dataset_name: str,
    preset_name: str,
    prompts: dict[str, PromptSpec],
    sam2_root: Path | None,
    checkpoint: Path,
    model_cfg: str,
    device: str = "cuda",
    variant_index: int = 1,
    output_name: str | None = None,
    overwrite: bool = False,
    seed: int = 0,
    include_video: str | None = None,
    foreground_dilation: int | None = None,
    mask_feather: int | None = None,
    background_color: str | None = None,
    background_color_alpha: float | None = None,
    texture_strength: float | None = None,
    blur_kernel: int | None = None,
    protect_prompt_box: bool = False,
    save_debug_frames_enabled: bool = False,
    debug_frame_indices: set[int] | None = None,
) -> dict[str, object]:
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; choose one of: {', '.join(PRESETS)}"
        )
    source, target = resolve_dataset_paths(
        data_root, dataset_name, preset_name, variant_index, output_name
    )
    source_videos = video_files(source, include_video=include_video)
    if not source_videos:
        raise FileNotFoundError(f"no videos found under: {source / 'videos'}")

    rng = np.random.default_rng(seed + variant_index)
    spec = variant_spec(PRESETS[preset_name], variant_index, rng)
    spec = override_spec(
        spec,
        foreground_dilation=foreground_dilation,
        mask_feather=mask_feather,
        background_color=background_color,
        background_color_alpha=background_color_alpha,
        texture_strength=texture_strength,
        blur_kernel=blur_kernel,
    )
    copy_dataset_tree(source, target, overwrite=overwrite)
    predictor = load_sam2_predictor(sam2_root, checkpoint, model_cfg, device)
    if save_debug_frames_enabled and debug_frame_indices is None:
        debug_frame_indices = {0}

    video_reports = []
    for offset, source_video in enumerate(source_videos):
        relative = source_video.relative_to(source)
        target_video = target / relative
        prompt = prompt_for_video(relative, prompts)
        debug_dir = None
        if save_debug_frames_enabled:
            debug_dir = target / "sam2_debug" / safe_debug_video_dir(relative)
        print(f"  video [{offset + 1}/{len(source_videos)}] {relative}", flush=True)
        report = augment_video(
            source_video,
            target_video,
            predictor,
            prompt,
            spec,
            seed=seed + variant_index * 1000 + offset,
            protect_prompt_box=protect_prompt_box,
            debug_dir=debug_dir,
            debug_frame_indices=debug_frame_indices,
        )
        report["prompt"] = asdict(prompt)
        print(f"    done: {report['written_frame_count']} frames", flush=True)
        video_reports.append(report)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "output_dataset": str(target),
        "preset": preset_name,
        "variant_index": variant_index,
        "spec": asdict(spec),
        "checkpoint": str(checkpoint),
        "model_cfg": model_cfg,
        "device": device,
        "protect_prompt_box": protect_prompt_box,
        "save_debug_frames": save_debug_frames_enabled,
        "debug_frame_indices": sorted(debug_frame_indices or []),
        "videos": video_reports,
        "note": "Only videos/**/*.mp4 were SAM2-background-augmented and re-encoded. Non-video LeRobot metadata/action/state files were copied unchanged.",
    }
    (target / "background_sam2_augmentation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def choose_presets(requested: str | Iterable[str], multiplier: int) -> list[str]:
    if isinstance(requested, str):
        if requested == "all":
            names = list(PRESETS)
        else:
            names = [item.strip() for item in requested.split(",") if item.strip()]
    else:
        names = list(requested)
    if not names:
        raise ValueError("at least one preset is required")
    return [names[i % len(names)] for i in range(multiplier)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create SAM2 foreground-mask based background-augmented copies of a LeRobot dataset."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--dataset", required=True, help="Dataset directory name under data root."
    )
    parser.add_argument(
        "--preset",
        default="sam2_bg_mixed",
        help="Preset name, comma-list, or 'all'.",
    )
    parser.add_argument("--multiplier", type=int, default=1)
    parser.add_argument("--output-name", default=None, help="Only valid when multiplier=1.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument(
        "--include-video",
        default=None,
        help="Optional comma-list of path substrings. Use this to test one camera first.",
    )
    parser.add_argument("--sam2-root", type=Path, default=Path("/root/sam2"))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/root/sam2/checkpoints/sam2.1_hiera_large.pt"),
    )
    parser.add_argument(
        "--model-cfg",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prompt-json",
        type=Path,
        default=None,
        help="JSON mapping path substrings or 'default' to {'box': [...]} or {'points': [...], 'labels': [...]}.",
    )
    parser.add_argument(
        "--box",
        default=None,
        help="Default first-frame foreground box: x1,y1,x2,y2.",
    )
    parser.add_argument(
        "--points",
        default=None,
        help="Default first-frame foreground points: x,y;x,y.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Labels for --points, comma-list where 1=foreground and 0=background.",
    )
    parser.add_argument(
        "--foreground-dilation",
        type=int,
        default=None,
        help="Override foreground mask dilation pixels.",
    )
    parser.add_argument(
        "--mask-feather",
        type=int,
        default=None,
        help="Override foreground/background transition feather pixels.",
    )
    parser.add_argument(
        "--protect-prompt-box",
        action="store_true",
        help="Force prompt box pixels to be protected even if SAM2 misses them.",
    )
    parser.add_argument(
        "--background-color",
        default=None,
        help="Blend background toward #RRGGBB or r,g,b.",
    )
    parser.add_argument(
        "--background-color-alpha",
        type=float,
        default=None,
        help="Background color blend strength in [0, 1].",
    )
    parser.add_argument(
        "--texture-strength",
        type=float,
        default=None,
        help="Override background low-frequency texture noise strength.",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=None,
        help="Override background blur kernel. 0 disables blur.",
    )
    parser.add_argument(
        "--save-debug-frames",
        action="store_true",
        help="Save original/mask/overlay/background-alpha images for selected frames.",
    )
    parser.add_argument(
        "--debug-frame-indices",
        default="0",
        help="Comma-list of frame indices to save when --save-debug-frames is used.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.multiplier < 1:
        raise SystemExit("--multiplier must be >= 1")
    if args.output_name and args.multiplier != 1:
        raise SystemExit("--output-name can only be used when --multiplier=1")
    prompts = load_prompt_specs(args.prompt_json, args.box, args.points, args.labels)
    debug_frame_indices = parse_frame_indices(args.debug_frame_indices)

    reports = []
    presets = choose_presets(args.preset, args.multiplier)
    for index, preset_name in enumerate(presets, start=1):
        print(f"[{index}/{args.multiplier}] creating preset={preset_name}")
        reports.append(
            augment_dataset(
                data_root=args.data_root,
                dataset_name=args.dataset,
                preset_name=preset_name,
                prompts=prompts,
                sam2_root=args.sam2_root,
                checkpoint=args.checkpoint,
                model_cfg=args.model_cfg,
                device=args.device,
                variant_index=index,
                output_name=args.output_name,
                overwrite=args.overwrite,
                seed=args.seed,
                include_video=args.include_video,
                foreground_dilation=args.foreground_dilation,
                mask_feather=args.mask_feather,
                background_color=args.background_color,
                background_color_alpha=args.background_color_alpha,
                texture_strength=args.texture_strength,
                blur_kernel=args.blur_kernel,
                protect_prompt_box=args.protect_prompt_box,
                save_debug_frames_enabled=args.save_debug_frames,
                debug_frame_indices=debug_frame_indices,
            )
        )
    print(
        json.dumps(
            {"created": [report["output_dataset"] for report in reports]},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
