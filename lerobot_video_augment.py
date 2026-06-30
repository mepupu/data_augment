#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create visually augmented LeRobot dataset copies.

The script copies a LeRobot dataset directory and re-encodes videos under
``videos/**/*.mp4`` with deterministic visual augmentations. Non-video files,
including actions, states, timestamps, tasks, and metadata, are copied unchanged.
"""

from __future__ import annotations

import argparse
import json
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
class AugmentSpec:
    name: str
    brightness: float = 0.0
    contrast: float = 1.0
    saturation: float = 1.0
    temperature: float = 0.0
    noise_std: float = 0.0
    blur_kernel: int = 0
    gamma: float = 1.0
    rain: bool = False


PRESETS: dict[str, AugmentSpec] = {
    "day": AugmentSpec(
        name="day", brightness=18, contrast=1.08, saturation=1.06, temperature=8
    ),
    "night": AugmentSpec(
        name="night",
        brightness=-48,
        contrast=0.9,
        saturation=0.78,
        temperature=-18,
        noise_std=4,
    ),
    "cloudy": AugmentSpec(
        name="cloudy", brightness=-10, contrast=0.86, saturation=0.82, temperature=-8
    ),
    "warm_indoor": AugmentSpec(
        name="warm_indoor", brightness=8, contrast=1.02, saturation=1.08, temperature=24
    ),
    "cold_indoor": AugmentSpec(
        name="cold_indoor",
        brightness=2,
        contrast=1.05,
        saturation=0.96,
        temperature=-22,
    ),
    "low_light_noise": AugmentSpec(
        name="low_light_noise",
        brightness=-36,
        contrast=0.95,
        saturation=0.82,
        noise_std=8,
    ),
    "motion_blur": AugmentSpec(
        name="motion_blur", brightness=0, contrast=1.0, saturation=1.0, blur_kernel=7
    ),
    "rainy": AugmentSpec(
        name="rainy",
        brightness=-20,
        contrast=0.88,
        saturation=0.72,
        temperature=-10,
        rain=True,
    ),
}


def default_data_root() -> Path:
    if DEFAULT_DATA_ROOT.exists():
        return DEFAULT_DATA_ROOT
    return LOCAL_DATA_ROOT


def build_output_dataset_name(
    dataset_name: str, preset_name: str, variant_index: int
) -> str:
    return f"{dataset_name}_aug_{preset_name}_{variant_index:02d}"


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


def video_files(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.glob("videos/**/*.mp4"))


def variant_spec(
    base: AugmentSpec, variant_index: int, rng: np.random.Generator
) -> AugmentSpec:
    """Create a gentle deterministic variation around a preset."""
    return AugmentSpec(
        name=base.name,
        brightness=base.brightness + float(rng.normal(0, 5)),
        contrast=max(0.65, base.contrast + float(rng.normal(0, 0.04))),
        saturation=max(0.45, base.saturation + float(rng.normal(0, 0.05))),
        temperature=base.temperature + float(rng.normal(0, 5)),
        noise_std=max(0.0, base.noise_std + float(rng.normal(0, 1.5))),
        blur_kernel=base.blur_kernel if variant_index % 2 else 0,
        gamma=max(0.65, base.gamma + float(rng.normal(0, 0.04))),
        rain=base.rain,
    )


def apply_image_augmentation(
    frame: np.ndarray, spec: AugmentSpec, rng: np.random.Generator
) -> np.ndarray:
    img = np.array(frame, dtype=np.float32, copy=True)

    if spec.gamma != 1.0:
        normalized = np.clip(img / 255.0, 0, 1)
        img = np.power(normalized, spec.gamma) * 255.0

    img = img * spec.contrast + spec.brightness

    if spec.temperature:
        # OpenCV frames are BGR. Positive temperature warms the image.
        img[:, :, 2] += spec.temperature
        img[:, :, 0] -= spec.temperature

    img = np.clip(img, 0, 255).astype(np.uint8)

    if spec.saturation != 1.0:
        gray = np.dot(img[:, :, :3].astype(np.float32), [0.114, 0.587, 0.299])
        img = np.clip(
            gray[:, :, None]
            + (img.astype(np.float32) - gray[:, :, None]) * spec.saturation,
            0,
            255,
        ).astype(np.uint8)

    if spec.noise_std > 0:
        noise = rng.normal(0, spec.noise_std, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if spec.blur_kernel and spec.blur_kernel > 1:
        kernel_size = (
            spec.blur_kernel if spec.blur_kernel % 2 == 1 else spec.blur_kernel + 1
        )
        pad = kernel_size // 2
        padded = np.pad(
            img.astype(np.float32), ((0, 0), (pad, pad), (0, 0)), mode="edge"
        )
        accum = np.zeros_like(img, dtype=np.float32)
        for offset in range(kernel_size):
            accum += padded[:, offset : offset + img.shape[1], :]
        img = np.clip(accum / kernel_size, 0, 255).astype(np.uint8)

    if spec.rain:
        img = add_rain_streaks(img, rng)

    return img


def add_rain_streaks(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    overlay = frame.copy()
    height, width = frame.shape[:2]
    count = max(20, (height * width) // 12000)
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        length = int(rng.integers(8, 24))
        for step in range(length):
            yy = min(height - 1, y + step)
            xx = min(width - 1, x + step // 5)
            overlay[yy, xx] = (210, 210, 210)
    return np.clip(
        frame.astype(np.float32) * 0.86 + overlay.astype(np.float32) * 0.14, 0, 255
    ).astype(np.uint8)


def copy_dataset_tree(source: Path, target: Path, overwrite: bool = False) -> None:
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"output dataset already exists: {target}")
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("*.mp4")
    shutil.copytree(source, target, ignore=ignore)


def augment_video(
    source_video: Path, target_video: Path, spec: AugmentSpec, seed: int
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="lerobot_aug_frames_") as tmpdir:
        frames_dir = Path(tmpdir)
        try:
            report = augment_video_frames_with_imageio(
                source_video, frames_dir, spec, seed
            )
        except ImportError:
            report = augment_video_frames_with_cv2(source_video, frames_dir, spec, seed)

        encode_augmented_frames(frames_dir, target_video, int(round(report["fps"])))
        report["target"] = str(target_video)
        report["writer"] = "lerobot.encode_video_frames"
        return report


def remove_partial_video(target_video: Path) -> None:
    try:
        target_video.unlink()
    except FileNotFoundError:
        pass


def ensure_lerobot_import_path() -> None:
    package_dir = LEROBOT_MAIN_SOURCE_ROOT / "lerobot"
    if package_dir.is_dir():
        source_root = str(LEROBOT_MAIN_SOURCE_ROOT)
        if source_root not in sys.path:
            sys.path.insert(0, source_root)


def is_depth_video(video_path: Path) -> bool:
    return any("depth" in part for part in video_path.parts)


def import_encode_video_frames():
    from lerobot.datasets.video_utils import encode_video_frames

    return encode_video_frames


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


def augment_video_frames_with_imageio(
    source_video: Path, frames_dir: Path, spec: AugmentSpec, seed: int
) -> dict[str, object]:
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(source_video))
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps") or 30)
    source_frame_count = metadata.get("nframes")
    if not isinstance(source_frame_count, int) or source_frame_count <= 0:
        source_frame_count = None

    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
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
            height, width = frame.shape[:2]

            # imageio returns RGB frames; the augmentation function expects BGR.
            bgr = np.ascontiguousarray(frame[:, :, ::-1])
            augmented = np.ascontiguousarray(
                apply_image_augmentation(bgr, spec, rng)[:, :, ::-1]
            )
            imageio.imwrite(frames_dir / f"frame-{written:06d}.png", augmented)
            written += 1
            if written == 1 or written % 100 == 0:
                if source_frame_count:
                    progress = f"{written}/{source_frame_count}"
                else:
                    progress = str(written)
                print(f"    frames: {progress}", end="\r", flush=True)
    except Exception:
        raise
    finally:
        reader.close()

    if written == 0:
        raise RuntimeError(
            f"no frames were decoded from video: {source_video}. "
            "The source codec may be unsupported by the active video backend."
        )
    if source_frame_count:
        print(f"    frames: {written}/{source_frame_count}")
    else:
        print(f"    frames: {written}")

    return {
        "source": str(source_video),
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": source_frame_count,
        "written_frame_count": written,
        "reader": "imageio",
    }


def augment_video_frames_with_cv2(
    source_video: Path, frames_dir: Path, spec: AugmentSpec, seed: int
) -> dict[str, object]:
    import cv2

    cap = cv2.VideoCapture(str(source_video))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {source_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        output_frame = apply_image_augmentation(frame, spec, rng)
        ok = cv2.imwrite(str(frames_dir / f"frame-{written:06d}.png"), output_frame)
        if not ok:
            cap.release()
            raise RuntimeError(f"failed to write augmented frame {written}")
        written += 1

    cap.release()
    if written == 0:
        raise RuntimeError(
            f"no frames were decoded from video: {source_video}. "
            "The source codec may be unsupported by the active OpenCV build."
        )
    return {
        "source": str(source_video),
        "fps": fps,
        "width": width,
        "height": height,
        "source_frame_count": frame_count,
        "written_frame_count": written,
        "reader": "opencv",
    }


def augment_dataset(
    data_root: Path,
    dataset_name: str,
    preset_name: str,
    variant_index: int = 1,
    output_name: str | None = None,
    overwrite: bool = False,
    seed: int = 0,
) -> dict[str, object]:
    if preset_name not in PRESETS:
        raise ValueError(
            f"unknown preset {preset_name!r}; choose one of: {', '.join(PRESETS)}"
        )

    source, target = resolve_dataset_paths(
        data_root, dataset_name, preset_name, variant_index, output_name
    )
    source_videos = video_files(source)
    if not source_videos:
        raise FileNotFoundError(f"no videos found under: {source / 'videos'}")

    rng = np.random.default_rng(seed + variant_index)
    spec = variant_spec(PRESETS[preset_name], variant_index, rng)
    copy_dataset_tree(source, target, overwrite=overwrite)

    video_reports = []
    for offset, source_video in enumerate(source_videos):
        relative = source_video.relative_to(source)
        target_video = target / relative
        print(
            f"  video [{offset + 1}/{len(source_videos)}] {relative}",
            flush=True,
        )
        report = augment_video(
            source_video,
            target_video,
            spec,
            seed=seed + variant_index * 1000 + offset,
        )
        print(f"    done: {report['written_frame_count']} frames", flush=True)
        video_reports.append(report)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dataset": str(source),
        "output_dataset": str(target),
        "preset": preset_name,
        "variant_index": variant_index,
        "spec": asdict(spec),
        "videos": video_reports,
        "note": "Only videos/**/*.mp4 were re-encoded. Non-video LeRobot metadata/action/state files were copied unchanged.",
    }
    report_path = target / "augmentation_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
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
        description="Create visually augmented copies of a LeRobot dataset."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--dataset", required=True, help="Dataset directory name under data root."
    )
    parser.add_argument(
        "--preset", default="all", help="Preset name, comma-list, or 'all'."
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Number of augmented dataset copies to create.",
    )
    parser.add_argument(
        "--output-name", default=None, help="Only valid when multiplier=1."
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260623)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.multiplier < 1:
        raise SystemExit("--multiplier must be >= 1")
    if args.output_name and args.multiplier != 1:
        raise SystemExit("--output-name can only be used when --multiplier=1")

    presets = choose_presets(args.preset, args.multiplier)
    reports = []
    for index, preset_name in enumerate(presets, start=1):
        print(f"[{index}/{args.multiplier}] creating preset={preset_name}")
        reports.append(
            augment_dataset(
                data_root=args.data_root,
                dataset_name=args.dataset,
                preset_name=preset_name,
                variant_index=index,
                output_name=args.output_name,
                overwrite=args.overwrite,
                seed=args.seed,
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
