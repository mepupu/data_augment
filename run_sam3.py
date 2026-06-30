import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

try:
    from ultralytics.models.sam import SAM3VideoSemanticPredictor
except ImportError:
    SAM3VideoSemanticPredictor = None


MODEL_CANDIDATES = (
    Path("/root/autodl-fs/sam3.pt"),
    Path("/root/sam3/sam3.pt"),
)
IMAGE_PATH = Path("/root/autodl-fs/datasets/head_first_frame.jpg")
OUTPUT_DIR = Path("/root/autodl-fs")
TABLE_OUTPUT = OUTPUT_DIR / "sam3_table_only.jpg"
BOX_OUTPUT = OUTPUT_DIR / "sam3_boxes_only.jpg"
GRIPPER_OUTPUT = OUTPUT_DIR / "sam3_grippers_only.jpg"
TABLE_TEXT_PROMPTS = (
    "tabletop",
    "table surface",
    "white table surface",
)
BIN_TEXT_PROMPTS = (
    "yellow plastic bin",
    "yellow storage box",
    "yellow container",
    "yellow material box",
)
GRIPPER_TEXT_PROMPTS = (
    "robot gripper",
    "robot end effector",
    "black robot gripper",
    "robot clamp",
)


def default_scene_config() -> dict:
    return {
        "model_candidates": [str(path) for path in MODEL_CANDIDATES],
        "image_path": str(IMAGE_PATH),
        "output_dir": str(OUTPUT_DIR),
        "objects": [
            {
                "name": "table",
                "prompts": list(TABLE_TEXT_PROMPTS),
            },
            {
                "name": "bins",
                "prompts": list(BIN_TEXT_PROMPTS),
                "output": BOX_OUTPUT.name,
            },
            {
                "name": "grippers",
                "prompts": list(GRIPPER_TEXT_PROMPTS),
                "output": GRIPPER_OUTPUT.name,
            },
        ],
        "composites": [
            {
                "name": "table_background",
                "base": "table",
                "subtract": ["bins", "grippers"],
                "output": TABLE_OUTPUT.name,
            }
        ],
    }


def load_scene_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        return default_scene_config()

    with Path(config_path).expanduser().open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    config.setdefault("model_candidates", [str(path) for path in MODEL_CANDIDATES])
    config.setdefault("image_path", str(IMAGE_PATH))
    config.setdefault("output_dir", str(OUTPUT_DIR))
    config.setdefault("objects", [])
    config.setdefault("composites", [])
    return config


def resolve_model_path(candidates=MODEL_CANDIDATES) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    searched = ", ".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"未找到 SAM3 模型文件，已检查: {searched}")


def resolve_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 AutoDL 实例是否有 GPU、torch 是否为 CUDA 版")
    return "cuda:0"


def load_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"未找到输入图片: {image_path}")
    return img


def mask_to_image_shape(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.shape != (height, width):
        mask_u8 = cv2.resize(mask_u8, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_u8 = (mask_u8 > 0).astype(np.uint8)
    return mask_u8


def merge_result_masks(results, height: int, width: int) -> np.ndarray:
    merged = np.zeros((height, width), dtype=np.uint8)
    for result in results:
        if getattr(result, "masks", None) is None:
            continue
        for mask in result.masks.data:
            mask_u8 = mask_to_image_shape(mask.cpu().numpy(), height, width)
            merged = np.maximum(merged, mask_u8)
    return merged


def normalize_results(results) -> list:
    if results is None:
        return []
    if isinstance(results, list):
        return results
    if isinstance(results, tuple):
        return list(results)
    return [results]


def _install_tokenizer_call_on_class(tokenizer_cls) -> bool:
    if "__call__" in getattr(tokenizer_cls, "__dict__", {}):
        return False

    def __call__(self, text, context_length=77):
        texts = [text] if isinstance(text, str) else list(text)
        start_token = getattr(self, "encoder", {}).get("<|startoftext|>")
        end_token = getattr(self, "encoder", {}).get("<|endoftext|>")
        tokenized = torch.zeros((len(texts), context_length), dtype=torch.long)

        for row, item in enumerate(texts):
            tokens = []
            if start_token is not None:
                tokens.append(start_token)
            tokens.extend(self.encode(item))
            if end_token is not None:
                tokens.append(end_token)
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                if end_token is not None:
                    tokens[-1] = end_token
            tokenized[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        return tokenized

    tokenizer_cls.__call__ = __call__
    return True


def _patch_tokenizers_in_object(obj, seen: set[int] | None = None) -> int:
    if obj is None:
        return 0
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    patched = 0
    obj_cls = type(obj)
    if obj_cls.__name__ == "SimpleTokenizer" and hasattr(obj, "encode"):
        patched += int(_install_tokenizer_call_on_class(obj_cls))

    for name in ("model", "backbone", "language_backbone", "tokenizer"):
        child = getattr(obj, name, None)
        if child is not None:
            patched += _patch_tokenizers_in_object(child, seen)
    return patched


def install_simple_tokenizer_call_patch(tokenizer_cls=None, root_obj=None) -> bool:
    patched = 0
    if tokenizer_cls is not None:
        patched += int(_install_tokenizer_call_on_class(tokenizer_cls))
    else:
        try:
            from ultralytics.models.sam.sam3.text_encoder_ve import SimpleTokenizer
        except ImportError:
            SimpleTokenizer = None
        if SimpleTokenizer is not None:
            patched += int(_install_tokenizer_call_on_class(SimpleTokenizer))

        for module in list(sys.modules.values()):
            candidate = getattr(module, "SimpleTokenizer", None)
            if isinstance(candidate, type):
                patched += int(_install_tokenizer_call_on_class(candidate))

    patched += _patch_tokenizers_in_object(root_obj)
    return patched > 0


def raise_clip_install_error(exc: TypeError) -> None:
    raise RuntimeError(
        "SAM3 文本分割需要 Ultralytics 适配的 CLIP。当前环境里的 tokenizer "
        "不兼容，导致 SimpleTokenizer 不可调用。\n\n"
        "请在 AutoDL 的 ultra-sam3 环境中执行：\n"
        "  pip uninstall clip -y\n"
        "  pip install git+https://github.com/ultralytics/CLIP.git\n\n"
        "然后重新运行：python run_sam3.py"
    ) from exc


def predict_semantic_results(
    model_path: Path,
    image_path: Path,
    predictor_cls=SAM3SemanticPredictor,
    prompts: tuple[str, ...] = TABLE_TEXT_PROMPTS,
    device: str = "cuda:0",
) -> list:
    install_simple_tokenizer_call_patch()
    predictor = predictor_cls(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "device": device,
            "verbose": False,
        }
    )
    predictor.set_image(str(image_path))
    install_simple_tokenizer_call_patch(root_obj=predictor)
    try:
        return normalize_results(predictor(text=list(prompts)))
    except TypeError as exc:
        if "SimpleTokenizer" not in str(exc) or "not callable" not in str(exc):
            raise
        install_simple_tokenizer_call_patch(root_obj=predictor)
        try:
            return normalize_results(predictor(text=list(prompts)))
        except TypeError as retry_exc:
            if "SimpleTokenizer" not in str(retry_exc) or "not callable" not in str(retry_exc):
                raise
            raise_clip_install_error(retry_exc)


def predict_table_surface_results(
    model_path: Path,
    image_path: Path,
    predictor_cls=SAM3SemanticPredictor,
    device: str = "cuda:0",
) -> list:
    return predict_semantic_results(
        model_path=model_path,
        image_path=image_path,
        predictor_cls=predictor_cls,
        prompts=TABLE_TEXT_PROMPTS,
        device=device,
    )


def predict_video_semantic_results(
    model_path: Path,
    video_path: Path,
    prompts: tuple[str, ...],
    predictor_cls=SAM3VideoSemanticPredictor,
    device: str = "cuda:0",
):
    if predictor_cls is None:
        raise RuntimeError("当前 Ultralytics 环境不支持 SAM3VideoSemanticPredictor")
    install_simple_tokenizer_call_patch()
    predictor = predictor_cls(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": str(model_path),
            "device": device,
            "verbose": False,
        }
    )
    install_simple_tokenizer_call_patch(root_obj=predictor)
    return predictor(source=str(video_path), text=list(prompts), stream=True)


def require_nonempty_mask(mask: np.ndarray, name: str) -> None:
    if not np.any(mask):
        raise RuntimeError(f"未能分割出{name}区域，请检查 SAM3 文本提示词或输入图像")


def remove_protected_regions(
    roi_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> np.ndarray:
    table_mask = roi_mask.copy()
    table_mask[protected_mask > 0] = 0
    return table_mask


def build_composite_mask(composite: dict, masks: dict[str, np.ndarray]) -> np.ndarray:
    base_name = composite["base"]
    if base_name not in masks:
        raise KeyError(f"composite base mask not found: {base_name}")

    protected = np.zeros_like(masks[base_name], dtype=np.uint8)
    for name in composite.get("subtract", []):
        if name not in masks:
            raise KeyError(f"composite subtract mask not found: {name}")
        protected = np.maximum(protected, masks[name])
    return remove_protected_regions(masks[base_name], protected)


def save_results(results, filename: Path) -> None:
    for result in results:
        result.save(filename=str(filename))


def save_mask_overlay(img: np.ndarray, mask: np.ndarray, filename: Path) -> None:
    output = img.copy()
    output[mask == 1] = [255, 0, 0]
    cv2.addWeighted(output, 0.4, img, 0.6, 0, output)
    cv2.imwrite(str(filename), output)


def apply_color_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color_bgr: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    output = frame.copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    color = np.array(color_bgr, dtype=np.float32)
    selected = mask > 0
    if np.any(selected):
        blended = output[selected].astype(np.float32) * (1.0 - alpha) + color * alpha
        output[selected] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def apply_video_effects(
    frame: np.ndarray,
    masks: dict[str, np.ndarray],
    effects: list[dict],
) -> np.ndarray:
    output = frame
    for effect in effects:
        if effect.get("type") != "color_overlay":
            raise ValueError(f"unsupported video effect type: {effect.get('type')}")
        target = effect["target"]
        if target not in masks:
            raise KeyError(f"video effect target mask not found: {target}")
        color = tuple(effect.get("color_bgr", effect.get("color", [255, 0, 0])))
        output = apply_color_overlay(
            output,
            masks[target],
            color_bgr=color,
            alpha=effect.get("alpha", 0.45),
        )
    return output


def bgr_to_rgb_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, :3][:, :, ::-1]
    return frame


class ImageioVideoWriter:
    def __init__(self, writer):
        self.writer = writer

    def write(self, frame: np.ndarray) -> None:
        self.writer.append_data(bgr_to_rgb_frame(frame))

    def release(self) -> None:
        self.writer.close()


def get_imageio_module():
    import imageio.v2 as imageio

    return imageio


def get_video_fps(video_path: Path, imageio_module=None) -> float:
    imageio = imageio_module or get_imageio_module()
    reader = imageio.get_reader(str(video_path))
    try:
        metadata = reader.get_meta_data()
        fps = metadata.get("fps")
    finally:
        reader.close()
    if not fps or fps <= 0:
        return 30.0
    return float(fps)


def open_video_writer(output_path: Path, fps: float, width: int, height: int):
    imageio = get_imageio_module()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output_path),
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    return ImageioVideoWriter(writer)


def is_video_config(config: dict) -> bool:
    return bool(config.get("video_path"))


def video_transcode_mode(config: dict) -> str:
    return str(config.get("transcode_input", "auto")).lower()


def can_decode_video(video_path: Path) -> bool:
    imageio = get_imageio_module()
    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
        return frame is not None
    except Exception:
        return False
    finally:
        reader.close()


def video_codec_name(video_path: Path) -> str | None:
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-i",
        str(video_path),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        return None

    output = "\n".join(part for part in (result.stderr, result.stdout) if part)
    match = re.search(r"Video:\s*([^,\s]+)", output)
    if not match:
        return None
    return match.group(1).lower()


def codec_requires_transcode_for_sam3(codec_name: str | None) -> bool:
    if not codec_name:
        return False
    return codec_name.lower() in {"av1", "av01"}


def ffmpeg_executable() -> str:
    try:
        import imageio_ffmpeg
    except ImportError:
        return "ffmpeg"
    return imageio_ffmpeg.get_ffmpeg_exe()


def transcode_video_to_h264(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(),
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("未找到 ffmpeg，请安装 imageio-ffmpeg 或系统 ffmpeg") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"视频转码失败: {source} -> {target}") from exc
    return target


def prepare_video_for_sam3(
    video_path: Path,
    config: dict,
    can_decode_video=can_decode_video,
    transcode_video=transcode_video_to_h264,
    video_codec_name=video_codec_name,
) -> Path:
    mode = video_transcode_mode(config)
    if mode in ("false", "no", "off", "0", "none"):
        return video_path
    if mode == "auto":
        codec_name = video_codec_name(video_path)
        if not codec_requires_transcode_for_sam3(codec_name) and can_decode_video(video_path):
            return video_path
    if mode not in ("auto", "true", "yes", "on", "1", "always"):
        raise ValueError(f"unsupported transcode_input value: {config.get('transcode_input')}")

    transcode_dir = Path(config.get("transcode_dir", "/tmp/sam3_video_inputs")).expanduser()
    target = transcode_dir / f"{video_path.stem}_h264.mp4"
    print(f"输入视频可能无法被当前环境解码，正在转码为 H.264: {target}")
    return transcode_video(video_path, target)


def raise_zero_frame_video_error(video_path: Path) -> None:
    raise RuntimeError(
        f"视频没有产生可写出的帧: {video_path}\n"
        "这通常是输入视频编码当前环境无法解码导致的，常见于 AV1/av01 视频。\n"
        "可以先转为 H.264/yuv420p 再运行，例如：\n"
        f"  ffmpeg -y -i {video_path} -map 0:v:0 -an -c:v libx264 -pix_fmt yuv420p /tmp/{video_path.stem}_h264.mp4\n"
        "也可以在配置中保持默认 transcode_input=auto，让脚本自动转码。"
    )


def result_frame(result) -> np.ndarray:
    frame = getattr(result, "orig_img", None)
    if frame is None:
        raise RuntimeError("SAM3 视频结果缺少 orig_img，无法写出处理后视频")
    return frame.copy()


def run_video_config(config: dict) -> Path:
    device = resolve_device()
    model_candidates = [Path(path) for path in config.get("model_candidates", MODEL_CANDIDATES)]
    model_path = resolve_model_path(model_candidates)
    print(f"SAM 3 模型加载成功: {model_path}")

    original_video_path = Path(config["video_path"]).expanduser()
    video_path = prepare_video_for_sam3(original_video_path, config)
    output_video = Path(config["output_video"]).expanduser()
    objects = config.get("video_objects", config.get("objects", []))
    if len(objects) != 1:
        raise ValueError("video mode currently expects exactly one video object")
    target = objects[0]
    target_name = target["name"]
    effects = config.get("effects", [])
    if not effects:
        raise ValueError("video mode requires at least one effect")

    print(f"正在通过 SAM3 视频接口跟踪 {target_name}...")
    results = predict_video_semantic_results(
        model_path=model_path,
        video_path=video_path,
        prompts=tuple(target["prompts"]),
        device=device,
    )
    fps = get_video_fps(video_path)
    writer = None
    frame_count = 0
    try:
        for result in results:
            frame = result_frame(result)
            height, width = frame.shape[:2]
            if writer is None:
                writer = open_video_writer(output_video, fps, width, height)
            mask = merge_result_masks([result], height, width)
            if target.get("required", True):
                require_nonempty_mask(mask, target_name)
            output = apply_video_effects(frame, {target_name: mask}, effects)
            writer.write(output)
            frame_count += 1
    finally:
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise_zero_frame_video_error(original_video_path)
    print(f"\nSAM3 视频处理完成: {output_video}")
    return output_video


def run_scene_config(config: dict) -> dict[str, Path]:
    device = resolve_device()
    model_candidates = [Path(path) for path in config.get("model_candidates", MODEL_CANDIDATES)]
    model_path = resolve_model_path(model_candidates)
    print(f"SAM 3 模型加载成功: {model_path}")

    image_path = Path(config["image_path"]).expanduser()
    output_dir = Path(config["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    img = load_image(image_path)
    height, width, _ = img.shape
    masks: dict[str, np.ndarray] = {}
    outputs: dict[str, Path] = {}

    for obj in config["objects"]:
        name = obj["name"]
        prompts = tuple(obj["prompts"])
        print(f"正在通过 SAM3 文本提示词分割 {name}...")
        results = predict_semantic_results(model_path, image_path, prompts=prompts, device=device)
        mask = merge_result_masks(results, height, width)
        if obj.get("required", True):
            require_nonempty_mask(mask, name)
        masks[name] = mask

        if obj.get("output"):
            output_path = output_dir / obj["output"]
            save_results(results, output_path)
            outputs[name] = output_path

    for composite in config.get("composites", []):
        name = composite["name"]
        print(f"正在生成组合掩码 {name}...")
        mask = build_composite_mask(composite, masks)
        if composite.get("required", True):
            require_nonempty_mask(mask, name)
        output_path = output_dir / composite["output"]
        save_mask_overlay(img, mask, output_path)
        outputs[name] = output_path

    print("\nSAM3 场景分割完成")
    print("生成的文件包括：")
    for index, (name, path) in enumerate(outputs.items(), start=1):
        print(f"  - {index}. {name}: {path}")
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configurable SAM3 semantic segmentation.")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON scene config. Defaults to the built-in table/bins/grippers scene.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_scene_config(args.config)
    if is_video_config(config):
        run_video_config(config)
    else:
        run_scene_config(config)


if __name__ == "__main__":
    main()
