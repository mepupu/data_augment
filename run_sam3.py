from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3SemanticPredictor


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


def install_simple_tokenizer_call_patch(tokenizer_cls=None) -> bool:
    if tokenizer_cls is None:
        try:
            from ultralytics.models.sam.sam3.text_encoder_ve import SimpleTokenizer
        except ImportError:
            return False
        tokenizer_cls = SimpleTokenizer

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
    return normalize_results(predictor(text=list(prompts)))


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


def save_results(results, filename: Path) -> None:
    for result in results:
        result.save(filename=str(filename))


def main() -> None:
    device = resolve_device()
    model_path = resolve_model_path()
    print(f"SAM 3 模型加载成功: {model_path}")

    img = load_image(IMAGE_PATH)
    height, width, _ = img.shape

    print("正在通过 SAM3 文本提示词分割桌面区域...")
    table_results = predict_table_surface_results(model_path, IMAGE_PATH, device=device)
    table_surface_mask = merge_result_masks(table_results, height, width)
    require_nonempty_mask(table_surface_mask, "桌面")

    print("正在通过 SAM3 文本提示词分割黄色物料盒...")
    box_results = predict_semantic_results(
        model_path, IMAGE_PATH, prompts=BIN_TEXT_PROMPTS, device=device
    )
    box_mask = merge_result_masks(box_results, height, width)
    require_nonempty_mask(box_mask, "物料盒")

    print("正在通过 SAM3 文本提示词分割机器人夹爪...")
    gripper_results = predict_semantic_results(
        model_path, IMAGE_PATH, prompts=GRIPPER_TEXT_PROMPTS, device=device
    )
    gripper_mask = merge_result_masks(gripper_results, height, width)
    require_nonempty_mask(gripper_mask, "夹爪")

    protected_mask = np.maximum(box_mask, gripper_mask)
    table_mask = remove_protected_regions(table_surface_mask, protected_mask)

    table_output = img.copy()
    table_output[table_mask == 1] = [255, 0, 0]
    cv2.addWeighted(table_output, 0.4, img, 0.6, 0, table_output)

    cv2.imwrite(str(TABLE_OUTPUT), table_output)
    save_results(box_results, BOX_OUTPUT)
    save_results(gripper_results, GRIPPER_OUTPUT)

    print("\n双臂机器人工作台全场景解耦分割完成")
    print("生成的独立特征文件包括：")
    print(f"  - 1. 物料盒区域:   {BOX_OUTPUT}")
    print(f"  - 2. 夹爪区域:     {GRIPPER_OUTPUT}")
    print(f"  - 3. 桌面候选背景: {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
