from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM


MODEL_CANDIDATES = (
    Path("/root/autodl-fs/sam3.pt"),
    Path("/root/sam3/sam3.pt"),
)
IMAGE_PATH = Path("/root/autodl-fs/datasets/head_first_frame.jpg")
OUTPUT_DIR = Path("/root/autodl-fs")
TABLE_OUTPUT = OUTPUT_DIR / "sam3_table_only.jpg"
BOX_OUTPUT = OUTPUT_DIR / "sam3_boxes_only.jpg"
LEFT_GRIPPER_OUTPUT = OUTPUT_DIR / "sam3_left_gripper_only.jpg"
RIGHT_GRIPPER_OUTPUT = OUTPUT_DIR / "sam3_right_gripper_only.jpg"


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


def require_prompt_bboxes(prompt_bboxes: list[list[int]]) -> None:
    if not prompt_bboxes:
        raise RuntimeError("未通过 HSV 找到黄色物料盒框，请检查阈值或输入图像")


def load_image(image_path: Path) -> np.ndarray:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"未找到输入图片: {image_path}")
    return img


def detect_yellow_bboxes(img: np.ndarray) -> list[list[int]]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([11, 40, 46])
    upper_yellow = np.array([30, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(
        yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    prompt_bboxes: list[list[int]] = []
    for contour in contours:
        if cv2.contourArea(contour) > 800:
            bx, by, bw, bh = cv2.boundingRect(contour)
            prompt_bboxes.append([bx, by, bx + bw, by + bh])
    return prompt_bboxes


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


def predict_gripper(model: SAM, image_path: Path, point: list[int], device: str):
    return model.predict(
        source=str(image_path),
        points=[point],
        labels=[1],
        device=device,
        verbose=False,
    )


def save_results(results, filename: Path) -> None:
    for result in results:
        result.save(filename=str(filename))


def main() -> None:
    device = resolve_device()
    model_path = resolve_model_path()
    model = SAM(str(model_path))
    print(f"SAM 3 模型加载成功: {model_path}")

    img = load_image(IMAGE_PATH)
    height, width, _ = img.shape

    prompt_bboxes = detect_yellow_bboxes(img)
    require_prompt_bboxes(prompt_bboxes)
    print(f"检测到 {len(prompt_bboxes)} 个黄色物料盒候选框")

    box_results = model.predict(
        source=str(IMAGE_PATH),
        bboxes=prompt_bboxes,
        device=device,
        verbose=False,
    )

    left_gripper_point = [int(width * 0.25), int(height * 0.85)]
    right_gripper_point = [int(width * 0.78), int(height * 0.85)]
    left_gripper_results = predict_gripper(
        model, IMAGE_PATH, left_gripper_point, device
    )
    right_gripper_results = predict_gripper(
        model, IMAGE_PATH, right_gripper_point, device
    )

    print("正在通过非目标减法计算桌面候选掩码...")
    total_workspace_mask = np.zeros((height, width), dtype=np.uint8)
    total_workspace_mask[250:height, :] = 1

    protected_mask = merge_result_masks(box_results, height, width)
    protected_mask = np.maximum(
        protected_mask,
        merge_result_masks(left_gripper_results, height, width),
    )
    protected_mask = np.maximum(
        protected_mask,
        merge_result_masks(right_gripper_results, height, width),
    )
    total_workspace_mask[protected_mask > 0] = 0

    table_output = img.copy()
    table_output[total_workspace_mask == 1] = [255, 0, 0]
    cv2.addWeighted(table_output, 0.4, img, 0.6, 0, table_output)

    cv2.imwrite(str(TABLE_OUTPUT), table_output)
    save_results(box_results, BOX_OUTPUT)
    save_results(left_gripper_results, LEFT_GRIPPER_OUTPUT)
    save_results(right_gripper_results, RIGHT_GRIPPER_OUTPUT)

    print("\n双臂机器人工作台全场景解耦分割完成")
    print("生成的独立特征文件包括：")
    print(f"  - 1. 物料盒区域:   {BOX_OUTPUT}")
    print(f"  - 2. 左夹爪区域:   {LEFT_GRIPPER_OUTPUT}")
    print(f"  - 3. 右夹爪区域:   {RIGHT_GRIPPER_OUTPUT}")
    print(f"  - 4. 桌面候选背景: {TABLE_OUTPUT}")


if __name__ == "__main__":
    main()
