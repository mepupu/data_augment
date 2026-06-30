import importlib.util
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


def resize_nearest(mask, size, interpolation=None):
    width, height = size
    y_idx = (np.arange(height) * mask.shape[0] / height).astype(int)
    x_idx = (np.arange(width) * mask.shape[1] / width).astype(int)
    return mask[y_idx[:, None], x_idx]


MODULE_PATH = Path(__file__).resolve().parents[1] / "run_sam3.py"
sys.modules.setdefault(
    "cv2",
    types.SimpleNamespace(
        resize=resize_nearest,
        INTER_NEAREST=0,
    ),
)
sys.modules.setdefault(
    "torch",
    types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        long=np.int64,
        tensor=lambda value, dtype=None: np.array(value, dtype=dtype),
        zeros=lambda shape, dtype=None: np.zeros(shape, dtype=dtype),
    ),
)
sys.modules.setdefault(
    "ultralytics",
    types.SimpleNamespace(SAM=object),
)
sys.modules.setdefault("ultralytics.models", types.SimpleNamespace())
sys.modules.setdefault(
    "ultralytics.models.sam",
    types.SimpleNamespace(SAM3SemanticPredictor=object, SAM3VideoSemanticPredictor=object),
)
spec = importlib.util.spec_from_file_location("run_sam3", MODULE_PATH)
run_sam3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(run_sam3)


class RunSam3HelperTests(unittest.TestCase):
    def test_resolve_model_path_requires_existing_file(self):
        missing_a = Path("/tmp/missing-a.pt")
        missing_b = Path("/tmp/missing-b.pt")

        with self.assertRaises(FileNotFoundError) as ctx:
            run_sam3.resolve_model_path([missing_a, missing_b])

        self.assertIn(str(missing_a), str(ctx.exception))
        self.assertIn(str(missing_b), str(ctx.exception))

    def test_mask_to_image_shape_resizes_to_image_size(self):
        mask = np.ones((2, 3), dtype=np.uint8)

        resized = run_sam3.mask_to_image_shape(mask, height=4, width=6)

        self.assertEqual(resized.shape, (4, 6))
        self.assertEqual(resized.dtype, np.uint8)
        self.assertEqual(resized.max(), 1)

    def test_merge_result_masks_handles_multiple_results_and_resizes(self):
        class FakeTensor:
            def __init__(self, arr):
                self.arr = arr

            def cpu(self):
                return self

            def numpy(self):
                return self.arr

        class FakeMasks:
            def __init__(self, arrays):
                self.data = [FakeTensor(arr) for arr in arrays]

        class FakeResult:
            def __init__(self, arrays):
                self.masks = FakeMasks(arrays)

        results = [
            FakeResult([np.array([[1, 0], [0, 0]], dtype=np.uint8)]),
            FakeResult([np.array([[0, 0], [0, 1]], dtype=np.uint8)]),
        ]

        merged = run_sam3.merge_result_masks(results, height=2, width=2)

        self.assertEqual(merged.tolist(), [[1, 0], [0, 1]])

    def test_predict_table_surface_results_uses_text_prompts(self):
        class FakePredictor:
            instances = []

            def __init__(self, overrides):
                self.overrides = overrides
                self.image_path = None
                self.text = None
                FakePredictor.instances.append(self)

            def set_image(self, image_path):
                self.image_path = image_path

            def __call__(self, text):
                self.text = text
                return ["table-result"]

        results = run_sam3.predict_table_surface_results(
            model_path=Path("/tmp/sam3.pt"),
            image_path=Path("/tmp/frame.jpg"),
            predictor_cls=FakePredictor,
        )

        predictor = FakePredictor.instances[0]
        self.assertEqual(results, ["table-result"])
        self.assertEqual(predictor.image_path, "/tmp/frame.jpg")
        self.assertEqual(predictor.text, list(run_sam3.TABLE_TEXT_PROMPTS))
        self.assertEqual(predictor.overrides["model"], "/tmp/sam3.pt")
        self.assertEqual(predictor.overrides["task"], "segment")
        self.assertEqual(predictor.overrides["device"], "cuda:0")
        self.assertNotIn("quantize", predictor.overrides)

    def test_predict_semantic_results_accepts_object_prompts(self):
        class FakePredictor:
            instances = []

            def __init__(self, overrides):
                self.overrides = overrides
                self.image_path = None
                self.text = None
                FakePredictor.instances.append(self)

            def set_image(self, image_path):
                self.image_path = image_path

            def __call__(self, text):
                self.text = text
                return ("semantic-result",)

        results = run_sam3.predict_semantic_results(
            model_path=Path("/tmp/sam3.pt"),
            image_path=Path("/tmp/frame.jpg"),
            prompts=("yellow plastic bin", "yellow storage box"),
            predictor_cls=FakePredictor,
            device="cuda:1",
        )

        predictor = FakePredictor.instances[0]
        self.assertEqual(results, ["semantic-result"])
        self.assertEqual(predictor.text, ["yellow plastic bin", "yellow storage box"])
        self.assertEqual(predictor.overrides["device"], "cuda:1")

    def test_prompt_sets_cover_table_bins_and_grippers(self):
        self.assertIn("table surface", run_sam3.TABLE_TEXT_PROMPTS)
        self.assertIn("yellow plastic bin", run_sam3.BIN_TEXT_PROMPTS)
        self.assertIn("robot gripper", run_sam3.GRIPPER_TEXT_PROMPTS)

    def test_install_simple_tokenizer_call_patch_adds_callable_encode_wrapper(self):
        class FakeTokenizer:
            encoder = {
                "<|startoftext|>": 101,
                "<|endoftext|>": 102,
            }

            def encode(self, text):
                return [ord(char) for char in text]

        patched = run_sam3.install_simple_tokenizer_call_patch(FakeTokenizer)
        tokenizer = FakeTokenizer()
        tokenized = tokenizer(["ab", "c"], context_length=5)

        self.assertTrue(patched)
        self.assertEqual(tokenized.tolist(), [[101, 97, 98, 102, 0], [101, 99, 102, 0, 0]])

    def test_install_simple_tokenizer_call_patch_leaves_callable_class_unchanged(self):
        class CallableTokenizer:
            def __call__(self, text, context_length=77):
                return text

        patched = run_sam3.install_simple_tokenizer_call_patch(CallableTokenizer)

        self.assertFalse(patched)

    def test_install_simple_tokenizer_call_patch_finds_predictor_tokenizer(self):
        class SimpleTokenizer:
            encoder = {
                "<|startoftext|>": 101,
                "<|endoftext|>": 102,
            }

            def encode(self, text):
                return [ord(char) for char in text]

        predictor = types.SimpleNamespace(
            model=types.SimpleNamespace(
                backbone=types.SimpleNamespace(
                    language_backbone=types.SimpleNamespace(tokenizer=SimpleTokenizer())
                )
            )
        )

        patched = run_sam3.install_simple_tokenizer_call_patch(root_obj=predictor)

        self.assertTrue(patched)
        self.assertTrue(callable(predictor.model.backbone.language_backbone.tokenizer))

    def test_predict_semantic_results_explains_simple_tokenizer_type_error(self):
        class SimpleTokenizer:
            encoder = {
                "<|startoftext|>": 101,
                "<|endoftext|>": 102,
            }

            def encode(self, text):
                return [ord(char) for char in text]

        class FakePredictor:
            calls = 0

            def __init__(self, overrides):
                self.model = types.SimpleNamespace(
                    backbone=types.SimpleNamespace(
                        language_backbone=types.SimpleNamespace(tokenizer=SimpleTokenizer())
                    )
                )

            def set_image(self, image_path):
                self.image_path = image_path

            def __call__(self, text):
                FakePredictor.calls += 1
                raise TypeError("'SimpleTokenizer' object is not callable")

        with self.assertRaises(RuntimeError) as ctx:
            run_sam3.predict_semantic_results(
                model_path=Path("/tmp/sam3.pt"),
                image_path=Path("/tmp/frame.jpg"),
                predictor_cls=FakePredictor,
                prompts=("table surface",),
            )

        self.assertEqual(FakePredictor.calls, 2)
        self.assertIn("pip uninstall clip -y", str(ctx.exception))
        self.assertIn("ultralytics/CLIP", str(ctx.exception))

    def test_require_nonempty_mask_rejects_empty_table_mask(self):
        empty = np.zeros((3, 3), dtype=np.uint8)

        with self.assertRaises(RuntimeError) as ctx:
            run_sam3.require_nonempty_mask(empty, "桌面")

        self.assertIn("桌面", str(ctx.exception))

    def test_remove_protected_regions_keeps_only_roi_minus_objects(self):
        roi_mask = np.ones((4, 4), dtype=np.uint8)
        protected_mask = np.zeros((4, 4), dtype=np.uint8)
        protected_mask[1:3, 1:3] = 1

        table_mask = run_sam3.remove_protected_regions(roi_mask, protected_mask)

        self.assertEqual(table_mask[0, 0], 1)
        self.assertEqual(table_mask[1, 1], 0)
        self.assertEqual(table_mask[2, 2], 0)

    def test_default_scene_config_keeps_current_objects_and_composite(self):
        config = run_sam3.default_scene_config()

        object_names = [item["name"] for item in config["objects"]]
        self.assertEqual(object_names, ["table", "bins", "grippers"])
        self.assertEqual(config["composites"][0]["base"], "table")
        self.assertEqual(config["composites"][0]["subtract"], ["bins", "grippers"])
        self.assertEqual(config["composites"][0]["output"], "sam3_table_only.jpg")

    def test_load_scene_config_reads_json_file(self):
        payload = {
            "image_path": "/tmp/frame.jpg",
            "output_dir": "/tmp/out",
            "objects": [
                {
                    "name": "shelf",
                    "prompts": ["storage rack", "metal shelf"],
                    "output": "shelf.jpg",
                }
            ],
            "composites": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "scene.json"
            config_path.write_text(json.dumps(payload), encoding="utf-8")

            config = run_sam3.load_scene_config(config_path)

        self.assertEqual(config["image_path"], "/tmp/frame.jpg")
        self.assertEqual(config["objects"][0]["name"], "shelf")
        self.assertEqual(config["model_candidates"], [str(path) for path in run_sam3.MODEL_CANDIDATES])

    def test_build_composite_mask_subtracts_named_masks(self):
        masks = {
            "table": np.ones((3, 3), dtype=np.uint8),
            "bins": np.array(
                [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            "grippers": np.array(
                [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        }

        composite = run_sam3.build_composite_mask(
            {"base": "table", "subtract": ["bins", "grippers"]}, masks
        )

        self.assertEqual(composite[0, 1], 0)
        self.assertEqual(composite[1, 1], 0)
        self.assertEqual(composite[2, 2], 1)

    def test_apply_color_overlay_changes_only_masked_pixels(self):
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)

        output = run_sam3.apply_color_overlay(
            frame, mask, color_bgr=(0, 0, 100), alpha=0.5
        )

        self.assertEqual(output[0, 0].tolist(), [0, 0, 50])
        self.assertEqual(output[0, 1].tolist(), [0, 0, 0])

    def test_predict_video_semantic_results_uses_official_video_predictor(self):
        class FakeVideoPredictor:
            instances = []

            def __init__(self, overrides):
                self.overrides = overrides
                self.call_kwargs = None
                FakeVideoPredictor.instances.append(self)

            def __call__(self, **kwargs):
                self.call_kwargs = kwargs
                return ["video-result"]

        results = run_sam3.predict_video_semantic_results(
            model_path=Path("/tmp/sam3.pt"),
            video_path=Path("/tmp/demo.mp4"),
            prompts=("table surface",),
            predictor_cls=FakeVideoPredictor,
            device="cuda:2",
        )

        predictor = FakeVideoPredictor.instances[0]
        self.assertEqual(results, ["video-result"])
        self.assertEqual(predictor.overrides["model"], "/tmp/sam3.pt")
        self.assertEqual(predictor.overrides["device"], "cuda:2")
        self.assertEqual(predictor.call_kwargs["source"], "/tmp/demo.mp4")
        self.assertEqual(predictor.call_kwargs["text"], ["table surface"])
        self.assertTrue(predictor.call_kwargs["stream"])

    def test_config_with_video_path_uses_video_mode(self):
        image_config = {"image_path": "/tmp/frame.jpg"}
        video_config = {"video_path": "/tmp/demo.mp4"}

        self.assertFalse(run_sam3.is_video_config(image_config))
        self.assertTrue(run_sam3.is_video_config(video_config))

    def test_video_config_defaults_to_auto_transcode(self):
        config = {"video_path": "/tmp/demo.mp4"}

        self.assertEqual(run_sam3.video_transcode_mode(config), "auto")

    def test_prepare_video_for_sam3_transcodes_when_probe_fails(self):
        calls = []

        def fake_can_decode(path):
            return False

        def fake_transcode(source, target):
            calls.append((source, target))
            return target

        with contextlib.redirect_stdout(io.StringIO()):
            prepared = run_sam3.prepare_video_for_sam3(
                Path("/tmp/demo.mp4"),
                {"transcode_input": "auto", "transcode_dir": "/tmp/sam3-cache"},
                can_decode_video=fake_can_decode,
                transcode_video=fake_transcode,
            )

        self.assertEqual(prepared, Path("/tmp/sam3-cache/demo_h264.mp4"))
        self.assertEqual(calls, [(Path("/tmp/demo.mp4"), Path("/tmp/sam3-cache/demo_h264.mp4"))])

    def test_prepare_video_for_sam3_keeps_decodable_video_in_auto_mode(self):
        prepared = run_sam3.prepare_video_for_sam3(
            Path("/tmp/demo.mp4"),
            {"transcode_input": "auto"},
            can_decode_video=lambda path: True,
            transcode_video=lambda source, target: target,
        )

        self.assertEqual(prepared, Path("/tmp/demo.mp4"))

    def test_zero_frame_video_error_mentions_av1_transcode(self):
        with self.assertRaises(RuntimeError) as ctx:
            run_sam3.raise_zero_frame_video_error(Path("/tmp/demo.mp4"))

        self.assertIn("AV1", str(ctx.exception))
        self.assertIn("ffmpeg", str(ctx.exception))

    def test_bgr_to_rgb_frame_swaps_channels_for_imageio(self):
        frame = np.array([[[1, 2, 3]]], dtype=np.uint8)

        rgb = run_sam3.bgr_to_rgb_frame(frame)

        self.assertEqual(rgb.tolist(), [[[3, 2, 1]]])

    def test_imageio_video_writer_appends_rgb_frames(self):
        appended = []

        class FakeWriter:
            def append_data(self, frame):
                appended.append(frame.copy())

            def close(self):
                appended.append("closed")

        writer = run_sam3.ImageioVideoWriter(FakeWriter())
        writer.write(np.array([[[1, 2, 3]]], dtype=np.uint8))
        writer.release()

        self.assertEqual(appended[0].tolist(), [[[3, 2, 1]]])
        self.assertEqual(appended[1], "closed")

    def test_get_video_fps_uses_imageio_metadata(self):
        class FakeReader:
            def get_meta_data(self):
                return {"fps": 24}

            def close(self):
                pass

        class FakeImageio:
            @staticmethod
            def get_reader(path):
                return FakeReader()

        fps = run_sam3.get_video_fps(Path("/tmp/demo.mp4"), imageio_module=FakeImageio)

        self.assertEqual(fps, 24.0)


if __name__ == "__main__":
    unittest.main()
