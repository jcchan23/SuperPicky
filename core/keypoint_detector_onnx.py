"""
ONNX keypoint detector module.

Drop-in alternative to core.keypoint_detector using
models/cub200_keypoint_resnet50.onnx via onnxruntime.
"""

import argparse
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def _create_session_with_fallback(model_path: str, providers):
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as exc:
        print(f"[Keypoint ONNX] CUDA init failed, fallback to CPU: {exc}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


@dataclass
class KeypointResult:
    left_eye: Tuple[float, float]
    right_eye: Tuple[float, float]
    beak: Tuple[float, float]
    left_eye_vis: float
    right_eye_vis: float
    beak_vis: float
    both_eyes_hidden: bool
    all_keypoints_hidden: bool
    best_eye_visibility: float
    visible_eye: Optional[str]
    head_sharpness: float


class ONNXKeypointDetector:
    IMG_SIZE = 416
    VISIBILITY_THRESHOLD = 0.3
    RADIUS_MULTIPLIER = 1.2
    NO_BEAK_RADIUS_RATIO = 0.15

    @staticmethod
    def _get_default_model_path() -> str:
        import sys

        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, "models", "cub200_keypoint_resnet50.onnx")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "models", "cub200_keypoint_resnet50.onnx")

    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._get_default_model_path()
        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.coords_name: Optional[str] = None
        self.vis_name: Optional[str] = None

    @staticmethod
    def _build_providers():
        available = set(ort.get_available_providers())
        providers = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def load_model(self):
        if self.session is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Keypoint model not found: {self.model_path}")

        providers = self._build_providers()
        self.session = _create_session_with_fallback(self.model_path, providers)
        self.input_name = self.session.get_inputs()[0].name

        outputs = self.session.get_outputs()
        if len(outputs) < 2:
            raise RuntimeError("Unexpected ONNX outputs for keypoint model")
        self.coords_name = outputs[0].name
        self.vis_name = outputs[1].name

    def _preprocess(self, pil_image: Image.Image) -> np.ndarray:
        img = pil_image.resize((self.IMG_SIZE, self.IMG_SIZE), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0).astype(np.float32)

    def detect(
        self,
        bird_crop: np.ndarray,
        box: Tuple[int, int, int, int] = None,
        seg_mask: np.ndarray = None,
    ) -> Optional[KeypointResult]:
        self.load_model()

        if bird_crop is None or bird_crop.size == 0:
            return None

        pil_crop = Image.fromarray(bird_crop).convert("RGB")
        batch = self._preprocess(pil_crop)

        coords, vis = self.session.run([self.coords_name, self.vis_name], {self.input_name: batch})
        coords = np.asarray(coords)[0]
        vis = np.asarray(vis)[0]

        left_eye = (float(coords[0, 0]), float(coords[0, 1]))
        right_eye = (float(coords[1, 0]), float(coords[1, 1]))
        beak = (float(coords[2, 0]), float(coords[2, 1]))

        left_eye_vis = float(vis[0])
        right_eye_vis = float(vis[1])
        beak_vis = float(vis[2])

        left_visible = left_eye_vis >= self.VISIBILITY_THRESHOLD
        right_visible = right_eye_vis >= self.VISIBILITY_THRESHOLD
        beak_visible = beak_vis >= self.VISIBILITY_THRESHOLD

        both_eyes_hidden = (not left_visible) and (not right_visible)
        all_keypoints_hidden = (not left_visible) and (not right_visible) and (not beak_visible)

        if left_visible and right_visible:
            visible_eye = "both"
        elif left_visible:
            visible_eye = "left"
        elif right_visible:
            visible_eye = "right"
        else:
            visible_eye = None

        head_sharpness = 0.0
        if visible_eye is not None:
            head_sharpness = self._calculate_head_sharpness(
                bird_crop,
                left_eye,
                right_eye,
                beak,
                left_eye_vis,
                right_eye_vis,
                beak_visible,
                box,
                seg_mask,
            )

        best_eye_visibility = max(left_eye_vis, right_eye_vis)

        return KeypointResult(
            left_eye=left_eye,
            right_eye=right_eye,
            beak=beak,
            left_eye_vis=left_eye_vis,
            right_eye_vis=right_eye_vis,
            beak_vis=beak_vis,
            both_eyes_hidden=both_eyes_hidden,
            all_keypoints_hidden=all_keypoints_hidden,
            best_eye_visibility=best_eye_visibility,
            visible_eye=visible_eye,
            head_sharpness=head_sharpness,
        )

    def _calculate_head_sharpness(
        self,
        bird_crop: np.ndarray,
        left_eye: Tuple[float, float],
        right_eye: Tuple[float, float],
        beak: Tuple[float, float],
        left_eye_vis: float,
        right_eye_vis: float,
        beak_visible: bool,
        box: Tuple[int, int, int, int] = None,
        seg_mask: np.ndarray = None,
    ) -> float:
        h, w = bird_crop.shape[:2]

        if left_eye_vis < self.VISIBILITY_THRESHOLD and right_eye_vis < self.VISIBILITY_THRESHOLD:
            eye = left_eye if left_eye_vis >= right_eye_vis else right_eye
            eye_px = (int(eye[0] * w), int(eye[1] * h))
            beak_px = (int(beak[0] * w), int(beak[1] * h))
            # Bug fix vs original line ~246: use beak_visible (not undefined beak_vis)
            if beak_visible:
                radius = int(self._distance(eye_px, beak_px) * self.RADIUS_MULTIPLIER)
            elif box is not None:
                box_size = max(box[2], box[3])
                radius = int(box_size * self.NO_BEAK_RADIUS_RATIO)
            else:
                radius = int(max(w, h) * self.NO_BEAK_RADIUS_RATIO)
            radius = max(10, min(radius, min(w, h) // 2))
            circle_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(circle_mask, eye_px, radius, 255, -1)
            if seg_mask is not None and seg_mask.shape[:2] == (h, w):
                head_mask = cv2.bitwise_and(circle_mask, seg_mask)
            else:
                head_mask = circle_mask
            return self._calculate_sharpness(bird_crop, head_mask) * 0.8

        if left_eye_vis >= self.VISIBILITY_THRESHOLD and right_eye_vis >= self.VISIBILITY_THRESHOLD:
            left_dist = self._distance(left_eye, beak)
            right_dist = self._distance(right_eye, beak)
            eye = left_eye if left_dist >= right_dist else right_eye
        elif left_eye_vis >= self.VISIBILITY_THRESHOLD:
            eye = left_eye
        else:
            eye = right_eye

        eye_px = (int(eye[0] * w), int(eye[1] * h))
        beak_px = (int(beak[0] * w), int(beak[1] * h))

        if beak_visible:
            radius = int(self._distance(eye_px, beak_px) * self.RADIUS_MULTIPLIER)
        elif box is not None:
            box_size = max(box[2], box[3])
            radius = int(box_size * self.NO_BEAK_RADIUS_RATIO)
        else:
            radius = int(max(w, h) * self.NO_BEAK_RADIUS_RATIO)

        radius = max(10, min(radius, min(w, h) // 2))

        circle_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(circle_mask, eye_px, radius, 255, -1)

        if seg_mask is not None and seg_mask.shape[:2] == (h, w):
            head_mask = cv2.bitwise_and(circle_mask, seg_mask)
        else:
            head_mask = circle_mask

        return self._calculate_sharpness(bird_crop, head_mask)

    def _calculate_sharpness(self, image: np.ndarray, mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = gx ** 2 + gy ** 2

        mask_pixels = mask > 0
        if mask_pixels.sum() == 0:
            return 0.0

        raw_sharpness = float(gradient_magnitude[mask_pixels].mean())

        min_val = 100.0
        max_val = 154016.0

        if raw_sharpness <= min_val:
            return 0.0
        if raw_sharpness >= max_val:
            return 1000.0

        log_val = math.log(raw_sharpness) - math.log(min_val)
        log_max = math.log(max_val) - math.log(min_val)
        return (log_val / log_max) * 1000.0

    @staticmethod
    def _distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


_detector_instance = None


def get_keypoint_detector() -> ONNXKeypointDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = ONNXKeypointDetector()
    return _detector_instance


def _load_crop(path: str) -> np.ndarray:
    arr = cv2.imread(path)
    if arr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _compare_with_torch(image_path: str, threshold: float = 0.3) -> int:
    if not importlib.util.find_spec("torch"):
        print("[Compare] torch unavailable, skip baseline comparison")
        return 0

    try:
        from core.keypoint_detector import KeypointDetector as TorchKeypointDetector
    except Exception:
        import sys

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from core.keypoint_detector import KeypointDetector as TorchKeypointDetector

    crop = _load_crop(image_path)

    onnx_detector = ONNXKeypointDetector()
    onnx_detector.load_model()
    onnx_result = onnx_detector.detect(crop)

    torch_detector = TorchKeypointDetector()
    torch_detector.load_model()
    torch_result = torch_detector.detect(crop)

    if onnx_result is None or torch_result is None:
        print("[Compare] one side returned None")
        return 1

    def d2(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    coord_err_left = d2(onnx_result.left_eye, torch_result.left_eye)
    coord_err_right = d2(onnx_result.right_eye, torch_result.right_eye)
    coord_err_beak = d2(onnx_result.beak, torch_result.beak)

    vis_err_left = abs(onnx_result.left_eye_vis - torch_result.left_eye_vis)
    vis_err_right = abs(onnx_result.right_eye_vis - torch_result.right_eye_vis)
    vis_err_beak = abs(onnx_result.beak_vis - torch_result.beak_vis)

    sharpness_err = abs(onnx_result.head_sharpness - torch_result.head_sharpness)

    print("\n[Compare Summary]")
    print(f"coord_err_left: {coord_err_left:.8f}")
    print(f"coord_err_right: {coord_err_right:.8f}")
    print(f"coord_err_beak: {coord_err_beak:.8f}")
    print(f"vis_err_left: {vis_err_left:.8f}")
    print(f"vis_err_right: {vis_err_right:.8f}")
    print(f"vis_err_beak: {vis_err_beak:.8f}")
    print(f"sharpness_err: {sharpness_err:.8f}")
    print(
        "class_flags: "
        f"onnx(all_hidden={onnx_result.all_keypoints_hidden}, both_hidden={onnx_result.both_eyes_hidden}, visible_eye={onnx_result.visible_eye}) "
        f"torch(all_hidden={torch_result.all_keypoints_hidden}, both_hidden={torch_result.both_eyes_hidden}, visible_eye={torch_result.visible_eye})"
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX keypoint detector with optional torch comparison")
    parser.add_argument("image", help="bird crop image path")
    parser.add_argument("--no-compare", action="store_true", help="skip torch baseline comparison")
    args = parser.parse_args()

    crop = _load_crop(args.image)
    detector = ONNXKeypointDetector()
    detector.load_model()

    result = detector.detect(crop)
    print("[ONNX Result]")
    print(result)

    if not args.no_compare:
        raise SystemExit(_compare_with_torch(args.image))
