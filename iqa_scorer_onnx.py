#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IQA scorer (ONNX backend).

This module mirrors iqa_scorer.py public API while replacing PyTorch
TOPIQ inference with ONNX Runtime model:
- models/cfanet_iaa_ava_res50.onnx
"""

import argparse
import importlib.util
import os
import sys
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image


def _ensure_utf8_stdout() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _get_topiq_onnx_path() -> str:
    model_name = "cfanet_iaa_ava_res50.onnx"

    search_paths = []
    if hasattr(sys, "_MEIPASS"):
        search_paths.append(os.path.join(sys._MEIPASS, "models", model_name))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(base_dir, "models", model_name))
    search_paths.append(os.path.join(base_dir, model_name))

    for path in search_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"TOPIQ ONNX model not found. expected models/{model_name}. searched: {search_paths}"
    )


class IQAScorer:
    """IQA scorer using ONNX TOPIQ model."""

    def __init__(self, device: str = "mps"):
        self.requested_device = device
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_mos_name: Optional[str] = None
        self._output_dist_name: Optional[str] = None
        self._providers = self._resolve_providers(device)

    @staticmethod
    def _resolve_providers(preferred: str):
        available = set(ort.get_available_providers())

        providers = []
        if preferred in ("cuda", "gpu") and "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

        # MPS execution provider is generally unavailable in most ORT wheels.
        if preferred == "mps":
            if "CoreMLExecutionProvider" in available:
                providers.append("CoreMLExecutionProvider")
            elif "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")

        if "CUDAExecutionProvider" in available and "CUDAExecutionProvider" not in providers:
            providers.append("CUDAExecutionProvider")

        providers.append("CPUExecutionProvider")
        return providers

    def _load_topiq(self):
        if self._session is not None:
            return self._session

        model_path = _get_topiq_onnx_path()
        try:
            self._session = ort.InferenceSession(model_path, providers=self._providers)
        except Exception as exc:
            # Fallback to CPU-only if GPU EP init fails.
            print(f"[IQA ONNX] CUDA init failed, fallback to CPU: {exc}")
            self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"[IQA ONNX] session provider: {self._session.get_providers()}")

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or len(outputs) < 1:
            raise RuntimeError("Unexpected TOPIQ ONNX IO signature")

        self._input_name = inputs[0].name
        self._output_mos_name = outputs[0].name
        self._output_dist_name = outputs[1].name if len(outputs) > 1 else None

        return self._session

    @staticmethod
    def _preprocess_pil(img: Image.Image) -> np.ndarray:
        img = img.convert("RGB").resize((384, 384), Image.LANCZOS)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = np.expand_dims(arr, axis=0)
        return arr.astype(np.float32)

    def calculate_nima(self, image_path: str) -> Optional[float]:
        return self.calculate_aesthetic(image_path)

    def calculate_aesthetic(self, image_path: str) -> Optional[float]:
        if not os.path.exists(image_path):
            print(f"[IQA ONNX] image not found: {image_path}")
            return None

        try:
            session = self._load_topiq()
            img = Image.open(image_path)
            batch = self._preprocess_pil(img)
            outputs = session.run(None, {self._input_name: batch})
            mos = float(np.asarray(outputs[0]).reshape(-1)[0])
            return max(1.0, min(10.0, mos))
        except Exception as exc:
            print(f"[IQA ONNX] calculate_aesthetic failed: {exc}")
            return None

    def calculate_from_array(self, img_bgr: np.ndarray) -> Optional[float]:
        if img_bgr is None or img_bgr.size == 0:
            return None

        try:
            import cv2

            session = self._load_topiq()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            batch = self._preprocess_pil(pil)
            outputs = session.run(None, {self._input_name: batch})
            mos = float(np.asarray(outputs[0]).reshape(-1)[0])
            return max(1.0, min(10.0, mos))
        except Exception as exc:
            print(f"[IQA ONNX] calculate_from_array failed: {exc}")
            return None

    def calculate_brisque(self, image_input) -> Optional[float]:
        # Deprecated in project, kept for API compatibility.
        return None

    def calculate_both(self, full_image_path: str, crop_image) -> Tuple[Optional[float], Optional[float]]:
        aesthetic_score = self.calculate_aesthetic(full_image_path)
        return aesthetic_score, None


_iqa_scorer_instance = None


def get_iqa_scorer(device: str = "mps") -> IQAScorer:
    global _iqa_scorer_instance
    if _iqa_scorer_instance is None:
        _iqa_scorer_instance = IQAScorer(device=device)
    return _iqa_scorer_instance


def calculate_nima(image_path: str) -> Optional[float]:
    scorer = get_iqa_scorer()
    return scorer.calculate_aesthetic(image_path)


def calculate_brisque(image_input) -> Optional[float]:
    return None


def _compare_with_torch(image_path: str, device: str) -> int:
    _ensure_utf8_stdout()

    if not importlib.util.find_spec("torch"):
        print("[Compare] torch unavailable, skip baseline comparison")
        return 0

    try:
        from iqa_scorer import IQAScorer as TorchIQAScorer
    except Exception as exc:
        print(f"[Compare] failed to import torch IQA scorer: {exc}")
        return 0

    onnx_scorer = IQAScorer(device=device)
    onnx_score = onnx_scorer.calculate_aesthetic(image_path)

    torch_scorer = TorchIQAScorer(device=device)
    torch_score = torch_scorer.calculate_aesthetic(image_path)

    print("\n[Compare Result]")
    print(f"image: {image_path}")
    print(f"onnx_score: {onnx_score}")
    print(f"torch_score: {torch_score}")

    if onnx_score is None or torch_score is None:
        print("abs_error: N/A (one side failed)")
        return 1

    abs_err = abs(onnx_score - torch_score)
    rel_err = abs_err / max(abs(torch_score), 1e-8)
    print(f"abs_error: {abs_err:.8f}")
    print(f"rel_error: {rel_err:.8f}")
    return 0


if __name__ == "__main__":
    _ensure_utf8_stdout()

    parser = argparse.ArgumentParser(description="IQA scorer ONNX with optional torch comparison")
    parser.add_argument("--image", default="img/_Z9W0960.jpg", help="test image path")
    parser.add_argument("--device", default="mps", help="preferred device: mps/cuda/cpu")
    parser.add_argument("--no-compare", action="store_true", help="skip torch baseline comparison")
    args = parser.parse_args()

    scorer = IQAScorer(device=args.device)
    score = scorer.calculate_aesthetic(args.image)
    print("[ONNX Score]")
    print(f"image: {args.image}")
    print(f"score: {score}")

    if not args.no_compare:
        raise SystemExit(_compare_with_torch(args.image, args.device))
