import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

SPEC_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_PATH = SPEC_DIR.resolve()
sys.path.insert(0, str(BASE_PATH))

from constants import APP_VERSION

EXPECTED_CONDA_ENV = "superpicky_gpu"
EXPECTED_ONNXRUNTIME_VERSION = "1.17.0"
REQUIRED_ONNXRUNTIME_BINARIES = {
    "onnxruntime_providers_cuda.dll",
    "onnxruntime_providers_shared.dll",
}
CUDA_RUNTIME_PATTERNS = (
    "cudart*.dll",
    "cublas*.dll",
    "cudnn*.dll",
    "cufft*.dll",
    "curand*.dll",
    "cusolver*.dll",
    "cusparse*.dll",
    "nvrtc*.dll",
    "nvToolsExt*.dll",
)
CUDA_RUNTIME_EXACT_NAMES = {
    "zlibwapi.dll",
}
CUDA_RUNTIME_SUPPORT_DLLS = {
    "asmjit.dll",
    "c10.dll",
    "c10_cuda.dll",
    "caffe2_nvrtc.dll",
    "cupti64_2022.3.0.dll",
    "libiomp5md.dll",
    "libiompstubs5md.dll",
    "torch_global_deps.dll",
}
CUDA_RUNTIME_EXCLUDED_PREFIXES = (
    "torch",
    "fbgemm",
    "shm",
    "uv",
)
GPU_RUNTIME_DIR_NAME = "gpu_runtime"
TORCH_LIB_PREFIX = f"torch{os.sep}lib{os.sep}"


def _log(message: str):
    print(f"[superpicky cuda spec] {message}")


def _require_expected_environment():
    executable = Path(sys.executable).resolve()
    env_name = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    env_path = os.environ.get("CONDA_PREFIX", "").strip()

    _log(f"python executable: {executable}")
    if env_name:
        _log(f"CONDA_DEFAULT_ENV={env_name}")
    if env_path:
        _log(f"CONDA_PREFIX={env_path}")

    env_markers = {part.lower() for part in executable.parts}
    if env_name == EXPECTED_CONDA_ENV:
        return
    if EXPECTED_CONDA_ENV.lower() in env_markers:
        return
    if env_path and Path(env_path).resolve().name.lower() == EXPECTED_CONDA_ENV.lower():
        return

    raise RuntimeError(
        f"This CUDA spec must run inside the '{EXPECTED_CONDA_ENV}' environment. "
        f"Current python: {executable}"
    )


def _import_validated_onnxruntime():
    import onnxruntime as ort

    ort_file = Path(ort.__file__).resolve()
    ort_version = getattr(ort, "__version__", "<unknown>")
    _log(f"onnxruntime version: {ort_version}")
    _log(f"onnxruntime module: {ort_file}")

    if ort_version != EXPECTED_ONNXRUNTIME_VERSION:
        raise RuntimeError(
            f"Expected onnxruntime-gpu {EXPECTED_ONNXRUNTIME_VERSION}, got {ort_version}. "
            f"Python: {Path(sys.executable).resolve()}, onnxruntime: {ort_file}"
        )

    return ort


def _require_existing_files(base_dir: Path, required_names: set[str], label: str):
    missing = sorted(name for name in required_names if not (base_dir / name).exists())
    if missing:
        raise RuntimeError(f"Missing required {label} files in {base_dir}: {', '.join(missing)}")


_require_expected_environment()
ORT = _import_validated_onnxruntime()
ORT_CAPI_DIR = Path(ORT.__file__).resolve().parent / "capi"
_require_existing_files(ORT_CAPI_DIR, REQUIRED_ONNXRUNTIME_BINARIES, "onnxruntime")


def _existing_tree(src: Path, dest: str):
    if src.exists():
        return [(str(src), dest)]
    return []


def _safe_collect_data_files(package_name: str):
    try:
        return collect_data_files(package_name)
    except Exception:
        return []


def _safe_copy_metadata(dist_name: str):
    try:
        return copy_metadata(dist_name)
    except Exception:
        return []


def _collect_onnxruntime_binaries():
    binaries = []
    seen = set()

    def add_entries(entries):
        for entry in entries:
            key = tuple(entry[:2])
            if key in seen:
                continue
            if os.path.basename(entry[0]).lower() == "onnxruntime_providers_tensorrt.dll":
                continue
            seen.add(key)
            binaries.append(entry)

    add_entries(collect_dynamic_libs("onnxruntime"))

    _log(f"collecting onnxruntime binaries from: {ORT_CAPI_DIR}")
    for pattern in ("*.dll", "*.pyd"):
        for path in sorted(ORT_CAPI_DIR.glob(pattern)):
            add_entries([(str(path), "onnxruntime/capi")])

    _require_existing_files(ORT_CAPI_DIR, REQUIRED_ONNXRUNTIME_BINARIES, "onnxruntime")

    return binaries


def _collect_gpu_runtime_binaries():
    binaries = []
    seen = set()

    def add_binary(path: Path):
        key = os.path.normcase(str(path))
        if key in seen or not path.exists():
            return
        seen.add(key)
        binaries.append((str(path), "gpu_runtime"))

    import torch

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    _log(f"torch lib directory: {torch_lib}")
    if not torch_lib.exists():
        raise RuntimeError(f"torch lib directory does not exist: {torch_lib}")

    for pattern in CUDA_RUNTIME_PATTERNS:
        for path in sorted(torch_lib.glob(pattern)):
            stem = path.stem.lower()
            if stem.startswith(CUDA_RUNTIME_EXCLUDED_PREFIXES):
                continue
            add_binary(path)

    for name in sorted(CUDA_RUNTIME_EXACT_NAMES):
        add_binary(torch_lib / name)

    for name in sorted(CUDA_RUNTIME_SUPPORT_DLLS):
        add_binary(torch_lib / name)

    if not binaries:
        raise RuntimeError(f"No CUDA runtime DLLs were collected from {torch_lib}")

    collected_names = sorted(Path(src).name for src, _ in binaries)
    _log(f"collected {len(collected_names)} GPU runtime DLLs")
    _log(f"gpu runtime DLL summary: {', '.join(collected_names)}")

    return binaries


def _prune_duplicate_cuda_binaries(binaries):
    gpu_runtime_names = set()
    pruned = []
    removed = []

    for entry in binaries:
        dest_name, src_name, typecode = entry
        normalized_dest = dest_name.replace("/", os.sep).replace("\\", os.sep)
        basename = os.path.basename(dest_name).lower()
        if normalized_dest.startswith(GPU_RUNTIME_DIR_NAME + os.sep):
            gpu_runtime_names.add(basename)

    for entry in binaries:
        dest_name, src_name, typecode = entry
        normalized_dest = dest_name.replace("/", os.sep).replace("\\", os.sep)
        basename = os.path.basename(dest_name).lower()
        if normalized_dest.startswith(TORCH_LIB_PREFIX) and basename in gpu_runtime_names:
            removed.append(dest_name)
            continue
        pruned.append(entry)

    if removed:
        _log(f"removed {len(removed)} duplicate torch/lib CUDA binaries")
        _log(f"removed duplicates: {', '.join(sorted(removed))}")

    return pruned

PACKAGE_DATA_NAMES = [
    "imageio",
    "rawpy",
    "pillow_heif",
    "onnxruntime",
]

all_datas = []
all_datas.extend(_existing_tree(BASE_PATH / "models", "models"))
all_datas.extend(_existing_tree(BASE_PATH / "exiftools_win", "exiftools_win"))
all_datas.extend(_existing_tree(BASE_PATH / "img", "img"))
all_datas.extend(_existing_tree(BASE_PATH / "locales", "locales"))
all_datas.extend(_existing_tree(BASE_PATH / "locales" / "en.lproj", "en.lproj"))
all_datas.extend(_existing_tree(BASE_PATH / "locales" / "zh-Hans.lproj", "zh-Hans.lproj"))
all_datas.extend(_existing_tree(BASE_PATH / "birdid" / "data", "birdid/data"))
all_datas.extend(_existing_tree(BASE_PATH / "SuperBirdIDPlugin.lrplugin", "SuperBirdIDPlugin.lrplugin"))

for package_name in PACKAGE_DATA_NAMES:
    all_datas.extend(_safe_collect_data_files(package_name))

for dist_name in [
    "imageio",
    "rawpy",
    "pillow_heif",
    "onnxruntime",
    "onnxruntime-gpu",
]:
    all_datas.extend(_safe_copy_metadata(dist_name))

hiddenimports = [
    "onnxruntime",
    "onnxruntime.capi",
    "onnxruntime.capi._ld_preload",
    "onnxruntime.capi._pybind_state",
    "onnxruntime.capi.onnxruntime_inference_collection",
    "onnxruntime.capi.onnxruntime_pybind11_state",
    "onnxruntime.capi.onnxruntime_validation",
    "PIL",
    "cv2",
    "numpy",
    "yaml",
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends.backend_agg',
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "imageio",
    "rawpy",
    "imagehash",
    "pywt",
    "pillow_heif",
    "core",
    "core.burst_detector",
    "core.config_manager",
    "core.exposure_detector",
    "core.file_manager",
    "core.flight_detector_onnx",
    "core.focus_point_detector",
    "core.keypoint_detector_onnx",
    "core.photo_processor",
    "core.rating_engine",
    "core.stats_formatter",
    "multiprocessing",
    "multiprocessing.spawn",
    "tools.update_checker",
    "packaging",
    "packaging.version",
    "birdid",
    "birdid.bird_identifier_onnx",
    "birdid.osea_classifier_onnx",
    "birdid.ebird_country_filter",
    "iqa_scorer_onnx",
    "birdid_server",
    "server_manager",
    "flask",
    "flask.json",
    "cryptography",
    "cryptography.fernet",
]

a = Analysis(
    ["main.py"],
    pathex=[str(BASE_PATH)],
    binaries=_collect_onnxruntime_binaries() + _collect_gpu_runtime_binaries(),
    datas=all_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        hook_name
        for hook_name, hook_path in [
            ("pyi_rth_cv2.py", BASE_PATH / "pyi_rth_cv2.py"),
            ("pyi_rth_onnx_cuda.py", BASE_PATH / "pyi_rth_onnx_cuda.py"),
        ]
        if hook_path.exists()
    ],
    excludes=[
        "PyQt5",
        "PyQt6",
        "tkinter",
        "ultralytics",
        "matplotlib",
        "torch",
        "torchvision",
        "torchaudio",
        "timm",
    ],
    noarchive=False,
    optimize=0,
)

a.binaries = _prune_duplicate_cuda_binaries(a.binaries)

pyz = PYZ(a.pure)

icon_ico = BASE_PATH / "img" / "icon.ico"
icon_icns = BASE_PATH / "img" / "SuperPicky-V0.02.icns"
exe_icon = None
if sys.platform == "win32" and icon_ico.exists():
    exe_icon = str(icon_ico)
elif icon_icns.exists():
    exe_icon = str(icon_icns)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SuperPicky",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=exe_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SuperPicky",
)

app = BUNDLE(
    coll,
    name="SuperPicky.app",
    icon=str(icon_icns) if icon_icns.exists() else None,
    bundle_identifier="com.jamesphotography.superpicky",
    info_plist={
        "CFBundleName": "SuperPicky",
        "CFBundleDisplayName": "SuperPicky",
        "CFBundleVersion": APP_VERSION,
        "CFBundleShortVersionString": APP_VERSION,
        "NSHighResolutionCapable": True,
        "NSAppleEventsUsageDescription": "SuperPicky needs AppleEvents permission to communicate with other apps.",
        "NSAppleScriptEnabled": False,
    },
)
