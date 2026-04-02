#!/usr/bin/env python3
"""Quick environment and model sanity checks for pe_assessment."""

import sys
from pathlib import Path


def _safe_status(prefix: str, text: str) -> None:
    """Print ASCII-only status line for Windows terminal compatibility."""
    print(f"  [{prefix}] {text}")


def check_imports() -> bool:
    """Validate required and optional imports."""
    print("Checking imports...")

    required_packages = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("sklearn", "scikit-learn"),
        ("scipy", "SciPy"),
        ("tqdm", "tqdm"),
        ("PIL", "Pillow"),
    ]

    optional_packages = [
        ("ultralytics", "Ultralytics (YOLO)"),
        ("matplotlib", "Matplotlib"),
        ("pandas", "Pandas"),
    ]

    all_ok = True

    for module, name in required_packages:
        try:
            __import__(module)
            _safe_status("OK", name)
        except Exception as exc:
            _safe_status("FAIL", f"{name} import failed: {exc}")
            all_ok = False

    print("\nOptional packages:")
    for module, name in optional_packages:
        try:
            __import__(module)
            _safe_status("OK", name)
        except Exception as exc:
            _safe_status("WARN", f"{name} unavailable: {exc}")

    return all_ok


def check_cuda() -> None:
    """Report CUDA status."""
    print("\nChecking CUDA...")

    try:
        import torch

        if torch.cuda.is_available():
            _safe_status("OK", "CUDA available")
            print(f"    device: {torch.cuda.get_device_name(0)}")
            print(
                "    memory: "
                f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        else:
            _safe_status("WARN", "CUDA unavailable, training will use CPU")
    except Exception as exc:
        _safe_status("FAIL", f"CUDA check failed: {exc}")


def check_project_structure() -> bool:
    """Validate required files/directories."""
    print("\nChecking project structure...")

    required_dirs = [
        "utils",
        "data/raw_videos",
        "data/skeletons",
        "data/annotations",
        "checkpoints",
    ]

    required_files = [
        "config.yaml",
        "0_preprocess_videos.py",
        "1_auto_annotate.py",
        "3_train_action.py",
        "4_train_phase.py",
        "5_train_quality.py",
        "6_inference.py",
    ]

    all_ok = True

    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            _safe_status("OK", f"{dir_path}/")
        else:
            _safe_status("WARN", f"{dir_path}/ missing, creating it")
            path.mkdir(parents=True, exist_ok=True)

    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            _safe_status("OK", file_path)
        else:
            _safe_status("FAIL", f"{file_path} missing")
            all_ok = False

    return all_ok


def test_models() -> bool:
    """Basic model construction and forward-pass tests."""
    print("\nTesting model definitions...")

    try:
        sys.path.append(str(Path(__file__).parent))
        from utils.models import QualityNet, STGCNAction, TemporalUNet

        _safe_status("OK", "Model modules imported")

        import torch

        action_model = STGCNAction(num_classes=6)
        phase_model = TemporalUNet(num_phases=5)
        quality_model = QualityNet(num_errors=10)
        _safe_status("OK", "Model instances created")

        dummy_input = torch.randn(2, 60, 17, 9)
        with torch.no_grad():
            action_out = action_model(dummy_input)
            phase_out = phase_model(dummy_input)
            quality_out = quality_model(dummy_input)

        _safe_status("OK", "Forward-pass checks passed")
        print(f"    action output: {action_out.shape}")
        print(f"    phase output: {phase_out.shape}")
        print(f"    quality output: {quality_out['overall'].shape}")
        return True
    except Exception as exc:
        _safe_status("FAIL", f"Model test failed: {exc}")
        return False


def main() -> int:
    print("=" * 60)
    print("Sports Action Assessment - Quick Check")
    print("=" * 60)

    results = []
    results.append(("Imports", check_imports()))
    check_cuda()
    results.append(("Project structure", check_project_structure()))
    results.append(("Model definitions", test_models()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for name, ok in results:
        status = "[OK] pass" if ok else "[FAIL] fail"
        print(f"{name}: {status}")

    all_passed = all(ok for _, ok in results)
    if all_passed:
        print("\n[OK] All checks passed.")
        print("\nQuick start:")
        print("  1. Put videos in data/raw_videos/<action>/")
        print("  2. Run: python 0_preprocess_videos.py")
        print("  3. Run: python 1_auto_annotate.py")
        print("  4. Run: python 3_train_action.py")
        return 0

    print("\n[FAIL] Some checks failed. Review logs above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
