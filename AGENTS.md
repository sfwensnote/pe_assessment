# AGENTS Guide for `pe_assessment`
This file is for coding agents working in this repository.
Follow it when planning, editing, testing, and validating changes.

## 1) Scope and priorities
- This is a Python project for sports action assessment from video.
- Main workflow: preprocess -> annotate -> review -> train -> infer -> export.
- Top-level numbered scripts (`0_` to `7_`) are the primary entrypoints.
- `utils/` contains core feature extraction, models, augmentation, and metrics.
- `ultralytics/` is vendored third-party code; treat it as external unless a task requires touching it.

## 2) Rule sources checked
- Checked `.cursor/rules/`: not present.
- Checked `.cursorrules`: not present.
- Checked `.github/copilot-instructions.md`: not present.
- No Cursor/Copilot rule files are currently available in this repo.
- If these files are added later, treat them as high-priority project rules.

## 3) Repository map (high value paths)
- `config.yaml`: global config for actions, training, thresholds, and paths.
- `0_preprocess_videos.py`: extract skeleton keypoints from raw videos.
- `1_auto_annotate.py`: rule-based phase and quality auto-labeling.
- `2_review_annotations.py`: CLI/manual review tool.
- `3_train_action.py`: action classification training.
- `4_train_phase.py`: phase segmentation training.
- `5_train_quality.py`: quality scoring/error detection training.
- `6_inference.py`: end-to-end video assessment.
- `7_export_model.py`: export models to ONNX/TorchScript and deploy package.
- `quick_test.py`: environment + model smoke check.
- `run_all.sh`: one-shot full pipeline runner.
- `requirements.txt`: pinned dependencies and dev tools.

## 4) Environment and setup commands
- Create env (recommended): `python -m venv .venv && source .venv/bin/activate`
- Install deps: `python -m pip install -r requirements.txt`
- Smoke-check environment: `python quick_test.py`
- Verify key imports quickly: `python -c "import torch, cv2, yaml, numpy"`

## 5) Build/train/inference commands
- Full pipeline (interactive review prompt included): `bash run_all.sh`
- Preprocess all actions: `python 0_preprocess_videos.py`
- Preprocess one action: `python 0_preprocess_videos.py --action pushup`
- Auto-annotate all: `python 1_auto_annotate.py`
- Auto-annotate one action: `python 1_auto_annotate.py --action pushup`
- Review annotations: `python 2_review_annotations.py --action pushup`
- Train action model: `python 3_train_action.py --epochs 100 --batch_size 64`
- Train phase model (all actions): `python 4_train_phase.py --epochs 80`
- Train phase model (single action): `python 4_train_phase.py --action pushup`
- Train quality model: `python 5_train_quality.py --epochs 60 --batch_size 32`
- Inference (auto action detection): `python 6_inference.py --video test.mp4`
- Inference (force action): `python 6_inference.py --video test.mp4 --action pushup`
- Inference JSON output: `python 6_inference.py --video test.mp4 --format json --output result.json`
- Export deploy models (ONNX): `python 7_export_model.py --format onnx`
- Export deploy models (TorchScript): `python 7_export_model.py --format torchscript`

## 6) Lint and formatting commands
- Formatter: `black` (listed in `requirements.txt`).
- Linter: `flake8` (listed in `requirements.txt`).
- Format core files only: `python -m black *.py utils/*.py`
- Format check only: `python -m black --check *.py utils/*.py`
- Lint core files only: `python -m flake8 *.py utils/*.py --max-line-length=88`
- Do not run repo-wide lint/format over `ultralytics/` unless task explicitly requires it.

## 7) Test commands (current state + usage)
- `pytest` is installed, but there is no dedicated `tests/` directory yet.
- Practical validation is currently script-level smoke/integration checks.
- Primary smoke test: `python quick_test.py`
- Pipeline-level validation (expensive): run relevant numbered script with small inputs.

### Single-test execution (important)
- Single test file: `python -m pytest tests/test_file.py -q`
- Single test function: `python -m pytest tests/test_file.py::test_case_name -q`
- Keyword-filtered run: `python -m pytest -k "keyword" -q`
- Stop on first failure: `python -m pytest -x -q`
- In this repo today, `python quick_test.py` is the nearest single-target check.

## 8) Code style guidelines inferred from codebase

### Imports
- Group imports in this order: standard library, third-party, local project.
- Separate groups with one blank line.
- Prefer explicit imports; avoid wildcard imports.
- Keep one import per line, except tightly related typing imports.
- Top-level runnable scripts often use `sys.path.append(str(Path(__file__).parent))`; preserve this pattern.

### Formatting and structure
- Use 4-space indentation.
- Keep functions/classes focused; extract helpers for repeated logic.
- Add a module docstring at file top with purpose and usage.
- Preserve existing CLI style: `argparse` with clear `--help` text.
- Prefer `Path` from `pathlib` over manual string path joins.
- Use `if __name__ == '__main__':` in executable scripts.

### Types and data contracts
- Add type hints for public functions/methods and non-trivial returns.
- Use `Dict`, `List`, `Tuple`, `Optional` (or built-in generics consistently).
- Keep tensor/array shape assumptions explicit, e.g. `[B, T, N, C]`.
- Use `np.ndarray` and `torch.Tensor` annotations where appropriate.
- Return structured dictionaries with stable keys for model outputs/reports.

### Naming conventions
- `snake_case` for functions, variables, and local helpers.
- `PascalCase` for classes.
- `UPPER_SNAKE_CASE` for constants (`CONFIG_PATH`, `CONFIG`, `ERROR_TYPES`).
- Keep action IDs lowercase snake_case (`pushup`, `jump_rope`, etc.).
- Keep script filenames numeric and ordered (`0_...` to `7_...`).

### Error handling and logging
- Prefer fail-soft behavior for batch loops: catch per-sample errors, continue, print concise context.
- Prefer fail-fast behavior for critical setup failures (missing model/dependency).
- Current style uses `print` and `tqdm` rather than a logging framework.
- If catching broad `Exception`, include actionable context (file/action/video id).
- Avoid silent failures; return explicit empty result or clear warning.

### Configuration and constants
- Centralize tunables and paths in `config.yaml`; avoid hardcoded values when config exists.
- Load config once near module top in script-style files.
- Reuse configured paths from `CONFIG['paths']`.
- Keep action definitions synchronized with `config.yaml` when adding/removing actions.

### Data/model pipeline expectations
- Skeleton feature contract is `[T, 17, 9]` after preprocessing.
- Training scripts expect JSON annotation + skeleton pairs by action directory.
- Preserve checkpoint key compatibility used by inference/export scripts.
- Preserve device fallback behavior: prefer CUDA, fallback to CPU.
- Avoid breaking output JSON schema used by downstream tools.

### Internationalization/content style
- User-facing messages are mostly Chinese; keep wording consistent in touched files.
- Keep code identifiers in English.
- Use `ensure_ascii=False` only where non-ASCII JSON output is intended.

## 9) Agent workflow checklist
- Read `README.md` and target script(s) before edits.
- Make minimal, focused changes aligned with existing patterns.
- Run formatter/linter only on touched core files.
- Run the lightest meaningful validation command for the change.
- If training/inference is too expensive, state what was not run and why.
- Do not modify vendored `ultralytics/` unless explicitly required.

## 10) Safety notes for automated edits
- Do not commit large generated artifacts (`.pth`, `.onnx`, data outputs).
- Respect `.gitignore` entries for checkpoints, deploy models, and raw data.
- Avoid introducing absolute paths; use config-relative or repo-relative paths.
- Keep CLI backward compatible where possible; add new args as optional.
- Prefer additive changes over risky rewrites in training/inference paths.
