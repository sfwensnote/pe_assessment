# AGENTS.md
Practical guide for coding agents working in `pe_assessment`.

## 1) Project overview
- Domain: sports action assessment from video (6动作: pushup/squat/situp/jump_rope/long_jump/pullup).
- Pipeline: preprocess keypoints -> auto label -> review -> train -> infer -> export.
- Main backend stack: Python + PyTorch + FastAPI.
- Main frontend stack: React + TypeScript + Vite (`web/`).
- Global runtime configuration is centralized in `config.yaml`.

## 2) High-value paths
- `0_preprocess_videos.py`: video -> skeleton json extraction.
- `1_auto_annotate.py`: rule-based phase/quality annotation.
- `2_review_annotations.py`: manual review tool.
- `3_train_action.py`: action classification training.
- `4_train_phase.py`: phase segmentation training.
- `5_train_quality.py`: quality scoring/error detection training.
- `6_inference.py`: end-to-end inference on video.
- `7_export_model.py`: export for deployment.
- `8_ingest_pipeline.py`: auto ingest/orchestration pipeline.
- `8_ingest_monitor.py`: ingest progress monitor.
- `9_tag_and_cleanup_videos.py`: tag + optional cleanup.
- `app/main.py`: FastAPI realtime/inference APIs.
- `utils/`: skeleton/model/augmentation/metrics core logic.
- `web/`: React frontend.
- `ultralytics/`: vendored upstream code; avoid edits unless task explicitly requires.

## 3) Rule sources checked
- `.cursor/rules/`: not found.
- `.cursorrules`: not found.
- `.github/copilot-instructions.md`: not found.
- No Cursor/Copilot repo rules exist at the moment.
- If any are added later, treat them as higher-priority agent instructions.

## 4) Environment setup
- Recommended virtual env:
  - `python -m venv .venv`
  - Windows: `.venv\\Scripts\\activate`
  - Unix: `source .venv/bin/activate`
- Install dependencies:
  - `python -m pip install -U pip`
  - `python -m pip install -r requirements.txt`
- Quick sanity check:
  - `python quick_test.py`

## 5) Build / run commands

### Python pipeline
- Preprocess all actions: `python 0_preprocess_videos.py`
- Preprocess single action: `python 0_preprocess_videos.py --action pushup`
- Auto annotate: `python 1_auto_annotate.py`
- Review annotation: `python 2_review_annotations.py --action pushup`
- Train action model: `python 3_train_action.py --epochs 100 --batch_size 64`
- Train phase models: `python 4_train_phase.py --epochs 80`
- Train single phase model: `python 4_train_phase.py --action pushup`
- Train quality model: `python 5_train_quality.py --epochs 60 --batch_size 32`
- Inference text/json: `python 6_inference.py --video test.mp4 --format json`
- Export model: `python 7_export_model.py --format onnx`

### Ingest/ops scripts
- Full ingest pipeline: `python 8_ingest_pipeline.py`
- Ingest selected actions: `python 8_ingest_pipeline.py --actions pushup,squat`
- Monitor once: `python 8_ingest_monitor.py`
- Monitor watch mode: `python 8_ingest_monitor.py --watch`
- Tag only: `python 9_tag_and_cleanup_videos.py`
- Tag + cleanup: `python 9_tag_and_cleanup_videos.py --cleanup`

### Backend / frontend
- Backend dev server: `uvicorn app.main:app --reload --port 8001`
- Frontend dev:
  - `cd web`
  - `npm install`
  - `npm run dev`
- Frontend build:
  - `cd web && npm run build`

## 6) Lint / format / test commands

### Python lint & format
- Format changed Python files: `python -m black *.py utils/*.py app/**/*.py`
- Format check: `python -m black --check *.py utils/*.py app/**/*.py`
- Lint core Python: `python -m flake8 *.py utils/*.py app/**/*.py`
- Do not run lint/format across `ultralytics/` unless task explicitly needs it.

### Tests
- Current status: no dedicated `tests/` directory in this repo.
- Primary validation command: `python quick_test.py`
- If pytest tests are added later, run all: `python -m pytest -q`

### Run a single test (important)
- Single file: `python -m pytest -q path/to/test_file.py`
- Single function: `python -m pytest -q path/to/test_file.py::test_fn`
- Single class method: `python -m pytest -q path/to/test_file.py::TestClass::test_method`
- Keyword filter: `python -m pytest -q -k "keyword"`
- Stop fast on first failure: `python -m pytest -q -x`

## 7) Code style guidelines

### Imports
- Order imports as: stdlib -> third-party -> local.
- Separate groups with one blank line.
- Prefer explicit imports; avoid wildcard imports.
- Keep `utils` imports absolute and stable.
- Preserve existing script bootstrap patterns when present (e.g., `sys.path.append(...)`).

### Formatting
- 4-space indentation, no tabs.
- Write Black-friendly code and avoid style-only churn.
- Keep functions focused; extract helper functions for repeated logic.
- Keep CLI scripts `argparse`-driven and backwards compatible.

### Types
- Add type hints to public/non-trivial functions.
- Keep array/tensor shape assumptions explicit in docstrings (`[B, T, N, C]`, `[T, 17, 9]`).
- Use consistent typing style (`Dict/List/Tuple/Optional` or builtin generics, not mixed randomly).

### Naming conventions
- `snake_case`: variables/functions/files.
- `PascalCase`: classes.
- `UPPER_SNAKE_CASE`: module-level constants.
- Preserve action keys (`pushup`, `jump_rope`, etc.) and numbered script names (`0_...` to `9_...`).

### Error handling
- Fail fast on critical init issues (missing config/model/dependency).
- Fail soft for batch processing loops (log and continue per file/sample).
- Avoid bare `except:`; catch specific exceptions where practical.
- Error output should include useful context: action type, path, and step name.

### Logging and outputs
- Existing project style uses `print` + `tqdm`; keep consistency unless refactor is requested.
- Keep progress logs concise; avoid noisy per-frame or per-iteration spam.
- For JSON containing Chinese text, keep `ensure_ascii=False` where needed.

### Config/data contracts
- Read runtime settings from `config.yaml` rather than hardcoding.
- Respect `CONFIG['paths']` for file locations.
- Preserve expected schemas for skeleton/annotation/inference JSON outputs.
- Keep checkpoint filenames compatible with `6_inference.py` and backend runtime loaders.

### Backend/frontend notes
- Backend APIs should keep response keys stable for `web/src` consumers.
- For frontend changes, keep TypeScript types aligned with `web/src/types.ts`.
- Preserve CORS and admin-token behavior unless task explicitly changes security model.

## 8) Agent workflow checklist
- Read relevant script/module before editing; follow local conventions.
- Make minimal, targeted changes.
- Run the lightest meaningful validation for touched code.
- State clearly what was/was not validated if full training/inference is too expensive.
- Do not commit generated artifacts (`.pth`, `.onnx`, videos, ingest outputs, caches).

## 9) Safety constraints
- Avoid absolute machine-specific paths in committed code.
- Keep large binary/model/data files out of git (respect `.gitignore`).
- Do not modify vendored `ultralytics/` unless explicitly requested.
- Prefer additive, low-risk edits over broad rewrites in training/inference paths.
