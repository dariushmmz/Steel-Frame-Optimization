# Steel Frame Optimization

Research software for two-stage LRFD and performance-based optimization of three- and nine-story
steel moment frames with a deterministic, checkpointable discrete genetic algorithm.

The three-story implementation and validation are complete. Phase 12 is executing the frozen
nine-story experiments. Results are research outputs and are not professional engineering
certification.

## Environment

- Python 3.12.6 exactly
- Dependencies frozen by `uv.lock`
- OpenSeesPy structural analysis

```powershell
$env:UV_PYTHON_DOWNLOADS = "never"
uv sync --frozen --python C:\path\to\python.exe
uv run --frozen --no-sync python --version
```

## Required external section data

The processed 273-section project input must be placed at:

`data/processed/aisc_v14_1_w_shapes.json`

Expected SHA-256:

`13f75d47cca20b3028c377ba46466fca5a0833c1ec578fb40c9b099d30fdb8cf`

The source release does not redistribute this input because rights to redistribute the originating
AISC workbook have not been established. GitHub Actions reconstructs the same hash-checked input
from encrypted repository secrets; it is not stored in the public repository.

## Verification

```powershell
uv run --frozen --no-sync ruff format --check .
uv run --frozen --no-sync ruff check .
uv run --frozen --no-sync mypy --strict
uv run --frozen --no-sync pytest -q
```

## Layout

- `src/steel_frame_optimization/`: structural analysis, optimization, and CLI code
- `configs/`: frozen frame, analysis, objective, solver, and campaign policies
- `deployment/`: checkpoint-aware local and cloud runners
- `tests/`: unit, analytical, integration, and regression checks
- `data/processed/`: externally supplied processed section input

The bounded Phase 12 GitHub workflow uses only a standard public-repository Linux runner, advances
at most 40 epochs per job, and preserves exact campaign identities and request accounting.
