# ZIP-RC notebook workflow

`notebooks/stages/` is the only source of truth for experiment notebooks. When changing Step 00–08, edit the matching file there; do not edit generated notebooks.

After any stage edit, run from the repository root:

```bash
python3 scripts/merge_stage_notebooks.py
```

This command must regenerate both forms in `notebooks/colab_enterprise/`:

- standalone `00_memory_and_config.ipynb` through `08_graduation_decision.ipynb`;
- `ZIP_RC_experiment_all_in_one.ipynb`.
- auxiliary setup notebooks declared by the merge script, including
  `gemini_pro_online_setup.ipynb`; auxiliary notebooks remain standalone and are
  not inserted into the Step 00–08 merged experiment.

Edit auxiliary notebook sources next to the numbered stages in
`notebooks/stages/`, then run the same merge command. Never edit their generated
copies directly. OAuth credentials must remain in the runtime's Antigravity
credential store; notebook sources and artifacts may contain only non-secret
settings such as the selected model slug.

All generated notebooks must retain the Colab Enterprise runtime contract:

- clone or fast-forward `/content/ZIP-RC-Colab` from the configured public GitHub repository;
- run model and training commands with `/content/mamba/envs/zip/bin/python`;
- run pandas, matplotlib, and display code in the Colab kernel;
- import shared helpers from the cloned repository;
- make Step 01–08 load the configuration written by Step 00.
- add `~/.local/bin` to `PATH` so a runtime-installed `agy` remains visible to
  grader subprocesses for the lifetime of that runtime.
- preserve progress bars with completed/total work, elapsed time, and estimated remaining time for every long-running experiment loop.
- preserve atomic JSON progress checkpoints in `artifacts/progress/`, throttled to avoid per-step disk writes, and keep the generated progress-monitor cells working.

Validate generator changes by running the generator, parsing every generated code cell with `ast.parse`, and checking `ruff` plus `py_compile` for the generator. Keep only the editable stage notebooks and generated Colab Enterprise notebooks; do not recreate root-level or `notebooks/colab/` notebook copies.

`scripts/build_stage_notebooks.py` is retained only as a compatibility wrapper and must delegate to `scripts/merge_stage_notebooks.py`; it must not write legacy root-level or `notebooks/colab/` copies.
