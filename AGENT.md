# ZIP-RC notebook workflow

`notebooks/stages/` is the only source of truth for experiment notebooks. When changing Step 00–08, edit the matching file there; do not edit generated notebooks.

After any stage edit, run from the repository root:

```bash
python3 scripts/merge_stage_notebooks.py
```

This command must regenerate both forms in `notebooks/colab_enterprise/`:

- standalone `00_memory_and_config.ipynb` through `08_graduation_decision.ipynb`;
- `ZIP_RC_experiment_all_in_one.ipynb`.

All generated notebooks must retain the Colab Enterprise runtime contract:

- clone or fast-forward `/content/ZIP-RC-Colab` from the configured public GitHub repository;
- run model and training commands with `/content/mamba/envs/zip/bin/python`;
- run pandas, matplotlib, and display code in the Colab kernel;
- import shared helpers from the cloned repository;
- make Step 01–08 load the configuration written by Step 00.
- preserve progress bars with completed/total work, elapsed time, and estimated remaining time for every long-running experiment loop.

Validate generator changes by running the generator, parsing every generated code cell with `ast.parse`, and checking `ruff` plus `py_compile` for the generator. Keep only the editable stage notebooks and generated Colab Enterprise notebooks; do not recreate root-level or `notebooks/colab/` notebook copies.
