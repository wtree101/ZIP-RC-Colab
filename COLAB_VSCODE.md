# ZIP-RC with Colab in VS Code

This repository provides generated Colab Enterprise notebooks for a single GPU:

- `notebooks/colab_enterprise/ZIP_RC_experiment_all_in_one.ipynb` for one-file execution;
- `notebooks/colab_enterprise/00_memory_and_config.ipynb` through
  `08_graduation_decision.ipynb` for stage-by-stage execution.

## Connect VS Code to Colab

1. Open this folder in VS Code.
2. Install the recommended **Google Colab** and **Jupyter** extensions.
3. Open one of the generated notebooks in `notebooks/colab_enterprise/`.
4. Click **Select Kernel** in the upper-right corner.
5. Choose **Colab**, sign in, then select **New Colab Server** or **Auto Connect**.
6. Use an RTX 3090 or Colab L4 for the default Qwen3-0.6B experiment.
7. Run one notebook section at a time. Markdown headings can be collapsed in
   VS Code, keeping the long pipeline manageable.

The notebook clones this repository into `/content/ZIP-RC-Colab` on the Colab
VM. This is necessary because a local VS Code workspace is not automatically the
remote runtime's filesystem.

## Persist results

From the VS Code command palette (`Cmd+Shift+P` on macOS), run:

`Colab: Mount Google Drive to Server...`

Then run the notebook's final **Save results** section. Without Drive, Colab's
ephemeral `/content` directory is deleted when the server is released.

## Expected resources

| Selected model | Typical GPU | VRAM | Smoke-test time |
|---|---|---:|---:|
| Qwen3-0.6B | T4 or L4 | 16-24 GB | 15-45 minutes |
| Qwen3-1.7B | A100 | 40 GB+ | 30-90 minutes |

Times include downloads and are estimates. The final KL-regularized stage is the
largest memory user because it loads a trainable model and a frozen reference
model.

## Scope

This validates rollout generation, grading, both training stages, and offline
scoring. The upstream repository does not currently include the adaptive
meta-action sampler needed to reproduce the paper's main Pareto-frontier results.
