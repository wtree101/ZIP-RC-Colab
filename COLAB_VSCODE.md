# ZIP-RC with Colab in VS Code

This folder contains the authors' official ZIP-RC repository plus a self-contained
notebook for a single Colab GPU:

- `notebooks/ziprc_colab_vscode.ipynb`

## Connect VS Code to Colab

1. Open this folder in VS Code.
2. Install the recommended **Google Colab** and **Jupyter** extensions.
3. Open `notebooks/ziprc_colab_vscode.ipynb`.
4. Click **Select Kernel** in the upper-right corner.
5. Choose **Colab**, sign in, then select **New Colab Server** or **Auto Connect**.
6. Prefer an A100 server for the paper's Qwen3-1.7B model. The notebook
   automatically switches to Qwen3-0.6B when GPU memory is below 35 GB.
7. Run one notebook section at a time. Markdown headings can be collapsed in
   VS Code, keeping the long pipeline manageable.

The notebook clones the official repository into `/content/ZIP-RC` on the Colab
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
