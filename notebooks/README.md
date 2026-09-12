# ZIP-RC stage notebooks

These notebooks implement the 0.6B experiment in `reference/plan.md`. Run them in numeric order; each notebook writes a machine-readable gate report to `artifacts/stage_reports/`.

| Notebook | Stage | Main visual checks |
|---|---|---|
| `00_memory_and_config.ipynb` | Memory, disk, GPU, shared config | Stacked RAM/disk/VRAM usage |
| `01_pilot_generation.ipynb` | 200-prompt pilot generation | Length histogram, completion/truncation rates |
| `02_pilot_quality.ipynb` | Pilot grading and go/no-go | Correctness balance, finished × correct, length by class |
| `03_training_data.ipynb` | 2,000-prompt data generation and grading | Length, label balance, per-prompt difficulty |
| `04_prompt_split.ipynb` | Prompt-level train/validation/test split | Split sizes, distribution stability, leakage gates |
| `05_predictor_training.ipynb` | Intermediate scoring and final KL training | Stage losses, KL, learning rate, value separation |
| `06_predictor_evaluation.ipynb` | Held-out predictor evaluation | AUROC/AUPRC, incorrect recall, calibration, remaining-length MAE |
| `07_controller_comparison.ipynb` | Fresh-rollout controller proxy | Accuracy versus average generated tokens |
| `08_graduation_decision.ipynb` | 0.6B graduation decision | Operational/scientific gate summary |

The default profile targets one RTX 3090 or Colab L4 with Qwen3-0.6B, BF16, a 4,096-token context, and a 2,048-token output cap. Edit the configuration cell in notebook 00 before running later stages.

For Colab, use the generated notebooks in `notebooks/colab/`. They fix the remote repository at `/content/ZIP-RC-Colab` and route training commands through `/content/mamba/envs/zip/bin/python`, while keeping visualization cells in the normal Colab kernel.

Notebook 07 is explicitly an offline counterfactual proxy because the released repository does not include the paper's online adaptive meta-action sampler. Treat its Pareto curve as a signal for whether implementing the online controller is worthwhile, not as a paper reproduction.

Regenerate the notebooks after editing their source templates with:

```bash
python3 scripts/build_stage_notebooks.py
```
