# Editable stage notebooks

This directory is the human-editable source for the ZIP-RC experiment.

- Open and edit the numbered IPYNB for the stage you want to change.
- Keep the first code cell in each file: the merge script replaces the Step 00 bootstrap and removes the repeated bootstrap from Steps 01–08.
- Add, remove, or reorder the remaining markdown and code cells normally.
- Do not manually edit files in `notebooks/colab_enterprise/`; regenerate them after saving a stage.
- Edit `gemini_pro_online_setup.ipynb` here as well. It is generated as a
  standalone auxiliary Notebook and is intentionally excluded from the merged
  Step 00–08 experiment.

From the repository root, run:

```bash
python3 scripts/merge_stage_notebooks.py
```

The command generates standalone Step 00–08 notebooks and the combined notebook in:

```text
notebooks/colab_enterprise/
```

Existing cells receive readable collapsible titles automatically. A newly added code cell receives a generic title and still merges correctly.

For the subscription-backed online grader, run Step 00 first, then import and
run the generated `gemini_pro_online_setup.ipynb`. Complete OAuth yourself in
the Colab Enterprise Terminal. Only the selected model slug belongs in
`artifacts/antigravity_config.json`; never place Antigravity credentials in the
repository or Notebook output.
