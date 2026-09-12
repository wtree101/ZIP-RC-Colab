# Colab Enterprise 运行入口

这里的 IPYNB 全部由 `notebooks/stages/` 生成，不要直接编辑：

- 想集中运行时，导入 `ZIP_RC_experiment_all_in_one.ipynb`；
- 想按阶段运行时，依次导入并运行 `00_memory_and_config.ipynb` 到 `08_graduation_decision.ipynb`。

在 Colab Enterprise 中选择 **My notebooks → Import → Your computer**，导入该文件，然后连接一个 L4 GPU runtime。使用左侧 Outline 在 Step 0–8 之间跳转，一次只运行当前 Step。

完成一个阶段后，重新运行 `Step 00.2 — 实验进度与下一步`；它会读取阶段报告、画出完成度，并提示下一步。

需要修改某个阶段时，直接编辑 `notebooks/stages/` 下对应的 IPYNB，然后运行：

```bash
python3 scripts/merge_stage_notebooks.py
```

分步 Notebook 和完整实验 Notebook 都会自动更新。每个分步 Notebook 都包含 Colab 初始化；Step 01–08 仍依赖 Step 00 写出的共享配置和前序阶段产物。

运行前需要满足两项：

- runtime 中存在 `/content/mamba/envs/zip/bin/python`；
- runtime 可以访问公开 GitHub，以便 Step 00 同步 `/content/ZIP-RC-Colab`。
