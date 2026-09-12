# Colab Enterprise 持久化进度

所有耗时命令会继续显示实时进度条；当样本、batch 或 step 完成时，最新状态会写入下列目录，并限制为最多约每 10 秒写一次：

```text
/content/ZIP-RC-Colab/artifacts/progress/*.json
```

每条 `run_repo(...)` 命令会打印它对应的 checkpoint 路径。文件包含任务和阶段名称、运行状态、当前/总量、百分比、已耗时、预计剩余时间、预计完成时间以及最后更新时间。

## 浏览器关闭后查看

1. 重新打开 Notebook，并重新连接原来的 runtime。
2. 运行标题为“查看持久化进度”的 cell。Step 00 和统一 Notebook 的“实验进度与下一步”面板也会显示这些记录。
3. 如果原来的 cell 仍占用当前 kernel，请在连接同一 runtime 的另一个可执行 Notebook/kernel 或终端中查看；同一个忙碌 kernel 会把新 cell 排队到旧任务之后。

也可以直接查看原始文件：

```python
from pathlib import Path
import json

progress_dir = Path("/content/ZIP-RC-Colab/artifacts/progress")
for path in sorted(progress_dir.glob("*.json")):
    print(path.name, json.loads(path.read_text()))
```

状态值含义：

- `running`：任务正在初始化或运行；
- `completed`：当前耗时阶段正常完成；
- `failed`：子进程以非零状态退出；
- `stopped`：进度条在达到总量前关闭，通常表示中断或提前退出。

“距上次更新/分钟”持续增大而状态仍为 `running` 时，应检查 runtime 是否已经停止、进程是否卡住，或任务是否仍处于无法量化的模型加载阶段。

## 保存范围

JSON 位于 runtime 的 `/content` 文件系统。关闭浏览器但 runtime 仍存在时可以继续更新和读取；runtime 被删除后不会保留。需要跨 runtime 保存时，把 `artifacts/progress/` 和实验产物一起同步到 Cloud Storage。

## 资源开销

实现只在内存中保留少量标量，并在进度变化时以最多约每 10 秒一次的频率原子改写一个通常不到几 KB 的 JSON 文件：

- 内存复杂度为 `O(1)`，不会保存 token、batch 或模型输出；
- 不增加 GPU 显存占用；
- 磁盘上每个命令只保留最新快照，不会每次更新追加日志；
- CPU 和磁盘 I/O 相比模型推理/训练可以忽略。
