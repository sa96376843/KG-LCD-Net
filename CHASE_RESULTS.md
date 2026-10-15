# CHASE_DB1 补全结果与稿件核验

状态：2026-10-15 已完成。两个模型各训练 2,500 步，并对同一 8 张测试图分别评价桥接前后输出。首次 Backbone 在第 1,637 步触发 500 MB 磁盘保护，之后按原检查点和 200 MB 保护阈值续跑；未更改模型、数据划分、随机种子或评价口径。

## 可追溯运行

- 数据：`data/CHASEDB1` 的 28 张图像，按文件名排序，前 20 张训练、末 8 张测试；第一观察者标签。测试图为 `Image_11L.jpg` 至 `Image_14R.jpg`。本次发布结果使用原论文工作目录 `work/tmpcache` 中预先生成的 RGB、标签、FOV 和 Frangi 缓存；公开仓库的 `code/prep_chase2.py` 可从原图重建仓库相对路径 `data/cache/chase_arrays/` 下的输入数组。
- 模型与优化：`code/model.py` 的 `KGLCDUNet(cin=4, base=24)`；Backbone 关闭 KG、soft-clDice 权重为 0，Full 开启 KG、soft-clDice 权重为 0.5。两者均使用种子 42、96×96 patch、batch 8、Adam 3×10⁻⁴、普通余弦衰减至 1×10⁻⁵、2,500 步。
- 推理与评价：完整图像对齐补零到 16 的倍数，概率阈值 0.5；仅在生成的 FOV 内逐图计算指标后取算术平均。桥接参数沿用冻结方案，未按 CHASE 测试结果调参。`ConnErr` 为预测与标注的连通分量数量之差的绝对值，并不度量血管图拓扑完全一致。
- 执行：原运行入口为论文工作目录中的 `chase_complete.py`；发布副本是 `code/chase_complete.py`，只调整缓存与输出路径。仓库根目录按 `README.md` 的命令依次运行 `--preflight`、`--train-only`、`--evaluate-only`。本次运行的预检、日志、逐图数值、概率图与最终模型权重在 `data/cache/chase_completion/` 下。

## 四组结果

| 输出 | Dice | IoU | Sen | Spe | Acc | AUC | clDice | ConnErr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Backbone | 0.7916 | 0.6555 | 0.8261 | 0.9737 | 0.9602 | 0.9800 | 0.8211 | 96.9 |
| Backbone + bridge | 0.7916 | 0.6555 | 0.8281 | 0.9734 | 0.9601 | 0.9800 | 0.8165 | 110.9 |
| Full | 0.7748 | 0.6326 | 0.8482 | 0.9657 | 0.9549 | 0.9792 | 0.8351 | 99.1 |
| Full + bridge | 0.7744 | 0.6321 | 0.8492 | 0.9654 | 0.9547 | 0.9792 | 0.8326 | 88.6 |

Full 相比 Backbone 的敏感度和 clDice 上升，Dice 下降。桥接使 Full 的 `ConnErr` 从 99.1 降至 88.6，但使 Backbone 的该误差从 96.9 升至 110.9。因此不能声称各模块普遍提高重叠精度或连接质量；这组对比也无法单独识别 KG 与 soft-clDice 的作用。文稿已同步修改摘要、CHASE 表格和结果分析、讨论与结论，并保留与既有文献协议可能不同的提示。单次划分和单种子结果只能支持这一次受控比较，不能证明稳定或跨数据集优势。

## 固定文件校验值（SHA-256）

- 论文目录原运行脚本 `chase_complete.py`: `60369D7FAAFAF50B2BFD0095C84B04790E8D4CFC6D8FEC79DFBD0DC2CDD823A2`（本仓库 `code/chase_complete.py` 的路径配置不同，故哈希不同）
- `data/cache/chase_completion/results_chase.json`: `46D7615E5DEB78D59C0A6A012541537471E499D2E60DED02B2CB53C64588C1CD`（发布副本统一为 LF 换行；论文工作目录原文件哈希为 `997BAA49EDB627B895C33B792E699031EA8D0A77397ED1ABC573DC1677BB2477`，数值内容相同）
- `data/cache/chase_completion/backbone_final.pt`: `8D09DBA3CE0164CEEADBE9D91A1BDA284F86362CDC6CACEF13FE4EB6947ADF7A`
- `data/cache/chase_completion/full_final.pt`: `628E815D4C502A4E22C2F3D2FEF2697B28A0C279EB4329F6AEA2FA9CBF6C1306`
