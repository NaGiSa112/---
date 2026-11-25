# 最终提交材料包核验记录（2026-11-13）

## 一、结论

**包内 `model/submission.csv` 已换成线上 78.51 那一版，并已验证包内代码能逐字节复现它。**

| 检查 | 结果 |
|---|---|
| 最高分文件 | `submissions/FINAL_2026-11-13/submission.csv` |
| SHA-256 | `e4142057d5073e2566bf9a41d9cb41f97e73c38c58793cc560e796d9aa1b2d70` |
| **包内重算 vs 该文件** | **SHA-256 完全相同**（2,218 图 → 1,414 study → 14,140 行） |
| 逐类排名一致性 | 10 / 10 类一致 |
| 包内 `model/cost.py` 自检 | 聚合 0 行不符、逐样本 0 不一致（max \|Δp\| 9.71e-17）、耗时 PASS |
| 端到端耗时 | 49–65 ms（多次测量），p95 ≤ 77 ms，上限 100 ms |
| 加载权重合计 | 377.45 MB（≤ 500 MB） |
| MANIFEST | 逐文件字节数与 SHA-256 全部匹配，0 失败，无 `__pycache__` |

## 二、本次改为按 78.51 版本重建时发现并修复的 4 个缺陷

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | 构建脚本的权重复制清单**写死为 3 个** | 新的第 4 个成员 `tiny_seed44_last12.pt` 不会被复制，评委加载必然失败 | 改为读取随包 `deploy.json` 的 members，逐个复制 |
| 2 | 清单 A1 从**仓库** `model/deploy.json` 取成员信息 | A1 会写成"ConvNeXt-Tiny ×2、输入 512"，与实际（×3、输入 544）不符 | `members_desc()` 改为读取随包 deploy 路径 |
| 3 | `__pycache__/*.pyc` 被打进包 | 构建时跑过 cost.py，`.pyc` 生成/变化导致 MANIFEST 每次校验都对不上 | 打包前删除，且 MANIFEST 与 zip 均排除 |
| 4 | 供体参考实现路径越界（继承自上一版问题） | `deploy.py` 从仓库路径读 `preprocess_predict.py`，包内不存在 | 文件打进 `code/cxr/donor_ref/`，加载器多路径查找 + `CXR_DONOR_REF` |

## 三、两个产物的区别

| 包 | zip 体积 | 预训练权重 | 建议 |
|---|---|---|---|
| `队长名_最终提交材料.zip` | **502.4 MB** | 随包（161 MB） | 若平台无 500 MB 上传限制才用 |
| `队长名_最终提交材料_精简版(不含预训练权重).zip` | **350.5 MB** | 仅公网链接 + SHA-256 | **推荐上传这一个** |

两个包的 `model/pretrain_model/` 目录都存在：
完整版含两个 `.pth` 文件；精简版含 `预训练权重来源与校验.md`（公网直链 + 每个文件的 SHA-256 + 校验命令）
与 `donor_v21_compliance.json`。征集通知对链接的要求是"公网可直接访问、无需登录或单独授权"，
两条链接均满足。

## 四、模型与合规状态

- 成员：ConvNeXt-Tiny ×3（输入 544，LSE α=0.30）+ EfficientNet-B3 ra2_in1k ×1（输入 384），权重 0.2833/0.2833/0.2833/0.15
- 融合：半幂融合；hflip TTA；侧位权重 0.6；患者收缩 0.85；CUDA Graph 用于 Tiny（捕获失败自动回退）
- 预训练：仅 ImageNet-1k（torchvision BSD-3-Clause / timm Apache-2.0），**无医学预训练**
- 训练数据：仅主办方 `train.csv`；验证集只用于检查点选择
- 测试集：未参与训练、调参、阈值拟合、统计量估计或伪标签

## 五、仍需用户提供

1. **队长名** —— 用于 zip 文件名与包内顶层目录名
2. **平台上传体积上限** —— 决定用完整版还是精简版

## 六、复现命令（评委侧）

```powershell
pip install -r code/requirements.txt
$env:CXR_DATA = "<主办方 cxr_first 目录>"
python code/predict.py --split test --batch-size 1 --lat-weight 0.6 --shrink 0.85 `
    --deploy model/deploy.json --out regenerated.csv
# regenerated.csv 应与 model/submission.csv 逐字节一致
python model/cost.py model/submission.csv <pictures_100> <pictures_5>
```
