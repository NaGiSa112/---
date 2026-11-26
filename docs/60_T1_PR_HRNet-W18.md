# T1-B: HRNet-W18 fold2 首折

> PR 描述（按 `TASKS/README.md` §四 的必填项逐条填）
> 交付报告：[`docs/58_T1_HRNet-W18_fold2_交付.md`](58_T1_HRNet-W18_fold2_交付.md)
> 代码补丁：[`docs/58_T1_code.patch`](58_T1_code.patch)
> 日期：2026-11-26

---

## 1. 代码提交号

本分支是 zip 快照交付（**仓库内无 `.git`**），无法给 commit hash。改动的三个文件逐文件给 SHA-256，等价可复核：

| 文件 | SHA-256 | 说明 |
|---|---|---|
| `src/cxr/models.py` | `b8ce95f1fa8c350f2df0d37128fa3667c434a0cf50d1128beb4948e964d920b2` | 登记 timm 异构骨干（efficientvit / repvit 走专用分支；hrnet 走通用 timm 分支） |
| `src/cxr/train.py` | `253b64c8377037174a878977b030a38ca7183b3a2fd2ed49e96810f61b193a2f` | 梯度累积（`--accum-steps`） |
| `src/cxr/config.py` | `8c7bce2c056ddf99b9773ae91c8c1b1216764266af553f559049ad3269de68e6` | `TrainConfig.accum_steps` |

**补丁基准 commit：`b6b1b73`**（`docs/56` 引用的快照）。补丁为 `docs/58_T1_code.patch`（含 3 改 + 6 新增，
40,639 字节 / `b24966bd…`），本机已验证：pre-image 哈希匹配 → `git apply --check` 退 0 → `git apply` 退 0 →
落地文件（3 个源码 + 6 个新脚本）与上表逐字节一致。完整四样东西见 `docs/59_T1_交付清单.md`。

新增脚本（工具，不改训练语义）：

| 文件 | SHA-256 | 用途 |
|---|---|---|
| `tools/t1_backbone_smoke.py` | `a29b31a38f05dca9b76563ad801d6aa9b5458b70a8f522c33265de6769edbc4e` | 上卡前：预训练加载 + 灰度等价 + 显存/吞吐 |
| `tools/t1_export_fold_probs.py` | `035ffbebe8cbd4c8f5c40158bcc216cec02c2756c97fef6698c6212ff70fe2dd` | ckpt → 逐图概率 CSV（翻转 TTA，无后处理） |
| `tools/t1_verify_csv.py` | `fbc903bc0c0b0d51943d474fa1d28062f3e1591027bc9926a4220d5a39a8939d` | T2 §2.1 验收（键集 / 图-患者-检查对应 / 取值 / SHA） |
| `tools/t1_accum_check.py` | `4ac22d36a1b745665b6c03f3b8ca9f2a5a6c29d31b1d5ac3c91f05c8507cb98d` | 累积的组算术 + 梯度等价自检 |
| `tools/compare_fold_tables.py` | `befd4f9338d97d144a3b67f02f38d9692a242c50d979c924095d021a57dda638` | 两份折表的划分差异与双向泄漏检查 |
| `tools/t1_paired_compare.py` | `6ffdd50bb3a8985033997574f7a704fbf332c88ab0f031f59e7584b13c5278ca` | 同留出集配对 bootstrap（默认**患者级**，对齐 `compare_folds.py`） |

---

## 2. 配置

| 项 | 值 |
|---|---|
| 骨干 | `hrnet_w18.ms_aug_in1k`（19.269M 参数 / 73.51 MB fp32） |
| 输入尺寸 | **384**（缓存 512 拉伸，`interpolate` bilinear+antialias，无裁剪无 letterbox） |
| 物理批量 × 累积 | **8 × 2 = 等效 16** |
| 蒸馏 | **关闭**（`distill_weight=0`） |
| 随机种子 | **42** |
| 轮数 | **12 轮，固定末轮**（不用 `best.pt`，不按 fold2 挑 epoch） |
| 折 | 训练 folds **0,1,3,4**（24,194 图）→ 留出 **fold 2**（6,150 图 / 3,706 检查 / 2,211 患者） |
| 归一化 | `per_image` |
| 学习率 / 调度 | 主干 1e-4、头 1e-3、wd 0.1、warmup 2 轮、cosine（均按**优化器步**） |
| 池化 | LSE α=0.25、τ=1 |
| 增强 | 裁剪 0.75–1.0 / 比例 0.85–1.18、hflip 0.5、旋转 ±7°(p .35)、擦除 .25、rbc .25、mixup/cutmix .2、label_smooth .03、patient_smooth .3 |

---

## 3. 预训练权重

`timm/hrnet_w18.ms_aug_in1k`（ImageNet-1K，MIT 许可，**非医学预训练**）

| 项 | 值 |
|---|---|
| 文件 | `model.safetensors` |
| 字节 | 85,602,968 |
| SHA-256 | `f9e3c451448c037d4dd549c96f3013d2926b492f69f27c74bbcc909e053e677c` |
| 下载途径 | `HF_ENDPOINT=https://hf-mirror.com`（国内直连 huggingface.co 失败） |

---

## 4. 实测吞吐与显存

| 项 | 值 |
|---|---|
| 纯训练吞吐（384 / 物理 8） | **35.3 图片/秒**（226.7 ms/微批，10 步；含 AdamW） |
| 端到端吞吐（含每轮验证与 workers=0 的读盘） | 27.2 图片/秒（24,194 图 ÷ 889 s） |
| 峰值显存 allocated / reserved（384 / bs8） | **1,799 / 1,926 MiB** |
| 峰值显存（384 / bs16，未采用） | 3,357 / 3,872 MiB |
| 12 轮总时长 | **9,245.6 s ≈ 2.57 h**（avg 770 s/轮，731–911 s） |
| GPU | RTX 4060 Laptop 8,188 MiB |

---

## 5. 折表

| 项 | 值 |
|---|---|
| 文件 | `data/patient_folds_5f.csv`（518,392 字节） |
| 本分支 SHA-256（LF 行尾） | `c465d1649eb94506f25c1d56074a010f24798d9b634b6876e3bed0723049d89d` |
| `TASKS/README.md` 的 SHA-256（CRLF 行尾） | `5b0cf113345adb4385089a8ee694dad6e5962899f8bb2cbce5f3a36d5d1dea21` |

**已查清：两者是同一份内容，差异只来自行尾符。** 把本文件的 `\n` 全部替换为 `\r\n`，SHA-256 精确等于
`5b0cf113…`（pandas 在 Windows 写 CSV 用 `os.linesep`，zip 打包时被转成 LF）。

划分一致性（`tools/compare_fold_tables.py`，本快照缺该脚本，已补）：

```
折号不一致的 Study：0
若以 A 的 fold 2 为评估集：B 的训练集里有 0 个 A-fold2 检查
若以 B 的 fold 2 为评估集：A 的训练集里有 0 个 B-fold2 检查
[结论] 两份折表划分一致，无泄漏风险
```

另用生成器 `tools/make_patient_folds.py`（`SEED=2026`）重跑比对：18,513 个 Study 折号不一致 **0** 个；
每折 studies `3712/3689/3706/3695/3711`、patients `2211×5` 完全复现。

---

## 6. 交付物与哈希

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `last.pt`（第 12 轮） | 77,977,845 | `c68486048ffa3391cd8731ec54e6bd84fc80d83b51915dc9a36c66381824e981` |
| `fold2_probs.csv` | 787,261 | `e829336c4208a7ebed23d8c4eb41b342ff5973a972f225ce470179511d61c972` |
| `valid_probs.csv` | 182,077 | `085dca6c1d5693ee8cab3305070bf13ba5ae9e526c63f18f603483e7ab689a6e` |
| `config.json` | 1,860 | `70d4fe912eab64d328996a05ddfdd3afb4945b1cbd3f2bd2e28523dd598042c0` |

（大文件不进 git，走队内渠道。）

---

## 7. 自检结果

```
[verify] fold2_probs.csv  rows=6150  expected=6150  studies=3706
[verify] 通过：键集不重不漏，p1..p10 ∈ (0,1) 且有限
[verify] valid_probs.csv  rows=1422  expected=1422  studies=859
[verify] 通过：键集不重不漏，p1..p10 ∈ (0,1) 且有限
```

单模（部署口径：侧位 0.6 加权 → 患者收缩 0.85）：

| 口径 | fold2 | valid |
|---|---:|---:|
| 图像级宏 AUC | 0.8121 | 0.8095 |
| 检查级宏 AUC | 0.8057 | 0.8007 |
| 检查级 + 收缩 0.85 | **0.8063** | **0.8023** |
| 类 1 | 0.675 | 0.729 |
| 类 4 | 0.729 | 0.713 |
| 类 8 | 0.755 | 0.704 |

---

## 8. 与协议的偏离（3 处，均已实测）

1. **物理批量 8 而非 2**：本机 8 GB（8188 MiB），385 输入下 bs16 峰值 3.4 GB 也装得下；等效批量仍严格 16，BN 统计/mixup 配对按物理 8（协议允许按物理批量决定）。
2. **drop_path 实际为 0**：timm HRNet 接受 `drop_path_rate` 但**不创建 StochasticDepth**（实测计数 0、属性不存在），传 0.15 会被静默忽略。
3. **512 输入作为对照臂已跑完**：`runs_t1/hrnet18_512_f2_accum2`（bs8、峰值 3023 MiB、3.4 h/12 轮）。
   与 384 的患者级配对比较：fold2 Δ −0.0001 ［−0.0036, +0.0035］、valid Δ +0.0000 ［−0.0071, +0.0076］，均跨 0。
   本次交付对象为 384 臂，512 臂产物同目录留档。限定条件：这是 **12 轮预算下**的测量结果；
   512 的图像级 val 在 e11 见顶后回落，384 到 e12 仍在上升。

另：`label_smooth 0.03` / `patient_smooth 0.3` 取自部署配方（T2 §2.3 增强清单未列），若干净 OOF 基线未开这两项，对比时需注明目标口径差异。

---

## 9. 未计算项（本机不可得）

```bash
python tools/e1_score_v2.py --probs .../valid_probs.csv  --name hrnet_w18_384
python tools/e1_score_v2.py --probs .../fold2_probs.csv --name hrnet_w18_384_fold2
```

单模强度 0.8023（valid）低于 0.8200 门槛，量级与 CAFormer 首折 0.8049 相当；**能否进集成取决于全 10 类 Spearman 是否 < 0.900**。本机没有 `runs/member_valid/`，这两项我算不了。
