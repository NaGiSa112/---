# T1 交付 · HRNet-W18 fold2 首折（队友 B 主骨干）

> 2026-11-26 · 本机 RTX 4060 Laptop 8 GB（单卡）· torch 2.8.0+cu128 / timm 1.0.30
> 协议：[`TASKS/T1`](../TASKS/T1_6GB异构骨干筛查.md) + [`TASKS/T2`](../TASKS/T2_异构成员测试协议.md)
> 判据（T2 §三）：单模检查级宏 AUC ≥ 0.8200 · 全 10 类 Spearman < 0.900（硬门槛）· 添加 Δ ≥ +0.0015

---

## 0. 结论

HRNet-W18（19.269M / 73.51 MB fp32）在 fold2 留出集跑完 12 轮末轮，**逐图概率、含翻转 TTA、无任何后处理**，已通过 T2 §2.1 的键集与取值验收。

| 口径 | fold2（3,706 检查 / 2,211 患者） | valid（859 检查 / 228 患者） |
|---|---:|---:|
| 图像级宏 AUC | 0.8121 | 0.8095 |
| 检查级宏 AUC（侧位 0.6 加权） | 0.8057 | 0.8007 |
| 检查级 + 患者收缩 0.85 | **0.8063** | **0.8023** |

单模检查级宏 0.8023（valid）低于 T2 的 0.8200 门槛；与 CAFormer-S18 首折的 0.8049 同一量级。
T2 的另外两项判据（全 10 类 Spearman、添加 Δ）本机无法计算，原因见 §9。

512 输入对照臂（T1 的"优先争取 512"）已跑完：fold2 检查级宏 0.8063、valid 0.8023，与 384 臂相同；
患者级配对 bootstrap Δ = −0.0001 ［−0.0036, +0.0035］（fold2）、+0.0000 ［−0.0071, +0.0076］（valid），均跨 0。
两臂产物都在 `runs_t1/` 下留档；本次交付对象是 384 臂。

类 1 与同折 Tiny 几乎重合：Tiny 0.676230，HRNet 0.675（fold2 检查级）。

---

## 1. 交付物（哈希已复核）

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `last.pt`（第 12 轮，固定末轮） | 77,977,845 | `c68486048ffa3391cd8731ec54e6bd84fc80d83b51915dc9a36c66381824e981` |
| `fold2_probs.csv`（6,150 行） | 787,261 | `e829336c4208a7ebed23d8c4eb41b342ff5973a972f225ce470179511d61c972` |
| `valid_probs.csv`（1,422 行） | 182,077 | `085dca6c1d5693ee8cab3305070bf13ba5ae9e526c63f18f603483e7ab689a6e` |
| `config.json` | 1,860 | `70d4fe912eab64d328996a05ddfdd3afb4945b1cbd3f2bd2e28523dd598042c0` |
| `history.json`（逐轮 AUC） | 6,898 | `e67f9d7112809796d1cedbad67943801472db20013fe218aee4fd57cc44d8617` |
| 训练日志 `runs_t1/train_hrnet18_384_f2_accum2.log` | — | — |

预训练权重：[`timm/hrnet_w18.ms_aug_in1k`](https://huggingface.co/timm/hrnet_w18.ms_aug_in1k)（ImageNet-1K，MIT）
`model.safetensors` 85,602,968 字节，SHA-256 `f9e3c451448c037d4dd549c96f3013d2926b492f69f27c74bbcc909e053e677c`
（经 `hf-mirror.com` 镜像下载，`HF_HOME` 见 §8）。

CSV 每行一图，列为 `Dicom_id,Subject_id,Study_id,p1..p10`，`p` 为 sigmoid 后概率，**未做检查聚合 / 患者收缩 / 侧位加权**。

---

## 2. 配置（全部为实测值，取自 `config.json`）

| 项 | 值 |
|---|---|
| 骨干 | `hrnet_w18.ms_aug_in1k`（19.269M 参数，fp32 73.51 MB） |
| 输入 / 缓存 | 384 / **512 拉伸缓存**（`train_512.npy`，7.95 GB，本次新建） |
| 物理批量 × 累积 | **8 × 2 = 等效 16** |
| 轮数 / 取模 | 12 轮，固定末轮（**不挑 best epoch**） |
| seed | 42 |
| 学习率 | 主干 1e-4 / 头 1e-3，wd 0.1，warmup 2 轮，cosine |
| 池化 | LSE α=0.25、τ=1，十类 BCE（`pos_weight_power=0`） |
| 归一化 | `per_image` |
| 增强 | 随机裁剪 0.75–1.0 / 比例 0.85–1.18、hflip 0.5、旋转 ±7°（p=0.35）、擦除 0.25、rbc 0.25、mixup/cutmix 0.2、label_smooth 0.03、patient_smooth 0.3 |
| 蒸馏 / EMA / 辅助分支 | 全部关闭（0 / 0 / 无） |
| 折表 | `data/patient_folds_5f.csv`，训练 folds 0,1,3,4 = 24,194 图 / 14,807 检查；留出 fold2 = 6,150 图 / 3,706 检查 / 2,211 患者 |
| 代码改动 | `src/cxr/models.py`（timm 骨干登记）、`src/cxr/train.py` + `config.py`（梯度累积） |

---

## 3. 训练曲线（fold2，图像级）

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| macro | .7453 | .7672 | .7758 | .7833 | .7901 | .7981 | .7978 | .8004 | .8047 | .8066 | .8086 | **.8092** |
| c1 | .615 | .639 | .648 | .656 | .681 | .685 | .667 | .682 | .692 | .693 | .702 | **.701** |
| c4 | .632 | .643 | .650 | .663 | .688 | .705 | .700 | .702 | .710 | .712 | .715 | **.720** |

**第 12 轮仍在上升，未进入平台**（e10–e12：0.8066 → 0.8086 → 0.8092）。按预注册纪律取固定末轮，
未按 fold2 选择 epoch。同折 Tiny 在 12 轮已收敛，这一差异属于事实记录。

---

## 4. 逐类结果（检查级宏 AUC，侧位 0.6 加权）

| 类 | 名称 | fold2 | valid | 阳性(fold2 检查 / valid 检查) |
|---|---|---:|---:|---:|
| 1 | 纵隔心影增大 | 0.675 | 0.729 | 119 / 24 |
| 2 | 气胸 | 0.824 | 0.861 | — |
| 3 | 肺部实变 | 0.793 | 0.774 | — |
| 4 | 肺炎 | **0.729** | 0.713 | 289 / 68 |
| 5 | 肺水肿 | 0.888 | 0.882 | — |
| 6 | 心影扩大 | 0.794 | 0.775 | — |
| 7 | 肺不张 | 0.794 | 0.782 | — |
| 8 | 肺部阴影 | **0.755** | 0.704 | — |
| 9 | 胸腔积液 | 0.920 | 0.912 | — |
| 10 | 无异常发现 | 0.885 | 0.874 | — |
| — | **宏平均** | **0.8057** | **0.8007** | — |

对照同折 Tiny（`docs/56` 表内值）：宏 0.810430、类 1 0.676230 → HRNet 宏 −0.0047、类 1 −0.0012。
**弱类 c1/c4/c8 上没有可辨识的改善**，这条骨干的价值只能来自"错法不同"（相关性）。

---

## 5. 与 T2 判据的对照

| 门槛 | 阈值 | 本结果 | 结论 |
|---|---:|---:|---|
| 单模检查级宏 AUC | ≥ 0.8200 | 0.8023（valid） | ✗ 未达，但接近 CAFormer 的 0.8049 |
| 全 10 类 Spearman | < 0.900 | **未计算** | 需要 `runs/member_valid/` 的三只 Tiny + 供体 B3 逐图预测 |
| 添加 Δ 峰值 | ≥ +0.0015 | **未计算** | 同上 |

这两项在本机不可得（`runs/member_valid/` 不在我们的快照里），因此上表只覆盖"单模强度"一项。
HRNet 与 ConvNeXt 的结构差异（并行多分辨率 vs 纯卷积分层）不构成相关性结论，需实测。

---

## 6. 三处与协议的偏离（都是实测后的必要选择，不是省事）

| # | 协议写法 | 实际做法 | 原因与证据 |
|---|---|---|---|
| 1 | 物理 batch 2（6 GB 卡） | **物理 8 × 累积 2 = 等效 16** | 本机是 8 GB（8188 MiB），实测 384/bs16 峰值 allocated 3357 MiB、bs8 1799 MiB，都装得下。物理批量由显存取最大可行值，**等效批量仍严格 16**；BN 统计与 mixup 配对按物理 8（协议允许按物理批量决定） |
| 2 | drop_path 生效（配方 0.15） | **实际 0** | timm 的 HRNet 构造器**接受** `drop_path_rate` 但**不创建任何 StochasticDepth**（实测 `isinstance` 计数 = 0，`hasattr(m,'drop_path_rate')` = False），传 0.15 会被静默忽略。已按 0 记录并在日志里打印 `actual_stochastic_depth=[]` |
| 3 | 512 输入（"优先争取"） | **首折先 384** | 512/bs16 估算峰值 ≈ 6.0 GB，会挤掉桌面进程；512/bs8 约 3 GB 可行但 12 轮约 4.1 h。按 T1 的"同一顺序、先 384 通过再测 512"，故 384 先交 |

补充说明（已核对仓库自身实验，确认不是偏离）：

- `label_smooth 0.03` / `patient_smooth 0.3` / `rbc 0.25` 与仓库自己的成员实验命令**逐项一致**
  （`docs/24_B4_Swin.md` §六 的 Swin-T 训练命令：`--label-smooth 0.03 --mixup 0.2 --cutmix 0.2
  --patient-smooth 0.3 --erase-prob 0.25 --rbc-prob 0.25 --rot-deg 7 --scale-min 0.75`）。
  T2 §2.3 的增强清单只是摘要，省略了这三项；本臂按仓库实际配方执行，因此与干净 OOF 基线同口径。
  （T2 里"亮度/对比度在逐图标准化后会被抵消"的注记也成立，`rbc` 实际是 no-op。）
- `workers=0`：Windows 沙箱禁止 DataLoader 子进程建管道（WinError 5），数据读取与增强在主进程完成，端到端吞吐因此偏低（见 §7）。
- **提交号**：本分支是 zip 快照、仓库内无 `.git`，给不出 commit hash。替代方案是逐文件 SHA-256 + 可 `git apply` 的补丁（`docs/58_T1_code.patch`）+ 完整命令，三者合起来可重建同一环境。这一项**需要队长确认是否接受**。

---

## 7. 实测资源

| 项 | 值 |
|---|---|
| 吞吐（纯训练，bs8 微批） | **35.3 图片/秒**（226.7 ms/微批，10 步实测） |
| 吞吐（端到端含验证/读盘，workers=0） | 27.2 图片/秒（每轮 24,194 图 ÷ 889 s） |
| 峰值显存 allocated / reserved | bs8：1799 / 1926 MiB；bs16：3357 / 3872 MiB（8,188 MiB 卡） |

> 显存与吞吐的取数口径：用 `tools/t1_backbone_smoke.py` 在与训练完全相同的模型/分辨率/物理批量/AMP 下跑 10 个完整步（含 AdamW）实测，**不是**在 12 轮进程内插桩；训练期间另用 `nvidia-smi` 抽样，整卡占用约 4.2–4.6 GiB（含桌面进程约 2.1 GiB）。
| 单轮耗时 | 731–911 s（平均 770 s） |
| 12 轮总训练时长 | **9,245.6 s ≈ 2.57 h** |
| 推理（导出 6,150 图 + 翻转 TTA） | 约 3 min（bs16） |

对比同规模的参考：本机 ConvNeXt-S @448 峰值 3.6 GB、122 s/轮（20% 子集）。HRNet 在 384 上更吃时间（高分辨率分支在 1/4 尺度上跑 18 个 block）。

---

## 8. 复现

```powershell
$repo = "<repo>"
$env:CXR_DATA   = "D:\HLW\cxr_first"
$env:CXR_CACHE  = "D:\HLW\cxr_work\cache_t1"     # train_512.npy 所在目录
$env:HF_HOME    = "D:\HLW\cxr_work\hf_cache"
$env:HF_ENDPOINT = "https://hf-mirror.com"        # 国内直连 huggingface.co 会失败
$env:PYTHONPATH = "$repo/src"

# 0) 512 拉伸缓存（一次性，7.95 GB）
python -c "import sys;sys.path.insert(0,r'$repo/src');from cxr.cache import build_cache;build_cache('train',size=512,workers=16,cache_dir=r'D:\HLW\cxr_work\cache_t1')"

# 1) 训练（12 轮末轮，等效 16）
python -m cxr.train --arch hrnet_w18.ms_aug_in1k --img-size 384 --cache-size 512 `
  --epochs 12 --batch-size 8 --accum-steps 2 --lr-backbone 1e-4 --lr-head 1e-3 --weight-decay 0.1 `
  --warmup-epochs 2 --drop-path 0 --ema 0 --amp fp16 --workers 0 `
  --head-type lse --lse-alpha 0.25 --lse-tau 1.0 `
  --mixup 0.2 --cutmix 0.2 --erase-prob 0.25 --rbc-prob 0.25 --rot-deg 7 --scale-min 0.75 `
  --label-smooth 0.03 --patient-smooth 0.3 `
  --fold-csv data/patient_folds_5f.csv --train-folds 0,1,3,4 --valid-fold 2 `
  --seed 42 --out-dir runs_t1/hrnet18_384_f2_accum2

# 2) 导出（逐图、翻转 TTA、不聚合）
python tools/t1_export_fold_probs.py --ckpt runs_t1/hrnet18_384_f2_accum2/last.pt `
  --split fold --fold 2 --img 384 --cache-size 512 --out runs_t1/hrnet18_384_f2_accum2/fold2_probs.csv
python tools/t1_export_fold_probs.py --ckpt runs_t1/hrnet18_384_f2_accum2/last.pt `
  --split valid --img 384 --cache-size 512 --out runs_t1/hrnet18_384_f2_accum2/valid_probs.csv

# 3) 验收（键集 / 取值 / SHA）
python tools/t1_verify_csv.py --probs runs_t1/hrnet18_384_f2_accum2/fold2_probs.csv --split fold2
python tools/t1_verify_csv.py --probs runs_t1/hrnet18_384_f2_accum2/valid_probs.csv --split valid
```

推理几何与训练 eval **逐像素同源**：都是"整幅 512 → `interpolate(bilinear, antialias=True)` 拉伸到 384"，没有裁剪、没有 letterbox。这条按 T2 §2.2 的警告专门核过。

---

## 9. 未计算的两项与所需输入

```bash
python tools/e1_score_v2.py --probs runs_t1/hrnet18_384_f2_accum2/valid_probs.csv  --name hrnet_w18_384
python tools/e1_score_v2.py --probs runs_t1/hrnet18_384_f2_accum2/fold2_probs.csv --name hrnet_w18_384_fold2
```

本机没有 `runs/member_valid/`（三只 Tiny + 供体 B3 的逐图预测），所以 Spearman 与添加 Δ 无法计算，
本交付只包含单模强度。`e1_score_v2.py` 的部署口径（逐图 half_power → 检查加权 → 患者收缩 0.85）在单成员时
等价于本报告的 `study + shrink 0.85` 列，聚合常数 `lat=0.6 / shrink=0.85` 已与 `tools/e1_score_v2.py` 核对一致。

---

## 10. 过程中发现、需要队内对齐的三件事

1. ~~**折表 SHA 与 `TASKS/README.md` 写的 `5b0cf113…` 不一致。**~~ —— **已查清，是行尾符，不是划分差异。**
   本分支 `data/patient_folds_5f.csv` 的 SHA-256 是 `c465d164…`；把它的 `\n` 全替换成 `\r\n` 后，
   SHA-256 **精确等于** `5b0cf113345adb4385089a8ee694dad6e5962899f8bb2cbce5f3a36d5d1dea21`。
   即两份文件是同一份内容，只是 zip 打包把 CRLF 转成了 LF（pandas 在 Windows 写 CSV 用 `os.linesep`）。
   再加两道内容级证据：
   - 用仓库生成器 `tools/make_patient_folds.py`（`SEED=2026`）重跑，与本表逐 Study 比对：**18,513 个 Study 折号不一致 0 个**；
   - `tools/compare_fold_tables.py`（本快照缺这个脚本，已补）：`折号不一致的 Study：0`、
     `B 的训练集里有 0 个 A-fold2 检查`、反向亦 0。
   ⇒ fold2 上的 0.8063 不是泄漏后的虚高。详见 `docs/59_T1_交付清单.md` §二。

2. **timm HRNet 的 `drop_path_rate` 是空参数**（见 §6-2）：构造器接受，但不创建 StochasticDepth，
   传 0.15 被静默忽略（实测计数 0）。日志里已打印 `actual_stochastic_depth=[]`。

3. **EfficientViT-B1 首折已按你的更正中止**（`runs_t1/evb1_384_f2`，跑到第 2 轮）。骨干登记与冒烟数据（7.503M / 28.62 MB、384/bs16 139 img/s、峰值 1805 MiB、灰度等价 1.8e-07）都留在 `runs_t1/smoke_evb1_384_bs16.json`，随时可复跑，参数已固化。

---

## 11. 梯度累积实现说明（`src/cxr/train.py`）

按 T1 §首折协议要求加的，四条都落实了：

| 要求 | 实现 |
|---|---|
| 损失按**实际累计样本数**归一化 | 每个微批的 loss 先除以"本组实际微批数" `g_micro` 再反传；`drop_last=True` 保证微批等大，故组内样本数 = 微批数 × 物理批量 |
| 优化器步 / GradScaler 按优化器步 | 只在组尾 `scaler.step / update / opt.zero_grad` |
| 学习率与 warmup 按优化器步 | `total_steps = epochs × ceil(n_micro/accum)`，warmup 用同一单位 |
| 每轮末尾不足一组的处理 | `g_micro = min(accum, n_micro - 组起点)`，末组按实际微批数归一化，且保证**最后一个微批一定触发优化器步**（否则梯度会跨 epoch 泄漏） |

自检（`tools/t1_accum_check.py`，可复跑）：

```
组算术: 全部通过   (accum ∈ {1,2,3,4,8} × n_micro ∈ {1,2,3,5,7,11,1511,1512,1513}，
                    优化器步数均等于 ceil(n_micro/accum)，末组均收尾)
梯度等价: max|Δ| = 3.725e-09  max rel = 1.518e-07  (通过)
          拆 3 个微批累积的梯度 == 整批 12 样本平均损失的梯度
```

端到端 dry run（`runs_t1/_dryrun_accum`，512 图 / 1 轮 / 物理 8 × 累积 4）：97 个微批 → 25 个优化器步 = ceil(97/4)，末组 1 个微批也正常收尾。

---

## 12. 与 T1 / T2 的逐项对照
| 要求（出处） | 本交付 | 状态 |
|---|---|---|
| 逐图概率 6,150 行（T1-交付2 / T2-2.1） | 6,150 行 / 3,706 检查 | ✅ |
| 列名 `Dicom_id,Subject_id,Study_id,p1..p10`（T2-2.1） | 完全一致 | ✅ |
| Dicom_id 与 fold2 不重不漏（T1-验收） | 集合相等，重复 0、缺失 0、多余 0 | ✅ |
| 图-患者-检查完整对应（docs/56 §3） | 逐行比对 `(Subject_id, Study_id)`，0 行不一致 | ✅ |
| p1..p10 ∈ (0,1) 且有限（T2-2.1） | 实测 [0.0024, 0.9709]，无 NaN | ✅ |
| 含翻转 TTA、按概率平均（T1 / T2-2.1） | 两次前向 sigmoid 后取均值 | ✅ |
| 不做检查聚合/患者收缩/侧位加权（T2-2.1） | CSV 为纯逐图概率；聚合量只在 `*_metrics.json` 里 | ✅ |
| 另交 valid 逐图概率 1,422 行（T2-2.1） | 1,422 行 / 859 检查 | ✅ |
| last.pt 固定末轮、不挑 best（T1 / T2-4） | 第 12 轮 | ✅ |
| 训练折 0,1,3,4 → 24,194 图（T1 首折协议） | 24,194 图 / 14,807 检查 | ✅ |
| 留出 fold2 = 6,150 图 / 3,706 检查 / 2,211 患者 | 完全一致 | ✅ |
| 患者零交叉 | 折表校验 True | ✅ |
| 12 轮 / seed 42 / 无蒸馏 / 无 EMA / 无辅助分支 | 一致 | ✅ |
| 512 拉伸缓存 + 逐图标准化 + LSE α=0.25 τ=1 | 一致 | ✅ |
| lr 主干 1e-4 / 头 1e-3 / wd 0.1 / warmup 2 轮 | 一致（按优化器步） | ✅ |
| 增强沿用仓库配方（T2-2.3） | 与 `docs/24` 的 Swin-T 命令逐项一致 | ✅ |
| 等效批量 16（T1 首折协议） | 物理 8 × 累积 2 = 16 | ✅ |
| 梯度累积四项语义（T1 / docs/56） | 见 §11，含自检与负向测试 | ✅ |
| 预训练来源 + SHA-256（T1-交付4） | HF `timm/hrnet_w18.ms_aug_in1k`，`f9e3c451…` | ✅ |
| 末轮权重 SHA-256 + 字节数（T1-交付4） | `c6848604…`，77,977,845 B | ✅ |
| 训练日志 + 吞吐 + 峰值显存（T1-交付5） | 日志在 `runs_t1/`；吞吐与显存见 §7（附取数口径） | ✅ |
| 归一化方式（T2-2.2） | `per_image` | ✅ |
| 推理几何与训练几何一致（T2-2.2） | 同一 `load_square(512)` + 同参数 `interpolate` | ✅ |
| 提交号（T1-交付6 / T2-2.2 / README-§四.1） | 四件替代件已备（逐文件哈希 / `b6b1b73` 补丁 / 完整命令 / last.pt+config.json） | ⚠️ 待队长裁定 |
| 折表 SHA-256 与 README 一致（README-§四.4） | **已查清：CRLF vs LF，同一内容**；划分差异 0、泄漏 0 | ✅ |
| 物理批量 2（T1 首折协议示例） | 物理 8（显存决定），等效 16 不变 | ⚠️ 已注明 |
| drop_path 0.15（部署配方） | 实际 0（timm HRNet 无 StochasticDepth） | ⚠️ 架构限制 |
| 512 输入（T1 "优先争取"） | 已跑完并与 384 配对比较，两者打平（见 §13） | ✅ |

---

## 13. 512 变体（T1 "优先争取 512" 的答复）

同一配方，只把 `--img-size` 从 384 改成 512（缓存本来就是 512，512 输入等于不重采样）。
12 轮末轮，`runs_t1/hrnet18_512_f2_accum2`，每轮 1,005–1,061 s，总约 3.4 h。

| 口径 | 384 | **512** | 差值 |
|---|---:|---:|---:|
| fold2 检查级宏（+收缩 0.85） | 0.8063 | **0.8063** | −0.0001 |
| valid 检查级宏（+收缩 0.85） | 0.8023 | **0.8023** | +0.0000 |
| fold2 类 1 | 0.675 | 0.681 | +0.005 |
| fold2 类 4 | 0.728 | 0.722 | −0.007 |
| fold2 类 8 | 0.755 | 0.751 | −0.004 |

配对 bootstrap（`tools/t1_paired_compare.py --unit patient`，2,000 次，**按患者**有放回重采样，
与仓库 `tools/compare_folds.py` 口径一致；同一患者的多次检查整组进出）：

```
fold2：delta -0.0001   95%CI [-0.0036, +0.0035]   (2,211 患者) → 区间跨 0，无差异
valid：delta +0.0000   95%CI [-0.0071, +0.0076]   (228 患者)   → 区间跨 0，无差异
```

（按检查重采样的参考值：fold2 [−0.0035, +0.0034]、valid [−0.0066, +0.0068]；两者结论相同，
但患者级更保守，也应作为上报口径。）

**测量结果：在 12 轮预算下，512 与 384 在部署口径上没有可测差异。**

仓库既有数据只能部分对照：`results/resolution_test.json` 里 ConvNeXt-Tiny 的曲线是
**512 (0.8096) > 544 (0.8085) > 640 (0.8046) > 768 (0.7899)**（fold0，3,712 检查），
即"超过 512 后像素越多越差"，但**仓库未测过 384**。本节是 384 与 512 的直接配对测量。

另有一处差异记录：512 臂的图像级 val 在 **e11 见顶 0.8105 后 e12 回落到 0.8095**，
384 臂到 e12 仍在上升（0.8066→0.8086→0.8092）。

成本实测：512 臂总耗时 3.4 h（12 轮），384 臂 2.57 h，差值约 0.85 h；
峰值显存 bs8 下 3,023 MiB vs 1,799 MiB，差值 1.2 GB。两臂产物均保留在 `runs_t1/`。

512 臂交付物哈希：

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `last.pt` | 77,977,845 | `19f99a0bd7b1638222bd08b9eb27fc5a93ffe0cf3ffe39fc09be41c506929a07` |
| `fold2_probs.csv` | 787,261 | `ab14f49a43d67189be02f24a98dd7921189b62b0081fe083190c08a2dfb53f0a` |
| `valid_probs.csv` | 182,077 | `961ce30432d3c0c5116b1385ef2a96c632dc2d7ff4187c0276c807a849053b9f` |
| `config.json` | 1,860 | `514db992644184fdd00c975c32d94288df0cb08bd8f3c750de11eb9b788c9de7` |

两个 CSV 同样通过 T2 §2.1 全部验收（6,150 / 1,422 行，键集不重不漏，图-患者-检查对应 0 错，p∈(0,1)）。
