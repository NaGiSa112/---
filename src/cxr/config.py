"""Project-wide configuration for the chest X-ray multi-label classifier."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))     # .../src/cxr  or  .../code/cxr


def _detect_root() -> str:
    """Project root = parent of `src/` (or of `code/` inside the submission package).

    Derived from the file location so the submission package keeps working after
    it is unpacked anywhere, instead of relying on a hard-coded absolute path.
    """
    if os.environ.get("CXR_ROOT"):
        return os.environ["CXR_ROOT"]
    return os.path.dirname(os.path.dirname(_HERE))


def _detect_data(root: str) -> str:
    """Locate the official `cxr_first/` directory.

    Search order: explicit CXR_DATA, then a sibling of the project/package root,
    then <root>/cxr_first, then the original development path.
    """
    if os.environ.get("CXR_DATA"):
        return os.environ["CXR_DATA"]
    candidates = [
        os.path.join(root, "cxr_first"),
        os.path.join(os.path.dirname(root), "cxr_first"),
        os.path.join(root, "..", "cxr_first"),
        r"E:\挑战赛\cxr_first",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.join(root, "cxr_first")


ROOT = _detect_root()
DATA = _detect_data(ROOT)
CACHE = os.environ.get("CXR_CACHE", os.path.join(ROOT, "cache"))
RUNS = os.path.join(ROOT, "runs")
SUBS = os.path.join(ROOT, "submissions")
REPORTS = os.path.join(ROOT, "reports")

NUM_CLASSES = 10
LABELS = [str(i) for i in range(1, NUM_CLASSES + 1)]

# 官方类别映射（2026-11-20 核实，取代此前的 NIHCXR 顺序假设）。
# 来源一：官网类别对照图
#   ncccu2026.oss-cn-zhangjiakou.aliyuncs.com/ueditor/20260927/5385a992766a388da1480b3720b999a7.png
#   SHA-256 034a098696d60c6a3d6ee041d75b6d22ec85c5c1dc272fc0ad44846c12364cb5
# 来源二：大赛规程 PDF 第 11 页「（3）类别标签说明」，本地
#   04_新一轮/00_规则/大赛规程_含类别表.pdf（两份文件逐项一致）
# 官方用的是 CheXpert 类序，不是 NIHCXR 类序：此前 10 个名字里有 9 个是错的（仅 10 号相同）。
# 该更正只影响报告文字与机制解释，不影响训练与评分 —— 全流程按数字 id 运作。
LABEL_NAMES = {
    "1": "Enlarged Cardiomediastinum",
    "2": "Pneumothorax",
    "3": "Consolidation",
    "4": "Pneumonia",
    "5": "Edema",
    "6": "Cardiomegaly",
    "7": "Atelectasis",
    "8": "Lung Opacity",
    "9": "Pleural Effusion",
    "10": "No Finding",
}
# 官方中文名，按规程原文记录，仅供报告使用
LABEL_NAMES_ZH = {
    "1": "纵隔心影增大",
    "2": "气胸",
    "3": "肺部实变",
    "4": "肺炎",
    "5": "肺水肿",
    "6": "心影扩大",
    "7": "肺不张",
    "8": "肺部阴影",
    "9": "胸腔积液",
    "10": "无异常发现",
}


@dataclass
class TrainConfig:
    arch: str = "convnext_tiny"
    img_size: int = 384
    cache_size: int = 448
    epochs: int = 12
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    label_smooth: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0
    drop_path: float = 0.1
    final_norm: bool = False    # retain pretrained ConvNeXt post-pooling LayerNorm
    grad_clip: float = 1.0
    ema_decay: float = 0.0
    workers: int = 8
    seed: int = 42
    fold: str = "train"          # "train" -> train.csv / valid.csv
    merge_valid: bool = False    # after model selection: retrain on train+valid
    init_ckpt: str = ""          # warm-start from a checkpoint (e.g. 384 -> 448 fine-tune)
    max_iters: int = 0           # 0 = train by epochs; >0 = stop after N optimizer steps
    limit_train: int = 0         # 0 = use all; >0 = stratified subsample (smoke tests)
    stratify: bool = True        # stratify the subsample by rarest positive label
    tag: str = ""                # label used in the run directory name
    out_dir: str = ""
    amp: str = "bf16"
    eval_every: int = 1
    pos_weight_power: float = 0.0   # 0 = plain BCE, 1 = fully balanced pos_weight
    aux_view_weight: float = 0.0    # >0 adds a multi-task view-position head (regulariser)
    n_aux: int = 5                  # AP / PA / LATERAL / LL / missing
    oversample_view: str = ""       # 对指定视图过采样（如 AP，让模型多看便携片）
    oversample_factor: float = 2.0   # 过采样倍数（2.0 = AP 图出现两次）
    chronic_min: int = 0            # >0 时，对“该患者检查数 >= chronic_min”的 study 过采样
    chronic_factor: float = 1.0     # 过采样倍数（6.0 = 这批 study 额外重复 5 遍）
    view_filter: str = ""          # 只在指定视图上训练（如 AP，做便携片专家模型）
    use_view_input: bool = False    # feed the one-hot acquisition view into the head
    jpg_cache: bool = False
    distill_csv: str = ""           # 老师软标签 CSV（Dicom_id + p1..p10），空=不蒸馏
    distill_weight: float = 0.0     # 0=纯硬标签，0.5=软硬各半
    loss: str = "bce"              # bce | focal | asl
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05
    layer_decay: float = 1.0        # 骨干逐层学习率衰减（1.0=关闭）
    rank_weight: float = 0.0         # 成对排序损失权重（直接优化 AUC）
    any_aux_weight: float = 0.0      # "有无异常"辅助头权重
    view_consistency: float = 0.0   # 同 study 两视角预测一致性权重（0=关闭）
    patient_smooth: float = 0.0     # 标签向患者均值混合的权重（0=关闭）
    patient_smooth_c1: float = -1.0 # >=0: 类1 单独使用该平滑系数（-1=跟随 patient_smooth）
    head_type: str = "lse"         # lse | query（类查询注意力头）
    lse_alpha: float = 0.0            # >0: 混合 LSE 空间池化（ChestX-ray8 式）；0=纯全局平均
    lse_tau: float = 1.0
    extra_channel: str = ""         # ""=仅灰度；"asym"=增加 |I−fliplr(I)| 通道（零初始化）
    label_blend_48: float = 0.0     # >0: 训练目标在类4↔类8之间互混（0=关闭）
    init_weights: str = ""         # 用外部主干权重初始化（只替换 backbone，分类头保持本次随机初始化）
    cache_dir: str = ""             # 换预处理变体缓存目录（空=默认 cache/）         # 从 cache_jpg/ 读 JPEG（远程低带宽场景）
    fold_csv: str = ""              # 外部折表(Study_id,fold)；给了就用它代替 cv_folds
    train_folds: str = "0,1,2"
    valid_fold: int = 3
    cv_folds: int = 0               # >1: patient-grouped k-fold on train.csv
    cv_fold: int = 0                # which fold is the holdout
    hflip: float = 0.5
    rot_deg: float = 7.0
    scale_min: float = 0.75
    brightness: float = 0.2
    contrast: float = 0.2
    rbc_prob: float = 0.25          # random brightness/contrast
    erase_prob: float = 0.25
    norm: str = "per_image"         # per_image | fixed
    extra: dict = field(default_factory=dict)
