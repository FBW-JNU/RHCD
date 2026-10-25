# RHCD

RHCD 通过逐轮超分辨率将多幅秘密图像嵌入不同分辨率的载体，利用 SA-MoE 融合载体与秘密图像，并通过跨层一致性监督实现多秘密图像的分层恢复。三轮设置采用 `64→96→144→216` 的尺寸链，秘密图像的原生尺寸为 `64/96/144`。

## 训练

以下命令均在 `RHCD/` 目录执行。可通过 `environment.yml` 创建运行环境：

```bash
conda env create -f environment.yml
conda activate rhcd
```

准备训练和验证图像清单：

```bash
python scripts/prepare_data.py \
  --train-dir /path/to/train --val-dir /path/to/val \
  --rounds 3 --output data/custom
```

启动三轮主网络的端到端训练，联合优化嵌入、秘密恢复和相邻层一致性：

```bash
python -u train.py --config config/three_round.yaml \
  data_root data/custom train_gpu '[0]' save_path runs/three_round
```

主网络训练完成后，启动 RD 降采样模块训练：

```bash
python -u train_rd.py --config config/three_round.yaml \
  --checkpoint runs/three_round/model/best.pth --output runs/rd \
  data_root data/custom train_gpu '[0]'
```

模型保存在运行目录的 `model/` 中；`best.pth` 为验证集选出的权重，`last.pth` 为最近一次保存的训练状态。从同结构模型初始化时，可增加 `weight /path/to/model.pth`；主网络续训时增加 `resume runs/three_round/model/last.pth`。

## 预训练模型

三轮预训练模型：[下载权重](https://res.masanli.top/RHCD_three_round_weights.zip)。

下载后，将主模型 `rhcd.pth` 和降采样模型 `rd.pth` 放入 `checkpoints/three_round/`。

## 测试

`test/test_3loop.py` 提供多轮嵌入与秘密图像提取。

```bash
python test/test_3loop.py embed \
  --checkpoint checkpoints/three_round/rhcd.pth \
  --cover /path/cover.png --secrets /path/s1.png /path/s2.png /path/s3.png \
  --device cuda:0 --output outputs/embed

python test/test_3loop.py extract \
  --checkpoint checkpoints/three_round/rhcd.pth \
  --rd-checkpoint checkpoints/three_round/rd.pth \
  --stego outputs/embed/stego_3.npy --carrier-level 3 \
  --device cuda:0 --output outputs/extract
```

提取默认输出当前层及低层秘密。可用 `--levels 3` 仅提取第三层秘密，或用 `--route stepwise` 进行逐级降采样恢复。输出包括 `.npy` 浮点载体、恢复结果、PNG 预览与 `run.json`；后续提取使用 `.npy` 载体。
