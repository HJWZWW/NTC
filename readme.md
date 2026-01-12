# NTC：Nonlinear Transform Coding
本代码库复现了 Balle 在 2018 年发表在 ICLR 上的经典论文：*Variational Image Compression with a Scale Hyperprior*，实现用神经网络作为非线性变换进行图像压缩的全流程。

## 训练
执行下面的命令

```
python train_Hyperprior.py
```

## 测试
执行下面的命令
```
python eval_Hyperprior.py
```

## 配置

在 `config/` 目录下存放了 NTC 的 `yaml` 格式的配置文件。内部定义了诸多超参数，包括数据集、权衡系数 $\lambda$、训练轮数、学习率、网络尺寸、是否导入预训练模型等等，可以按需自行修改。

## 模型
所有的模型代码存放在 `model/` 目录下。`Hyperprior.py` 文件里是 NTC 的主干模型；`analysis_transform.py` 和 `synthesis_transform.py` 里分别是 NTC 的分析变换和综合变换的实现，本代码库采用卷积神经网络，与原论文保持一致，但也可以替换成 Transformer 等更复杂的 backbone；`layer.py` 里存放用到的一些必要的基本神经网络模块和层的实现，用于搭建模型时直接调用。

## 工具函数
所有的工具函数存放在 `utils/utils.py` 内。

## 度量指标

目前仅支持 PSNR（Peak Signal-to-Noise Ratio，峰值信噪比）作为图像质量的评价指标，用于训练和测试。可以根据需求，扩展加入 MS-SSIM、LPIPS 等指标，可以在 `loss/distiontion.py` 文件内进行扩展。

## 学习文档
关于 NTC 和 NTSCC 的入门教程已经放在 `doc/` 目录下，之后可能会随时更新。
