# LaGoVAD-PreVAD

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{liu2025lagovad,
  title={Language-guided Open-world Video Anomaly Detection under Weak Supervision},
  author={Liu, Zihao and Wu, Xiaoyu and Wu, Jianqin and Wang, Xuxu and Yang, Linlin},
  journal={arXiv preprint arXiv:2503.13160},
  year={2025}
}
```

## 2. To install the environment, please run the following script:
```shell
bash scripts/install.sh
```

## 3. To download the dataset, please run the following script:
```shell
bash scripts/download_dataset.sh
```

## 4. To download the pretrained weight, please run the following script:
```shell
bash scripts/download_weight.sh
```

## 5. To train, test, and infer the model for the PreVAD dataset, please run the following scripts:
```shell
bash scripts/train.sh
bash scripts/test.sh
bash scripts/infer.sh
```

## 6. Acknowledgement
* [Kamino666/LaGoVAD-PreVAD](https://github.com/Kamino666/LaGoVAD-PreVAD)
