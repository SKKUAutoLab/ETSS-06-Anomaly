# Tramba

![arch](assets/arch.jpg)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@ARTICLE{qiu2025tramba,
  title={Salient Object Detection in Traffic Scene Through the TSOD10K Dataset},
  author={Qiu, Yu and Sun, Yuhang and Mei, Jie and Xiao, Lin and Xu, Jing},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```

## 2. To install the environment, run the following script:
```shell
bash scripts/install.sh
```

## 3. To download the dataset, run the following script:
```shell
bash scripts/download_dataset.sh
```

## 4. To download pretrained weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 5. To train and test the model for the TSOD10K dataset, run the following scripts:
```shell
bash scripts/train.sh
bash scripts/test.sh
```

## 6. Acknowledgement
* [mj129/Tramba](https://github.com/mj129/Tramba)
