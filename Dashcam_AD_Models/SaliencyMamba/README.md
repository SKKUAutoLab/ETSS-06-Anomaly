# SaliencyMamba

![arch](assets/arch.png)

## 1. Introduction

<!-- [ALGORITHM] -->

```BibTeX
@article{zhao2025salmamba, 
  title={SalM²: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention}, 
  volume={39}, 
  DOI={10.1609/aaai.v39i2.32157},  
  number={2},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Zhao, Chunyu and Mu, Wentao and Zhou, Xian and Liu, Wenbo and Yan, Fei and Deng, Tao}, 
  year={2025}, 
  month={Apr.}, 
  pages={1647-1655} 
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

## 4. To download weights, run the following script:
```shell
bash scripts/download_weights.sh
```

## 5. To train, test, and visualize the model for the TrafficGaze dataset, run the following scripts:
```shell
bash scripts/train.sh
bash scripts/test.sh
bash scripts/vis.sh
```

## 6. Acknowledgement
* [zhao-chunyu/SaliencyMamba](https://github.com/zhao-chunyu/SaliencyMamba)
