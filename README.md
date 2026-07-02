### Automation Lab, Sungkyunkwan University

# ETSS-06: Anomaly Detection

This is the official repository of 

**OpenAnomaly: An Open Source Implementation of Anomaly Detection Methods.**

## 1. Setup
### 1.1. Using conda
#### 1.1.1. Using environment.yml
```bash
conda env create -f envs/environment.yml
conda activate anomaly
pip install -e .
```

#### 1.1.2. Using requirements.txt
```bash
conda create --name anomaly python=3.10.12
conda activate anomaly
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r envs/requirements.txt
pip install -e .
```

### 1.2. Using uv
```bash
uv venv anomaly --python=3.10.12
source anomaly/bin/activate
uv pip install -e .
```

## 2. Dataset Preparation
### 2.1. Industry Anomaly Detection Datasets
For the MVTec dataset, please download it from this [link](https://www.mvtec.com/company/research/datasets/mvtec-ad)

For the BTAD dataset, please download it from this [repository](https://github.com/pankajmishra000/VT-ADL)

For the VisA dataset, please download it from this [repository](https://github.com/amazon-science/spot-diff)

For the MVTec-Loco dataset, please download it from this [link](https://www.mvtec.com/company/research/datasets/mvtec-loco)

For the MPDD dataset, please download it from this [repository](https://github.com/stepanje/MPDD)

For the DTD dataset, please download it from this [link](https://www.robots.ox.ac.uk/~vgg/data/dtd/)

For the DAGM dataset, please download it from this [link](https://www.kaggle.com/datasets/mhskjelvareid/dagm-2007-competition-dataset-optical-inspection)

For the WFDD dataset, please download it from this [repo](https://github.com/cqylunlun/GLASS)

For the Real-IAD dataset, please download it from this [link](https://huggingface.co/datasets/Real-IAD/Real-IAD)

For the MVTec3D dataset, please download it from this [link](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad)

For the Eyecandies dataset, please download it from this [link](https://eyecan-ai.github.io/eyecandies/download)

For the MadSim dataset, please download it from this [repository](https://github.com/EricLee0224/PAD)

For the Anomaly-ShapeNet dataset, please download it from this [repository](https://github.com/Chopper-233/Anomaly-ShapeNet)

For the Real3D-AD dataset, please download it from this [repository](https://github.com/M-3LAB/Real3D-AD)

For the PD-REAL dataset, please download it from this [repository](https://github.com/Andy-cs008/PD-REAL)

For the BrokenChairs dataset, please download it from this [repository](https://github.com/VICO-UoE/Looking3D)

For the ToysAD-8K and PartsAD-15K datasets, please download it from this [repository](https://github.com/VICO-UoE/OddOneOutAD)

For the ELPV dataset, please download it from this [repository](https://github.com/zae-bayern/elpv-dataset)

For the SDD dataset, please download it from this [link](https://www.vicos.si/resources/kolektorsdd/)

For the AITEX dataset, please download it from this [link](https://www.kaggle.com/datasets/nexuswho/aitex-fabric-image-database/data)

For the Brain MRI dataset, please download it from this [link](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection?select=yes)

For the Head CT dataset, please download it from this [link](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)

For the Chest X-rays and OCT datasets, please download it from this [link](https://data.mendeley.com/datasets/rscbjbr9sj/2)

For the ISIC2018 dataset, please download it from this [link](https://challenge.isic-archive.com/data/#2018)

For the Br35H dataset, please download it from this [link](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

For the KSDD2 dataset, please download it from this [link](https://www.vicos.si/resources/kolektorsdd2/)

for the MTD dataset, please download it from this [repository](https://github.com/abin24/Magnetic-tile-defect-datasets.)

For the MulSen-AD dataset, please download it from this [repository](https://github.com/zzzbbbzzz/mulsen-ad)

For the 3CAD dataset, please download it from this [repository](https://github.com/EnquanYang2022/3CAD)

For the OCT2017 dataset, please download it from this [link](https://data.mendeley.com/datasets/rscbjbr9sj/3)

For the APTOS dataset, please download it from this [link](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data)

For the SensumSODF dataset, please download it from this [link](https://www.sensum.eu/sensumsodf-dataset/)

For the infra 3DALv2 dataset, please download it from this [repository](https://github.com/Jingyixiong/3D-Multi-FPFHI)

For the VAD dataset, please download it from this [link](https://drive.google.com/file/d/1LbHHJHCdkvhzVqekAIRdWjBWaBHxPjuu/view)

For the ContinualAD dataset, please download it from this [link](https://huggingface.co/datasets/Continual-Mega/Continual-Mega-Neurips2025)

For the GoodsAD dataet, please download it from this [repository](https://github.com/jianzhang96/GoodsAD)

For the M2AD dataset, please download it from this [link](https://huggingface.co/datasets/ChengYuQi99/M2AD)

For the MVTec-2K dataset, please download it from this [repository](https://github.com/cnulab/HiAD)

For the ITDD dataset, please download it from this [repository](https://github.com/cqylunlun/CRAS)

For the MIP dataset, please download it from this [repository](https://github.com/Kaichen-Yang/piad_baseline)

For the Kvasir-SEG dataset, please download it from this [link](https://figshare.com/articles/figure/Polyp_DataSet_zip/21221579)

For the MMAD dataset, please download it from this [repository](https://github.com/jam-cc/mmad)

For the MTD dataset, please download it from this [repository](https://github.com/abin24/Magnetic-tile-defect-datasets)

For LiverCT, RESC, HIS, ChestXray, and OCT17 datasets, please download them from this [repository](https://github.com/FantasticGNU/UniVAD)

For the MiniShift dataset, please download it from this [repository](https://github.com/hustCYQ/MiniShift-Simple3D)

For the MVTecQA dataset, please download it from this [repository](https://github.com/amoreZgx1n/SAGE)

For the BraTs2018 dataset, please download it from this [link](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

For the ToysAD-8K dataset, please download it from this [repository](https://github.com/VICO-UoE/OddOneOutAD)

For the 3D-ADAM dataset, please download it from this [repository](https://github.com/longkukuhi/RAD)

For the RESC dataset, please download it from this [repository](https://github.com/DorisBao/BMAD)

For the VID-AD dataset, please download it from this [repository](https://github.com/nkthiroto/VID-AD)

### 2.2. Unsupervised Anomaly Detection Datasets
For UCSD Ped2, CUHK Avenue, and ShanghaiTech datasets, please download them from this [repository](https://github.com/aseuteurideu/STEAL)

For the PAB dataset, please download it from this [repository](https://github.com/Shuyu-XJTU/CMP)

For the HumanSAM dataset, please download it from this [repository](https://github.com/dejian-lc/humansam)

For the Vad-R1 dataset, please download it from this [repository](https://github.com/wbfwonderful/Vad-R1)

For the LayoutAD dataset, please download it from this [repository](https://github.com/IMU-Group/LayoutAD)

For the Synthetic dataset, please download it from this [repository](https://github.com/MaticFuc/AnomalyVFM)

### 2.3. Video Anomaly Detection Datasets
For the ShanghaiTech dataset, please download extracted features from this [repository](https://github.com/tianyu0207/RTFM).

For the UCF-Crime dataset, please download extracted features from this [repository](https://github.com/carolchenyx/MGFN.) or [repo](https://github.com/Roc-Ng/DeepMIL).

For the UCF-Traffic dataset, please download extracted features from this [repository](https://github.com/VFWm614/TA-NET)

For the XD-Violence dataset, please download extracted features from this [link](https://roc-ng.github.io/XD-Violence/).

For the TAD dataset, please download extracted features from this [repository](https://github.com/ktr-hubrt/WSAL)

For the UCA dataset, please download it from this [repository](https://github.com/Xuange923/Surveillance-Video-Understanding)

For the PreVAD dataset, please download it from this [repository](https://github.com/Kamino666/LaGoVAD-PreVAD)

### 2.4. Dashcam Traffic Anomaly Detection Datasets
For ROL and DoTA datasets, please download them from this [repository](https://github.com/monjurulkarim/risky_object)

For CCD, DAD, and A3D datasets, please download them from this [repository](https://github.com/Cogito2012/UString)

For the DADA2000 dataset, please download them from this [repository](https://github.com/JWFangit/LOTVS-DADA)

For the W3DA dataset, please download it from this [repository](https://github.com/yuchen2199/Explainable-Driver-Attention-Prediction)

For the VRU dataset, please download it from this [repository](https://github.com/Kimyounggun99/VRU-Accident)

For the RoadSocial dataset, please download it from this [repository](https://github.com/roadsocial/roadsocial)

For the TrafficGaze dataset, please download it from this [repository](https://github.com/zhao-chunyu/SaliencyMamba)

For the MASKER_MD dataset, please download it from this [repository](https://github.com/ispc-lab/GSC)

For the DADA-Seg dataset, please download it from this [repository](https://github.com/jamycheung/ISSAFE)

For the DADA-1000 dataset, please download it from this [repository](https://github.com/jwfanggit/lotvs-cap)

For the CUVA dataset, please download it from this [repository](https://github.com/fesvhtr/CUVA)

For the Russia Crash dataset, please download it from this [link](https://www.kaggle.com/datasets/sivoha/car-crash-dataset-russia-2022-2023)

For the TSOD10K dataset, please download it from this [link](https://github.com/mj129/Tramba)

For the MM-AU dataset, please download it from this [link](https://github.com/jeffreychou777/LOTVS-MM-AU)

For the CrashChat dataset, please download it from this [repository](https://github.com/Liangkd/CrashChat)

For the AV-TAU dataset, please download it from this [repository](https://github.com/HarryHsing/EchoTraffic)

For the nexar-collision-prediction dataset, please download it from this [link](https://www.kaggle.com/competitions/nexar-collision-prediction)

### 2.5. Surveillance Traffic Anomaly Detection Datasets
For the TUMTrafficQA dataset, please download it from this [repository](https://github.com/TraffiX-VideoQA/TraffiX-Qwen)

For the WWC dataset, please download it from this [repository](https://github.com/VICA-Lab-HKUST-GZ/WWC-Predictor)

For the ACCIDENT dataset, please download it from this [link](https://www.kaggle.com/datasets/picekl/accident)

## 3. Usage
### 3.1.1. Supported Models for 2D Industrial Anomaly Detection
| Models         | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD               | WFDD | Real-IAD | CIFAR-10           | Fashion MNIST      | KSDD2              | SensumSODF         | ContinualAD        | EIDIs-SOD          | CrackSeg9k         | ZJU-Leaper         | M2AD               | MVTec-2K           | APTOS              | Kvasir-SEG         | MTD                | BraTs2018          | RESC                | VID-AD             |
|----------------|--------------------|--------------------|--------------------|--------------------|--------------------|------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|---------------------|--------------------|
| CutPaste       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SSAPS          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| DRAEM          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SPR            | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| ADShift        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| FOD            | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| PatchSVDD      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| CPR            | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| FAIR           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| GLASS          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SimpleNet      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SegAD          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| CDO            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SPD            | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| OnlineInReaCh  | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| FUN-AD         | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| PBAS           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| MSAD           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| PANDA          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| InReaCh        | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| REB            | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| HETMM          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| SuperSimpleNet | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| Continual-Mega | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| WPFormer       | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| PADNet         | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| M2AD           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| LWinNN         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| ComAD          | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| HiAD           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| PatchGuard     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| UniNet         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                 | :x:                |
| AnomalyNCD     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                 | :x:                |
| COBRA          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| IPAD           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| TailedCore     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| GLCF           | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| TFA-Net        | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| ADL4VAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| VarAD          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| CostFilter-AD  | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| CLAD           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| AnyAD          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                 | :x:                |
| GCR            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| AR             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark:  | :x:                |
| VID-AD         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :heavy_check_mark: |
| FSR            | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| ASBench        | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
| LogiCo         | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :z:                | :x:                | :x:                | :x:                | :x:                | :x:                 | :x:                |
 
### 3.1.2. Supported Models for 3D Industrial Anomaly Detection
| Models                 | MVTec3D            | Eyecandies         | MadSim             | Anomaly-ShapNet    | Real3D-AD          | BrokenChairs | PD-Real | MulSen-AD          | infra 3DALv2       | MIP                | ToysAD-8K          |
|------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------|---------|--------------------|--------------------|--------------------|--------------------|
| BTF                    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CFM                    | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CPMF                   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Looking3D              | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| M3DM                   | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PAD                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Shape-Guided           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| SplatPose              | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CRD                    | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CMDIAD                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MulSen-AD              | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :heavy_check_mark: | :x:                | :x:                | :x:                |
| FPFHI                  | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :heavy_check_mark: | :x:                | :x:                |
| EasyNet                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Real3D-AD              | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| 3DSR                   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PO3AD                  | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MC4AD                  | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| R3D-AD                 | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| C3DAD                  | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Reg2Inv                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MVR                    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MC3D-AD                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PIAD                   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :heavy_check_mark: | :x:                |
| GLFM                   | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| ISMP                   | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| SplatPosePlus          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CIF                    | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PASDF                  | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| RIF                    | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CASL                   | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Odd-One-Out            | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :heavy_check_mark: |
| BridgeNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| GS-CLIP                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CMDR-IAD               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| BTP-3DAD               | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| FIND                   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CM3D-AD                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MuSc-V2                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Shape-Anomaly-Codebook | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| IB-IUMAD               | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| AF3AD                  | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |

### 3.1.3. Supported Models for Anomaly Generative Industrial Anomaly Detection
| Models          | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD | MVTec3D            | Fractals           |
|-----------------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|--------------------|--------------------|
| Defect Spectrum | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                |
| DualAnoDiff     | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                |
| MAGIC           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                |
| SeaS            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :heavy_check_mark: | :x:                |
| AnoStyler       | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                |
| O2MAG           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :x:                |
| fractal4AD      | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                | :heavy_check_mark: |

### 3.1.4. Supported Models for Diffusion Industrial Anomaly Detection
| Models           | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD | MVTec3D            |
|------------------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|--------------------|
| AnoDDPM          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| D3AD             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DDAD             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiffAD           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| DiffusionAD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| GLAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| RealNet          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:      | :x:                |
| AnomalyDiffusion | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| TransFusion      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :heavy_check_mark: |
| AnoGen           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| InvAD            | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| MDPS             | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| ReplayCAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| DGDM             | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| AnomalyAny       | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| DeCo-Diff        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| CDAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |
| VLMDiff          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| CCAD             | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| C3FGS            | :x:                | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :x:                |

### 3.1.5. Supported Models for Knowledge Distillation Industrial Anomaly Detection
| Models       | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD | WFDD | Real-IAD | 3CAD               | APTOS              | OCT2017            | ISIC2018           | VAD                |
|--------------|--------------------|--------------------|--------------------|------------|------|------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|
| DeSTSeg      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| DMAD         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| EfficientAD  | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| IKD          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| MemKD        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| MixedTeacher | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| MKD          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| RD           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| RD++         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| STFPM        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| DAF          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| URD          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| 3CAD         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| ReContrast   | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:      | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| SK-RD4AD     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| SCRD4AD      | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| AAND         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |
| SingleNet    | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.1.6. Supported Models for Large Language Models Industrial Anomaly Detection
| Models     | MVTec              | VisA               | MMAD               | MVTecQA            |
|------------|--------------------|--------------------|--------------------|--------------------|
| AnomalyGPT | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| KAG-prompt | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| Echo       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| MMAD       | :x:                | :x:                | :heavy_check_mark: | :x:                |
| SAGE       | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.1.7. Supported Models for Memory Bank Industrial Anomaly Detection
| Models       | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD | WFDD | Real-IAD | 3D-ADAM |
|--------------|--------------------|--------------------|--------------------|------------|------|------|----------|---------|
| CFA          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| MemSeg       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| PaDiM        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| PatchCore    | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| SPADE        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| SFRAD        | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| SA-PatchCore | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |
| RAD          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:      | :x:     |
| RAID         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:  | :x:  | :x:      | :x:     |
| MH-PatchCore | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:  | :x:  | :x:      | :x:     |

### 3.1.8. Supported Models for Multi-class Industrial Anomaly Detection
| Models       | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD           | MVTec3D            | ITDD               | MMAD               |
|--------------|--------------------|--------------------|--------------------|------------|--------------------|------|--------------------|--------------------|--------------------|--------------------|
| CRAD         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| Real-IAD     | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| UniAD        | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| HVQ          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| MSTAD        | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| MambaAD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                | :x:                |
| ViTAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                | :x:                |
| InvAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                | :x:                |
| Dinomaly     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| RLR          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| UniFormaly   | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| OneNIP       | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| IUF          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| MoEAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| PMAD         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| UniAS        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| Omni-AD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| LGC          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| RAS          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| Wave-MambaAD | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CRAS         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| AnomalyMoE   | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :x:                |
| MVAD         | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                | :x:                |
| UniMMAD      | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.1.9. Supported Models for Noisy Industrial Anomaly Detection
| Models    | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|-----------|--------------------|--------------------|------|------------|------|------|----------|
| SoftPatch | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.10. Supported Models for Normalizing Flows Industrial Anomaly Detection
| Models      | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD | WFDD | Real-IAD | MVTec 3D           |
|-------------|--------------------|--------------------|--------------------|--------------------|------|------|----------|--------------------|
| AST         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      | :heavy_check_mark: |
| BGAD        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |
| CFLOW-AD    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |
| DifferNet   | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |
| HGAD        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:  | :x:  | :x:      | :x:                |
| MSFlow      | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |
| PyramidFlow | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |
| DADF        | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:  | :x:  | :x:      | :x:                |
| DeCoFlow    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                |

### 3.1.11. Supported Models for Out-of-Distribution Industrial Anomaly Detection
| Models    | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD | WFDD | Real-IAD | Chest X-rays       | OCT                | ISIC2018           | Br35H              | UCF101             | HMDB51             | EPIC-Kitchens      | Kinetics-600       | VOC2017            | COCO2017           | CAMELYON17         | APTOS              |
|-----------|--------------------|--------------------|--------------------|--------------------|------|------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| ADShift   | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GeneralAD | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CKAAD     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:  | :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MultiOOD  | :x:                | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| OLN-SSOS  | :x:                | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| VOS       | :x:                | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                |
| RobustND  | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| FiCo      | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:  | :x:  | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.1.12. Our Industrial Anomaly Detection Models
| Models     | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD | WFDD               | Real-IAD |
|------------|--------------------|--------------------|--------------------|------------|------|--------------------|----------|
| **DM-GRD** | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:  | :heavy_check_mark: | :x:      |

### 3.1.13. Supported Models for Segment Anything Industrial Anomaly Detection
| Models  | MVTec              | BTAD | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|---------|--------------------|------|------|------------|------|------|----------|
| UCAD    | :heavy_check_mark: | :x:  | :x:  | :x:        | :x:  | :x:  | :x:      |
| SALAD   | :heavy_check_mark: | :x:  | :x:  | :x:        | :x:  | :x:  | :x:      |
| SAM-SPT | :heavy_check_mark: | :x:  | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.14. Supported Models for Supervised Industrial Anomaly Detection
| Models | MVTec              | BTAD               | VisA | MVTec LOCO | MPDD | WFDD | Real-IAD |
|--------|--------------------|--------------------|------|------------|------|------|----------|
| DevNet | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |
| PRNet  | :heavy_check_mark: | :heavy_check_mark: | :x:  | :x:        | :x:  | :x:  | :x:      |
| DRA    | :heavy_check_mark: | :x:                | :x:  | :x:        | :x:  | :x:  | :x:      |

### 3.1.15. Supported Models for Low-shot Industrial Anomaly Detection
| Models              | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD               | WFDD               | Real-IAD | ELPV               | SDD                | AITEX              | BrainMRI           | HeadCT             | DAGM               | DTD                | Br35H              | ISIC               | ColonDB            | ClinicDB           | TN3K               | Real3D             | Eyecandies         | MVTec3D            | MVTec-FS           | KSDD2              | MTD                | GoodsAD            | LiverCT            | RESC               | HIS                | ChestXray          | OCT17              | Synthetic          |
|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------| -------------------|
| WinCLIP             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PromptAD            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| InCTRL              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyCLIP         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| APRIL-GAN           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaCLIP             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PointAD             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AMI-Net             | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ResAD               | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MVP-PCLIP           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VCP-PCLIP           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnoVL               | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MVREC               | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Segment-Any-Anomaly | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CLIP-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GPT-4V              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FADE                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyDINO         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FiLo                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MetaUAS             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FoundAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| NAGL                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CoDeGraph           | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AD-DINOv3           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ReMP-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FAPrompt            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RareCLIP            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MuSc                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DictAS              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Bayes-PFL           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MultiADS            | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AF-CLIP             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TPS                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RegAD               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AA-CLIP             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| INP-Former          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LogSAD              | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| IIPAD               | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LAFT                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniVAD              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| Anomagic            | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaptCLIP           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PCSNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DIVAD               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TF-IDG              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniADC              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Tipsomaly           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MRAD                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SubspaceAD          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| D24FAD              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VisualAD            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GATE-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyVFM          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| MoECLIP             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CoPS                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Res2CLIP            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FB-CLIP             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CMAD                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.1.16. Supported Models for Test-Time Adaptation Industrial Anomaly Detection
| Models  | MVTec              |
|---------|--------------------|
| TopoTTA | :heavy_check_mark: |

### 3.2. Supported Models for Dashcam Traffic Anomaly Detection
| Models        | ROL                | DoTA               | CCD                | DAD                | A3D                | CTA                | DADA-2000          | W3DA               | VRU                | RoadSocial         | TrafficGaze        | MASKER_MD          | DADA-Seg           | DADA-1000          | CUVA               | Russia Crash       | TSOD10K            | MM-AU              | CrashChat          | AV-TAU             | nexar-collision-prediction |
|---------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------------------------|
| AMNet         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| Baseline GRU  | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| DSTA          | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| UString       | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| XAI-Accident  | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| CTA           | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| TTHF          | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| W3DA          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| RARE          | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| BADAS         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| DriveCLIP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| DriveCLIP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| PromptTAD     | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| RoadSocial    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| Hawk          | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| CGNAM         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| SaliencyMamba | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| DRIVE         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| GSC           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| MMUDA         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| LOTVS-CAP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| LOTVS-CAAD    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| Graph-Graph   | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| CRASH         | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| CUVA          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                        |
| Ctrl-Crash    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                        |
| Tramba        | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                        |
| LOTVS-MM-AU   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                        |
| CrashChat     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                        |
| EchoTraffic   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                        |
| RiskProp      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark:         |

### 3.3. Supported Models for Unsupervised Anomaly Detection
| Models                      | Avenue             | ShanghaiTech       | Ped2               | Arbitrary Video    | PAB                | HumanSAM           | Vad-R1             | LayoutAD           | Cityscapes         |
|-----------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 3DNet                       | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| GMM_DAE                     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Jigsaw-VAD                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| OGAM-MRAM                   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SwinAnomaly                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| GenerateAnomaliesFromNormal | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| F2LM                        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VADMamba                    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AED-MAE                     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CMP                         | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| DDL                         | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| HumanSAM                    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| TAO                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Vad-R1                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| LayoutAD                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| PixOOD                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

### 3.4. Supported Models for Surveillance Traffic Anomaly Detection
| Models        | UCF-Traffic        | TAD                | TSP6K              | SO-TAD             | TUMTrafficQA       | WWC | ACCIDENT           |
|---------------|--------------------|--------------------|--------------------|--------------------|--------------------|-----|--------------------|
| TA-NET        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x: | :x:                |
| TSP6K         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x: | :x:                |
| SO-TAD        | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x: | :x:                |
| TraffiX-Qwen  | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x: | :x:                |
| WWC-Predictor | :x:                | :x:                | :x:                | :x:                | :x:                | :x: | :x:                |
| ACCIDENT      | :x:                | :x:                | :x:                | :x:                | :x:                | :x: | :heavy_check_mark: |
| SynCrash      | :x:                | :x:                | :x:                | :x:                | :x:                | :x: | :heavy_check_mark: |

### 3.5. Supported Models for Weakly-supervised Anomaly Detection
| Models         | Ten crops          | Flatten crops      | UCF-Crime          | XD-Violence        | ShanghaiTech       | MSAD               | ECA9               | GTA-Crime          | UCA                | PreVAD             |
|----------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| MIL            | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| RTFM           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| WSAL           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| GCN            | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| MGFN           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| HyperVD        | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| UR-DMU         | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| BN-WVAD        | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| CMA-LA         | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| ARNet          | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| MyModel        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| MyModel1       | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| MSAD           | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| SIAVC          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| IEF-VAD        | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VadCLIP        | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GTA-Crime      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| AnomalyCLIP    | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| LAVAD          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UCA            | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| DSANet         | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LaGoVAD-PreVAD | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.6. Supported Models for Feature Extraction
| Models                        | RGB                | Point Cloud        | Depth              | Text               | Mask               | Optical Flow       | Audio              | Stereo Matching    | TIR                | Video              | Orient             | Feature Matching   | Surface Normal     | Edge               |
|-------------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| C3D                           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| I3D                           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Dense Trajectory              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Foreground Mask               | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| HoF                           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| HoG                           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Motion Boundary               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Motion Magnitude              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TVL1                          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Depth Anything                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Depth Anything V2             | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DepthCLIP                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LapDepth                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MegaDepth                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DepthFM                       | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Depth Pro                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Amodal Depth Anything         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Video Depth Anything          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PatchRefiner                  | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MiDaS                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Structure-Guided Ranking Loss | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Distill-Any-Depth             | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| supdepth4thermal              | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RollingDepth                  | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniDepth                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TIDE                          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MoGe                          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MODEST                        | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FocusOnDepth                  | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PromptDA                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GeometryCrafter               | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniK3D                        | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Murre                         | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VGGT                          | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LiteVGGT                      | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RobustVGGT                    | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| StreamVGGT                    | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Prior-Depth-Anything          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FlashDepth                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DepthAnything-AC              | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MonSter                       | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DORNet                        | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DepthSplat                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DCL                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Monodepth                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Monodepth2                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PWC-Net                       | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Lotus                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Lotus-2                       | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Marigold-DC                   | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DAC                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DA-2                          | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| GeoWizard                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| MVInverse                     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| Pixel-Perfect                 | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| HQ-SAM                        | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaBins                       | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| BoRe-Depth                    | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Depth-Anything-3              | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DAP                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PanDA                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Genfocus                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DKT                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DA360                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LingBot-Depth                 | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PatchRefinerV2                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DVD                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnySplat                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| InfiniDepth                   | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| NVPanoptix-3D                 | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniDAC                        | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GemDepth                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CLIP                          | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LongCLIP                      | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MobileCLIP                    | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| OpenCLIP                      | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TinyCLIP                      | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RWKV-CLIP                     | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FastVLM                       | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LeGrad                        | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Perception Models             | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Spatial-CLIP                  | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Segment-Anything              | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Segment-Anything-v2           | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SegmentAnyRGBD                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| mm-sam                        | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| cat-sam                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TinySAM                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CLIPSeg                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RobustSAM                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DIS                           | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| U-2-Net                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| BASNet                        | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Point-E                       | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VGGT-Long                     | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Utonia                        | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| EVF-SAM                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Grounded-SAM-1                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Grounded-SAM-2                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LENS                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| EdgeTAM                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UnSAMv2                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAM3                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAM3D                         | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAMWISE                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DiffusionVAS                  | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Semantic-SAM                  | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CutLER                        | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TokenCut                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAM3-DMS                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| OpenSeeD                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GeoMotion                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Fast-SAM-3D-Body              | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| INSID3                        | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FSS-SAM3                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Falcon-Perception             | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FlowDIS                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| InstructSAM                   | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AlphaCLIP                     | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RAM                           | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Marigold                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ZoeDepth                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DepthCrafter                  | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnyDepth                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| NeuFlow v2                    | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RAFT                          | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Flow-Anything                 | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GMFlow                        | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniMatch                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FastFlowNet                   | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FlowNet                       | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ReCoVEr                       | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| WAFT                          | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DPT                           | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FastSAM                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MobileSAM                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Segmenter                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UISE                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| YOSO                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| OpenWorldSAM                  | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| X-Decoder                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniPixel                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SEEM                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Ref-SAM3D                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VideoMaMa                     | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RESAnything                   | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAM4MLLM                      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAMTok                        | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| gen2seg                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DecAF                         | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Fast-SAM3D                    | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SAAS                          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CoT-RVS                       | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ImageBind                     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FoundationStereo              | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ZeroStereo                    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| S2M2                          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GGEV                          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| StereoAnyVideo                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Fast-FoundationStereo         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LiteAnyStereo                 | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| sRGB-TIR                      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Video Features                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| Orient-Anything               | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Orient-Anything-V2            | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| LiftFeat                      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| MINIMA                        | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| EDM                           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| RoMa                          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| RoMav2                        | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| OmniGlue                      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| MASA                          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| MATCHA                        | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| L2M                           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| DSINE                         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| NormalCrafter                 | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| StableNormal                  | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| TransNormal                   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| RoSE                          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| PiDiNet                       | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| SAM-Audio                     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.7. Supported Models for AI City Challenge
| Models   | Iowa DOT           |
|----------|--------------------|
| CETCVLAB | :heavy_check_mark: |
| SIS_Lab  | :heavy_check_mark: |

### 3.8. Supported Models for Extension Research
<div align="center">
    <b>Architectures</b>
</div>
<table align="center">
    <tbody>
        <tr align="center" valign="bottom">
            <td>
                <b>Attention Modules</b>
            </td>
            <td>
                <b>Convolution Architectures</b>
            </td>
            <td>
                <b>Generative Models</b>
            </td>
            <td>
                <b>Geometric Models</b>
            </td>
            <td>
                <b>Losses</b>
            </td>
            <td>
                <b>Token Mixer Models</b>
            </td>
            <td>
                <b>Optimizer</b>
            </td>
            <td>
                <b>Transformer Models</b>
            </td>
            <td>
                <b>UNet Models</b>
            </td>
            <td>
                <b>3D Reconstruction Models</b>
            </td>
            <td>
                <b>MoE Models</b>
            </td>
            <td>
                <b>Mamba Models</b>
            </td>
            <td>
                <b>KAN Models</b>
            </td>
            <td>
                <b>Implicit Neural Representations Models</b>
            </td>
            <td>
                <b>Low-rank Decomposition Models</b>
            </td>
            <td>
                <b>Neural Operator Models</b>
            </td>
            <td>
                <b>Spiking Neural Networks/Event camera Models</b>
            </td>
            <td>
                <b>Efficient Models</b>
            </td>
            <td>
                <b>Vision Language Models</b>
            </td>
            <td>
                <b>Neural ODE Models</b>
            </td>
            <td>
                <b>OT Models</b>
            </td>
            <td>
                <b>Flow Matching Models</b>
            </td>
            <td>
                <b>Visualization Models</b>
            </td>
            <td>
                <b>Image Captioning Models</b>
            </td>
            <td>
                <b>Knowledge Distillation Models</b>
            </td>
            <td>
                <b>Domain Generalization Models</b>
            </td>
            <td>
                <b>Data Augmentation Models</b>
            </td>
            <td>
                <b>Object Detection Models</b>
            </td>
            <td>
                <b>Physics Models</b>
            </td>
            <td>
                <b>Representation Learning Models</b>
            </td>
            <td>
                <b>RWKV Models</b>
            </td>
            <td>
                <b>Referring Expression Models</b>
            </td>
            <td>
                <b>Scene Graph Generation Models</b>
            </td>
            <td>
                <b>World Models</b>
            </td>
            <td>
                <b>Hopfield Models</b>
            </td>
            <td>
                <b>Brain-inspired Models</b>
            </td>
            <td>
                <b>Image Editing Models</b>
            </td>
            <td>
                <b>Image Similarity Models</b>
            </td>
            <td>
                <b>Active Learning Models</b>
            </td>
            <td>
                <b>Upsample Models</b>
            </td>
            <td>
                <b>Autoregressive Models</b>
            </td>
            <td>
                <b>CSZL Models</b>
            </td>
            <td>
                <b>Segmentation Models</b>
            </td>
            <td>
                <b>In-Context Learning Models</b>
            </td>
            <td>
                <b>Continual Learning Models</b>
            </td>
            <td>
                <b>Toolbox</b>
            </td>
            <td>
                <b>Energy Models</b>
            </td>
            <td>
                <b>Reinforcement Learning Models</b>
            </td>
            <td>
                <b>Quantum Models</b>
            </td>
            <td>
                <b>Test-Time Training Models</b>
            </td>
            <td>
                <b>Superpixel Models</b>
            </td>
        <tr valign="top">
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Attention_Modules/A2">A2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/ACmix">ACmix Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/AFT">AFT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Agent-Attention">Agent-Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/AoA">Attention on Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Axial">Axial Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Axial Attention">Axial Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/BAM">BAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/BidirectionalCrossAttention">Bidirectional Cross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CBAM">CBAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CoAtNet">CoAtNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CompositionalAttention">Compositional Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Coord">Coord Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CoT">CoT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CrissCross">CrissCross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Crossformer">Crossformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/DANet">DANet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/DAT">DAT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/DeformableAttention">Deformable Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/ECA">ECA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/EMSA">EMSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/External">External Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/FlashTransformer">Flash Transformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/GFNet">GFNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/GSA">GSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Halo">Halo Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Hamburger">Hamburger Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/KroneckerAttention">Kronecker Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/LambdaNet">LambdaNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Linformer">Linformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/LocalAttention">Local Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MOATransformer">MOA Transformer</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MobileViT">MobileViT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MobileViTv2">MobileViTv2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MUSE">MUSE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Nonlocal">Non-local Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/NystromAttention">Nystrom Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Outlook">Outlook Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/ParNet">ParNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/PolarizedSelfAttention">PolarizedSelf Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/PSA">PSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Residual">Residual Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/S2">S2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SCSE">SCSE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SelfAttention">Self-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SGE">SGE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Shuffle">Shuffle Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SimAM">SimAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SimplifiedSelfAttention">Simplified Self-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SK">SK Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SlotAttention">Slot Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/TaylorLinearAttention">Taylor Linear Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Triplet">Triplet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/UFO">UFO Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/ViP">ViP Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/RingAttention">Ring Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/HierarchicalAttention">Hierarchical Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/RectifiedLinearAttention">Rectified Linear Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/STAM">Space Time Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/HaloAttentiono">Halo Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CoordinateDescentAttention">Coordinate Descent Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MemoryCompressedAttention">Memory Compressed Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Mega">Mega Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Tranception">Tranception Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/InfiniAttention">Infini Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/SparseAttention">Sparse Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MultimodalCrossAttention">Multi-modal Cross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/AttentionConvolution">Attention with Convolution</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MoA">MoA</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/LinearAttentionTransformer">LinearAttentionTransformer</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Performer">Performer</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/ISAB">ISAB</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MemoryEfficientAttention">MemoryEfficientAttention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/Perceiver">Perceiver</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CoLT5-attention">CoLT5-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/GatedSlotAttention">GatedSlotAttention</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/JVP">JVP</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/CBSA">CBSA</a></li>
                <li><a href="Extension_Research_Models/Attention_Modules/MCA">MCA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Convolution_Architectures/DynamicConvolution">Dynamic Convolution</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/WTConv">Wavelet Convolution</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/InvertibleResidualNetworks">Invertible Residual Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/FrEIA">FrEIA</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/FFF">Fast Feedforward Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/GLOM">GLOM</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Firefly">Firefly</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/BEM">Bayesian Enhancement Model</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/mincam">Minimalist Vision with Freeform Pixels</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/AKOrN">AKOrN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Any-Precision-DNNs">Any-Precision Deep Neural Network</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/FDConv">FDConv</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/OverLoCK">OverLoCK</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/ConvNeXt">ConvNeXt</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/HSNet">HSNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/homeostatic-neural-networks">Homeostatic Neural Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/VAT">VAT</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/NC-Net">NC-Net</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/ANCNet">ANCNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/CVNet">CVNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/ConverseNet">ConverseNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/ConvNeXt-V2">ConvNeXt-V2</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/FFC">FFC</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Contextual-Convolutional-Networks">Contextual-Convolutional-Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/CoConv">CoConv</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/LogViG">LogViG</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Optical-FCNN">Optical-FCNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/FDAM">FDAM</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/SN-Net">SN-Net</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Nested-Learning">Nested-Learning</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/CycleMLP">CycleMLP</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/SimSiam">SimSiam</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/TRL">TRL</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/DDL">DDL</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/V4D">V4D</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/BayesianCNN">BayesianCNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/F-Conv">F-Conv</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/SPCNN">SPCNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/BayesNCL">BayesNCL</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Tversky">Tversky</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/SMT">SMT</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/ExpoMotion">ExpoMotion</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/WONN">WONN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/HNN">HNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/KomplexNet">KomplexNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_Architectures/Metriplector">Metriplector</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Generative_Models/EMA">EMA</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/VectorQuantization">Vector Quantization</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/VQ-VAE">VQ-VAE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/TiTok">TiTok</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiT">DiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Fast-DiT">Fast DiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DTR">DTR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/ANT">ANT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/R3GAN">R3GAN</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DALL-E">DALL-E</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/VAR">VAR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CleanDIFT">CleanDIFT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Vec2Vec">Vec2Vec</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FoD">FoD</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/taming-transformers">taming-transformers</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiS">DiS</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/MAR">MAR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiCo">DiCo</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Tweedie">Tweedie</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DDPM">DDPM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Guided-Diffusion">Guided-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Improved-Diffusion">Improved-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FSDDIM">FSDDIM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DDIM">DDIM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/RectifiedFlow">RectifiedFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/GMFlow">GMFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/ScoreSDE">ScoreSDE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FlowGrad">FlowGrad</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/InstaFlow">InstaFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/EDTR">EDTR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/TorchCFM">TorchCFM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/RAE">RAE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CSBM">CSBM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/REPA">REPA</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/InDI">InDI</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/U-ViT">U-ViT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiffMoE">DiffMoE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/MDM">MDM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/JiT">JiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DDT">DDT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/SiT">SiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FractalGen">FractalGen</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CoDe">CoDe</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DAAM">DAAM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Glance">Glance</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/ViBT">ViBT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/TREAD">TREAD</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/MaskDiT">MaskDiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/EDM">EDM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/ToMeSD">ToMeSD</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Open-Sora">Open-Sora</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/REPA-E">REPA-E</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LearningWithNoise">LearningWithNoise</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/S-NODE">S-NODE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LTX-Video">LTX-Video</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/MobileI2V">MobileI2V</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Next-Patch-Prediction">Next-Patch-Prediction</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DimensionX">DimensionX</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Cosmos-Predict2.5">Cosmos-Predict2.5</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DeepVerse">DeepVerse</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Cosmos-Transfer2.5">Cosmos-Transfer2.5</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Wan2.2">Wan2.2</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CogVideo">CogVideo</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LTX-2">LTX-2</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiT-Extrapolation">DiT-Extrapolation</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Self-Refining">Self-Refining</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/UniVideo">UniVideo</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FlashWorld">FlashWorld</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/PixelGen">PixelGen</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FastVMT">FastVMT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/REPA-G">REPA-G</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DummyForcing">DummyForcing</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CSBM">CSBM</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/MVAR">MVAR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/EAR">EAR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Drifting">Drifting</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LongLive">LongLive</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Self-Forcing">Self-Forcing</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Driftin">Driftin</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FlowCache">FlowCache</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LayerSync">LayerSync</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Spacetime-Diffusion">Spacetime-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Mobile-O">Mobile-O</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/RollingSink">RollingSink</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/CHORDS">CHORDS</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Helios">Helios</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DiffiT">DiffiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/WAE">WAE</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/FD-Loss">FD-Loss</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/GWF">GWF</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/DecQ">DecQ</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/RectifiedHR">RectifiedHR</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Semanticist">Semanticist</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/PixelDiT">PixelDiT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Quant-VideoGen">Quant-VideoGen</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Dynamical-Regimes-of-Diffusion">Dynamical-Regimes-of-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/ControlNet">ControlNet</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/LayoutDiffusion">LayoutDiffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Un-0">Un-0</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/OrientationDiffusion">OrientationDiffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/GAT">GAT</a></li>
                <li><a href="Extension_Research_Models/Generative_Models/Gaussian-VAE">Gaussian-VAE</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Geometric_Models/geoopt">geoopt</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HCL">HCL</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyperbolicCV">Hyperbolic CV</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyperbolicEmbedding">Hyperbolic Embedding</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyperbolicVisionTransformer">Hyperbolic Vision Transformer</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/meru">meru</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HCL">HCL</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/PoincareResnet">Poincare Resnet</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypMix">HypMix</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CurvatureGeneration">Curvature Generation</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CO-SNE">CO-SNE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypAD">HypAD</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HVT">HVT</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HiHPQ">HiHPQ</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/GM-VAE">GM-VAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/GHSW">GHSW</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/DOHSC-DO2HSC">DOHSC-DO2HSC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/GW-Regularization">GW-Regularization</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hypformer">Hypformer</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HAE">HAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HYSP">HYSP</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HR-MLP">HR-MLP</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/hierahyp">hierahyp</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypLiLoc">HypLiLoc</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/UnHyperML">UnHyperML</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypStructure">HypStructure</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HierarchyCPCC">HierarchyCPCC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CIDER">CIDER</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HYPE">HYPE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/DMT-HI">DMT-HI</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HHCH">HHCH</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/hypvl">hypvl</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/hycoclip">hycoclip</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyProMeta">HyProMeta</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/LResNet">LResNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/hyperbolic_wrapped_distribution">Hyperbolic Wrapped Distribution</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/pvae">pvae</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/DistilVPR">DistilVPR</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypHC">HypHC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/L3DMC">L3DMC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/GPr-Net">GPr-Net</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/longtail_segmentation">Longtail Segmentation</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hyperbolic_Feature_Augmentation">Hyperbolic Feature Augmentation</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/hyperfuture">hyperfuture</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CurAML">CurAML</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/BREEDS">BREEDS</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/OTCPCC">OTCPCC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HPEC">HPEC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/QCNN">QCNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CLIP-M3">CLIP-M3</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hyp-OW">Hyp-OW</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/AngularGap">AngularGap</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/vMF">vMF</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Rotation_Trick">Rotation_Trick</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HySAC">HySAC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/QRNN">QRNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/QCLNet">QCLNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Quaternion-Capsule-Networks">Quaternion-Capsule-Networks</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/QGAN">QGAN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HSN">HSN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyCoRe">HyCoRe</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HLFormer">HLFormer</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HOVER">HOVER</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyperCLIP">HyperCLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypCD">HypCD</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HS-DTE">HS-DTE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypGCD">HypGCD</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/H-CLIP">H-CLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/ATMG">ATMG</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Geo-Sign">Geo-Sign</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HAMN">HAMN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HexFormer">HexFormer</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/PHyCLIP">PHyCLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hyperbolic_Interpretability">Hyperbolic_Interpretability</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HIVE">HIVE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HoroPCA">HoroPCA</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypModalAlign">HypModalAlign</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypPNet">HypPNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HIDISC">HIDISC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HDD">HDD</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypDAE">HypDAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyperRL">HyperRL</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Graph-Prompt">Graph-Prompt</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HypLL">HypLL</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hyperbolic-Vim">Hyperbolic-Vim</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/phier">phier</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/MGIoU">MGIoU</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Hyperbolic_Action">Hyperbolic_Action</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/CQH">CQH</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/WMDD">WMDD</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HybCBC">HybCBC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HBNN">HBNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/UNCHA">UNCHA</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HAC">HAC</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Fractal-GNN">Fractal-GNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HyRo">HyRo</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/TEDFusion">TEDFusion</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/Meridian">Meridian</a></li>
                <li><a href="Extension_Research_Models/Geometric_Models/HTC">HTC</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Losses/OT-CLIP">OT CLIP</a></li>
                <li><a href="Extension_Research_Models/Losses/GradNorm">GradNorm</a></li>
                <li><a href="Extension_Research_Models/Losses/Class-Balanced-Loss">Class-Balanced-Loss</a></li>
                <li><a href="Extension_Research_Models/Losses/InfoNCE">InfoNCE</a></li>
                <li><a href="Extension_Research_Models/Losses/LabelAwareRanked-Loss">LabelAwareRanked-Loss</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Token_Mixer_Models/gMLP">gMLP</a></li>
                <li><a href="Extension_Research_Models/Token_Mixer_Models/MLP-Mixer">MLP Mixer</a></li>
                <li><a href="Extension_Research_Models/Token_Mixer_Models/Segformer">Segformer</a></li>
                <li><a href="Extension_Research_Models/Token_Mixer_Models/ResMLP">ResMLP</a></li>
                <li><a href="Extension_Research_Models/Token_Mixer_Models/ActiveMLP">ActiveMLP</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Optimizer/AdamAtan2">Adam Atan 2</a></li>
                <li><a href="Extension_Research_Models/Optimizer/AdaMod">AdaMod</a></li>
                <li><a href="Extension_Research_Models/Optimizer/Adan">Adan</a></li>
                <li><a href="Extension_Research_Models/Optimizer/Lion">Lion</a></li>
                <li><a href="Extension_Research_Models/Optimizer/GradientAscent">Gradient Ascent</a></li>
                <li><a href="Extension_Research_Models/Optimizer/DecoupledLionW">Decoupled LionW</a></li>
                <li><a href="Extension_Research_Models/Optimizer/GradientEquillibrum">Gradient Equillibrum</a></li>
                <li><a href="Extension_Research_Models/Optimizer/CV-Model-Compression">Computer Vision Model Compression Techniques</a></li>
                <li><a href="Extension_Research_Models/Optimizer/Grokfast">Grokfast</a></li>
                <li><a href="Extension_Research_Models/Optimizer/AdEMAMix">AdEMAMix</a></li>
                <li><a href="Extension_Research_Models/Optimizer/SlimAdamW">SlimAdamW</a></li>
                <li><a href="Extension_Research_Models/Optimizer/CAME">CAME</a></li>
                <li><a href="Extension_Research_Models/Optimizer/Sophia">Sophia</a></li>
                <li><a href="Extension_Research_Models/Optimizer/VRADAM">VRADAM</a></li>
                <li><a href="Extension_Research_Models/Optimizer/NL_Optimizers">NL_Optimizers</a></li>
                <li><a href="Extension_Research_Models/Optimizer/PRM">PRM</a></li>
                <li><a href="Extension_Research_Models/Optimizer/IO_Adam">IO_Adam</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Transformer_Models/RotaryEmbedding">Rotary Embedding</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/TNT">Transformer in Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SplineTransformers">Spline-based Transformers</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DyT">DyT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Diff-Transformer">Diff-Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ELN">ELN</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MAE">MAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/PVT">PVT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Vision_GNN">Vision_GNN</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINO">DINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINOv2">DINOv2</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/RC-MAE">RC-MAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/VGP">VGP</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/EoMT">EoMT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SwinIRDehaze">SwinIRDehaze</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Twins">Twins</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Deformable-DETR">Deformable-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DETR">DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ConditionalDETR">ConditionalDETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MetaFormer">MetaFormer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/HcNet">HcNet</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINOv3">DINOv3</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/HyperMAE">HyperMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/RoPE-ViT">RoPE-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SparK">SparK</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SBT">SBT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/PH-Reg">PH-Reg</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/JAFAR">JAFAR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MultiMAE">MultiMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Vision-LSTM">Vision-LSTM</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DinoUNet">DinoUNet</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SegDINO">SegDINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MedDINOv3">MedDINOv3</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/REOrder">REOrder</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/LIT">LIT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MedicalMAE">MedicalMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Attention-Transfer">Attention-Transfer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Test-Time-Register">Test-Time-Register</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/PolaFormer">PolaFormer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/BRIXEL">BRIXEL</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ThinkingViT">ThinkingViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Franca">Franca</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SMCA-DETR">SMCA-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ChangeDINO">ChangeDINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/CRATE">CRATE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DReX">DReX</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/RADIO">RADIO</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/CARE_Transformer">CARE_Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/FGTS">FGTS</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SnapViT">SnapViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/FractalDB">FractalDB</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Derf">Derf</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/POA">POA</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Pixio">Pixio</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/NEPA">NEPA</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Co2S">Co2S</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ALViT">ALViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DAB-DETR">DAB-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MDETR">MDETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DN-DETR">DN-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MobileViT">MobileViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SSMAE">SSMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/CAS-ViT">CAS-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/EdgeNeXt">EdgeNeXt</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/LoRA-DETR">LoRA-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DIFFSSR">DIFFSSR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ViT-Plasticity">ViT-Plasticity</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINOv3_Animal">DINOv3_Animal</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/AdaptOVCD">AdaptOVCD</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Raptor">Raptor</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ViT-5">ViT-5</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MLLA">MLLA</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/LAST-ViT">LAST-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DiT4SR">DiT4SR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/LocAtViT">LocAtViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/FastDINOv2">FastDINOv2</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/BinaryAttention">BinaryAttention</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/AdapterTune">AdapterTune</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/HAViT">HAViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/PlaneCycle">PlaneCycle</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Procedural_Warmup_ViT">Procedural_Warmup_ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/EUPE">EUPE</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/PeftCD">PeftCD</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ProM3E">ProM3E</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SAPIENS2">SAPIENS2</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINOv3-IML">DINOv3-IML</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/NeXt2Former-CD">NeXt2Former-CD</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MM-DINO">MM-DINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINO_Soars">DINO_Soars</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/GramSR">GramSR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DMD">DMD</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/SVL">SVL</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/KST">KST</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/UniRefiner">UniRefiner</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MS-DETR">MS-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MDS-DETR">MDS-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/MaskSub">MaskSub</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DVANet">DVANet</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DINO-Med3D">DINO-Med3D</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/VIOLIN">VIOLIN</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DnA">DnA</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/Circulant-Attention">Circulant-Attention</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/DiffRate">DiffRate</a></li>
                <li><a href="Extension_Research_Models/Transformer_Models/ToMe">ToMe</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/UNet_Models/Uformer">Uformer</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/ReversibleUnet">Reversible U-Net</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/X-UNet">X-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/MIMO-UNet">MIMO-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/UNetv2">UNetv2</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/SAM3-UNet">SAM3-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/SAM2-UNet">SAM2-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/SAM-UNet">SAM-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/SAM2-UNeXT">SAM2-UNeXT</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/Hyperbolic-U-Net">Hyperbolic-U-Net</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/xLSTM-UNet">xLSTM-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_Models/DINOv3-UNet">DINOv3-UNet</a></li>
            </ul>
        </td>
         <td>
            <ul>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/2D_Gaussian_Splatting">2D Gaussian Splatting</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/3D_Gaussian_Splatting">3D Gaussian Splatting</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/iNeRF">iNeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/NeRF">NeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SyncDreamer">SyncDreamer</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/D2GV">D2GV</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GaussianImage">GaussianImage</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Easi3R">Easi3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/VGGSfM">VGGSfM</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DUSt3R">DUSt3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Fast3R">Fast3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MBPR">MBPR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GroupRec">GroupRec</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/PromptHMR">PromptHMR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/CrowdRec">CrowdRec</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/2DGS_Inpaint">2DGS_Inpaint</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GSASR">GSASR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/ContinuousSR">ContinuousSR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DeblurGS">DeblurGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Deblur-NeRF">Deblur-NeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/BAD-Gaussians">BAD-Gaussians</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/CLIPNeRF">CLIPNeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/CLIP2NeRF">CLIP2NeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/nerfstudio">nerfstudio</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SpikeNeRF">SpikeNeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/gsplat">gsplat</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/NerfAcc">NerfAcc</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Human3R">Human3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/4DGaussians">4DGaussians</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MapAnything">MapAnything</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/UFM">UFM</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MuM">MuM</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SAM-Body4D">SAM-Body4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/HSMR">HSMR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Sharp">Sharp</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Multi-View-Foundation-Models">Multi-View-Foundation-Models</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Pi3">Pi3</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DVGT-1">DVGT-1</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/EgoX">EgoX</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/PedGen">PedGen</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/VGGT4D">VGGT4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/InfiniteVGGT">InfiniteVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DroneSplat">DroneSplat</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/WorldGen">WorldGen</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LiftImage3D">LiftImage3D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MASt3R">MASt3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Gen3R">Gen3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/UniSH">UniSH</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/instant-ngp">instant-ngp</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/EasyVolcap">EasyVolcap</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/4K4D">4K4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/V-DPM">V-DPM</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Geo4D">Geo4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MonST3R">MonST3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MVGGT">MVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LongSplat">LongSplat</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GaussianSR">GaussianSR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Instant-GI">Instant-GI</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LIG">LIG</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/IGGT">IGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/VeGaS">VeGaS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GSVC">GSVC</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/4DGS">4DGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GaussianToken">GaussianToken</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/D-NeRF">D-NeRF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Deformable-3D-Gaussians">Deformable-3D-Gaussians</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SAT-HMR">SAT-HMR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/FastGS">FastGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DashGaussian">DashGaussian</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/FastVGGT">FastVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Shape-of-Motion">Shape-of-Motion</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Deblur4DGS">Deblur4DGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Any4D">Any4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/JOSH">JOSH</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MotionCrafter">MotionCrafter</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/und3rstand">und3rstand</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GaussianImage++">GaussianImage++</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/ODE-GS">ODE-GS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LangSplat">LangSplat</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LangSplatV2">LangSplatV2</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/4DLangSplat">4DLangSplat</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/UBS">UBS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/OmniVGGT">OmniVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/AMB3R">AMB3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Flow3r">Flow3r</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/CUT3R">CUT3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GT2-GS">GT2-GS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MoE-GS">MoE-GS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SGI">SGI</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LoGeR">LoGeR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Mobile-GS">Mobile-GS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LASER">LASER</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/NanoGS">NanoGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MatryoshkaGaussianSplatting">MatryoshkaGaussianSplatting</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/QuantVGGT">QuantVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/PAGE4D">PAGE4D</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/DVGT-2">DVGT-2</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/GLD">GLD</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Node-RF">Node-RF</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/4RC">4RC</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/LL-Gaussian">LL-Gaussian</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/PanoVGGT">PanoVGGT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Stream3R">Stream3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SAGS">SAGS</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MegaSaM">MegaSaM</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/VGGT-Omega">VGGT-Omega</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/CoMHR">CoMHR</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Lite3R">Lite3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/PDI-Bench">PDI-Bench</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SelfEvo">SelfEvo</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/VGG-TTT">VGG-TTT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Scal3R">Scal3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/R3">R3</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/ViGeo">ViGeo</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SurGe">SurGe</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/MAD">MAD</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/D4RT">D4RT</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/TTT3R">TTT3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Multi-HMR-2">Multi-HMR-2</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/SphericalVoronoi">SphericalVoronoi</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Trust3R">Trust3R</a></li>
                <li><a href="Extension_Research_Models/3D_Reconstruction_Models/Splat_Analyzer">Splat_Analyzer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-CNN">MoE CNN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Soft-MoE">Soft MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ViT-Soft-MoE">ViT Soft MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/PEER">PEER</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ST-MoE">ST MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Sparse-MoE">Sparse MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/LIMoE">LIMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoD">MoD</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Mamba">MoE Mamba</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MHMoE">Multi-Head MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoNE">MoNE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Fusion">MoE-Fusion</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE">MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SinkhornRouter">Sinkhorn Router</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/BlackMamba">BlackMamba</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoU">MoU</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Switch-MoE">Switch MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DiT-MoE">DiT MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/TC-MoA">TC-MoA</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Tutel">Tutel</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoSE">MoSE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/M4oE">M4oE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/NID">NID</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/FAME">FAME</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MVMoE">MVMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Switch-DiT">Switch-DiT</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ScatterMoE">ScatterMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SFAN">SFAN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SMoE">SMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DeepMoE">DeepMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/GMoE">GMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoFME">MoFME</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ShiftAddViT">ShiftAddViT</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoA">MoA</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE_PromptCL">MoE-PromptCL</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/CRMoE">CRMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Jetpack">MoE Jetpack</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/muMoE">muMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/CLIP-MoE">CLIP-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoH">MoH</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Adapters4CL">MoE-Adapters4CL</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DAMEX">DAMEX</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Neural-Experts">Neural-Experts</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ConsistentMoE">ConsistentMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Mod-Squad">Mod-Squad</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DMP">DMP</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/AMIR">AMIR</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Efficient-WEMoE">Efficient-WEMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/PnD">PnD</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/weight-ensembling_MoE">weight-ensembling_MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MEGAN">MEGAN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DeMo">DeMo</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SM3Det">SM3Det</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MEASNet">MEASNet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SeemoRe">SeemoRe</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoCE-IR">MoCE-IR</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/AdaptIR">AdaptIR</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/pFedMoAP">pFedMoAP</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Earth-Adapter">Earth-Adapter</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DM-Adapter">DM-Adapter</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MicarVLMoE">MicarVLMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MFG-HMoE">MFG-HMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/QuARF">QuARF</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/LFME">LFME</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MixBridge">MixBridge</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/EMOE">EMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/LiveEdit">LiveEdit</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/beyond-param-count">Beyond Parameter Count</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/VMoBA">VMoBA</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/I2MoE">I2MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoPE">MoPE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Diff-Retinex++">Diff-Retinex++</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Inter2Former">Inter2Former</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-FFD">MoE-FFD</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/BMFH-Net">BMFH-Net</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MFRNet">MFRNet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/BIG-MoE">BIG-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Robust-MoE-CNN">Robust-MoE-CNN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/JTDMoE">JTDMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Diff-MoE">Diff-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ICL">ICL</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Face-MoGLE">Face-MoGLE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/AMoED">AMoED</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Switch-NeRF">Switch-NeRF</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/T-MoENet">T-MoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Adapters++">MoE-Adapters++</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MiN">MiN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MINGLE">MINGLE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/HotMoE">HotMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/R-MoE">R-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DynMoE">DynMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/EMoE">EMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoME">MoME</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE_CL">MoE_CL</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/AVMOE">AVMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DeMoE">DeMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SMoPE">SMoPE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/GM-MoE">GM-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SPMTrack">SPMTrack</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SFAN">SFAN</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Forensic-MoE">Forensic-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/HyMoENet">HyMoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/APMoENet">APMoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MCMoE">MCMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoMKE">MoMKE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/CoMoE">CoMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/LPMoE">LPMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/EfficientMoE">EfficientMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/STAR">STAR</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/GeoMoE">GeoMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/AMOE">AMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DMDNet">DMDNet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/D-MoLE">D-MoLE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/PA-MoE">PA-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Gazeformer">Gazeformer</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Meta-Expert">Meta-Expert</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ProMoE">ProMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoETTA">MoETTA</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/WMNet">WMNet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/DMFusion">DMFusion</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/GRAM">GRAM</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Point-MoE">Point-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/GazeMoE">GazeMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/CA-MoE">CA-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MEGADance">MEGADance</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/LE-ProtoPNet">LE-ProtoPNet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MedMoE">MedMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/M4Fuse">M4Fuse</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/StyleExpert">StyleExpert</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SpecMoE">SpecMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/VidPrism">VidPrism</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/VDMoE">VDMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/TAG-MoE">TAG-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/WEFT">WEFT</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/UAMoE">UAMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/RSDet">RSDet</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SFAM">SFAM</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoASE">MoASE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ParaX">ParaX</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/ReMoE">ReMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/NESTOR">NESTOR</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/M4-SAM">M4-SAM</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/M-CBEs">M-CBEs</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/SegMoTE">SegMoTE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/HD-DinoMoE">HD-DinoMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/FA-MoE">FA-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/Inf-Dehaze">Inf-Dehaze</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/FedDG-MoE">FedDG-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/CalibMoE">CalibMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/MoE-Confidence">MoE-Confidence</a></li>
                <li><a href="Extension_Research_Models/MoE_Models/FaceMoE">FaceMoE</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaVision">MambaVision</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/TinyViM">TinyViM</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VM-UNet">VM-UNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/TransMamba">TransMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Vim">Vim</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Vamba">Vamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MobileMamba">MobileMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/DAMamba">DAMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/CLIMB-ReID">CLIMB-ReID</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Mamba-Reg">Mamba-Reg</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaOut">MambaOut</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MAP">MAP</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/ARM">ARM</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/LFTramba">LFTramba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/PoinTramba">PoinTramba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Tran-Mamba">Tran-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Dimba">Dimba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MxT">MxT</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/HMT-Unet">HMT-Unet</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VM-UNetv2">VM-UNetv2</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/AiM">AiM</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/CWNet">CWNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/FreqMamba">FreqMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/LocalMamba">LocalMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VMamba">VMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/RetinexRawMamba">RetinexRawMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/W-Mamba">W-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VMatcher">VMatcher</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/SP-Mamba">SP-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/M3amba">M3amba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Mamba3D">Mamba3D</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/HSRMamba">HSRMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/WalMaFa">WalMaFa</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Wave-Mamba">Wave-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/IRSRMamba">IRSRMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaIR">MambaIR</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/DFSSM">DFSSM</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/FreMamba">FreMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaLLIE">MambaLLIE</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VCMamba">VCMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/V2M">V2M</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/BMTNet">BMTNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/PRISM">PRISM</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/AdaSFFuse">AdaSFFuse</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaMoE">MambaMoE</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/SpectralVMamba">SpectralVMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/EAMamba">EAMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/WDMamba">WDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/CAB">CAB</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MambaHoME">MambaHoME</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/VSSD">VSSD</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/AFM-Net">AFM-Net</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/SFMFusion">SFMFusion</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/ReIDMamba">ReIDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MatMamba">MatMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/TPCM-SegNet">TPCM-SegNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/NDMamba">NDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/MM-DETR">MM-DETR</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/TranSamba">TranSamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/ACMamba">ACMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/CMFS">CMFS</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/UVM-Net">UVM-Net</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Dino-Mamba">Dino-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/TimeViper">TimeViper</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/Mamba-3">Mamba-3</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/EQ-VMamba">EQ-VMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_Models/HCMNet">HCMNet</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/KAN_Models/EfficientKAN">Efficient KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/Vision-KAN">Vision KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/Fast-KAN">Fast KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/FourierKAN">Fourier KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/MoE-KAN">MoE KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAT">KAT</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/GR_KAN">GR_KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAN-COIN">KAN COIN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KM_UNet">KM_UNet</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/U-KAN">U-KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/MedViTV2">MedViTV2</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/RKAN">RKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/FewSound">FewSound</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAN-AD">KAN-AD</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/LDSB">LDSB</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/cmKAN">cmKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAC">KAC</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KArAt">KArAt</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/QKAN">QKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAN-IDIR">KAN-IDIR</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/JacobiKAN">JacobiKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/ChebyKAN">ChebyKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/GroupKAN">GroupKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/ABFR-KAN">ABFR-KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/FasterKAN">FasterKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/DGKAN">DGKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/ViK">ViK</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/SSTKAN">SSTKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_Models/KAConvNet">KAConvNet</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/INR_Models/siren">siren</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CoLIE">CoLIE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/WIRE">WIRE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeRCo">NeRCo</a></li>
                <li><a href="Extension_Research_Models/INR_Models/GKAN">GKAN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SCONE">SCONE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LINR">LINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FR-INR">FR-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeRD">NeRD</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INRAD">INRAD</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SineKAN">SineKAN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeRP">NeRP</a></li>
                <li><a href="Extension_Research_Models/INR_Models/MINER">MINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/O-INR">O-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/ImpSq">ImpSq</a></li>
                <li><a href="Extension_Research_Models/INR_Models/ANR">ANR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Partition-INR">Partition INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/COIN">COIN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LoE">LoE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FreSh">FreSh</a></li>
                <li><a href="Extension_Research_Models/INR_Models/RFF">RFF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FINER">FINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FINER++">FINER++</a></li>
                <li><a href="Extension_Research_Models/INR_Models/xinc">xinc</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Deblur_INR">Deblur-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Trans-INR">Trans-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/GINR-IPC">GINR-IPC</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Local-Global-INR">Local-Global-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeRV">NeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/HNeRV">HNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NerPE">NerPE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/IRL-INR">IRL-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DI-INR">DI-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LIIF">LIIF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/awesome">awesome</a></li>
                <li><a href="Extension_Research_Models/INR_Models/QIREN">QIREN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/COMBINER">COMBINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/RECOMBINER">RECOMBINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/TAF">TAF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/nir">nir</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INR-Harmonization">INR-Harmonization</a></li>
                <li><a href="Extension_Research_Models/INR_Models/BDINR">BDINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/VideoINR">VideoINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/PFNR">PFNR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DINER">DINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/TSINR">TSINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CQ-NIR">CQ-NIR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Sobolev_INRs">Sobolev_INRs</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Poly_INR">Poly_INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INR-ST">INR-ST</a></li>
                <li><a href="Extension_Research_Models/INR_Models/O2V">O2V</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SketchINR">SketchINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INR-ECGAN">INR-ECGAN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LTE">LTE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/HNET">HNET</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LTEW">LTEW</a></li>
                <li><a href="Extension_Research_Models/INR_Models/PartSDF">PartSDF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Spherical-INR">Spherical-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/UncertaINR">UncertaINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LosslessINR">LosslessINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DDiF">DDiF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SAPE">SAPE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Symmetric-Power-Transformation-INR">Symmetric-Power-Transformation-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/MWT">MWT</a></li>
                <li><a href="Extension_Research_Models/INR_Models/HiNeRV">HiNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NG">NG</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NFN">NFN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/EVOS-INR">EVOS-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INT">INT</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FAOR">FAOR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SNF">SNF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/RPP">RPP</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeuralSVG">NeuralSVG</a></li>
                <li><a href="Extension_Research_Models/INR_Models/STINR">STINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FeINFN">FeINFN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Spener">Spener</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Moner">Moner</a></li>
                <li><a href="Extension_Research_Models/INR_Models/VI3NR">VI3NR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/PhysINR">PhysINR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/IGA-INR">IGA-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/MetricGrids">MetricGrids</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeuRBF">NeuRBF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/HyperNVD">HyperNVD</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SL2A-INR">SL2A-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeRT">NeRT</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Implicit-Zoo">Implicit-Zoo</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INRFlow">INRFlow</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Transformer-NFN">Transformer-NFN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SIREN-square">SIREN-square</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INF">INF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FFN">FFN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INR-benchmark">INR-benchmark</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Residual-INR">Residual-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/STAF">STAF</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INCODE">INCODE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FFNeRV">FFNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Neural-Knitworks">Neural-Knitworks</a></li>
                <li><a href="Extension_Research_Models/INR_Models/IDIR">IDIR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/MetaSeg">MetaSeg</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CLIT">CLIT</a></li>
                <li><a href="Extension_Research_Models/INR_Models/STRAINER">STRAINER</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SUICA">SUICA</a></li>
                <li><a href="Extension_Research_Models/INR_Models/INR-Bench">INR-Bench</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NeuroQuant">NeuroQuant</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CST-Net">CST-Net</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LINR-PCGC">LINR-PCGC</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Split-Layer">Split-Layer</a></li>
                <li><a href="Extension_Research_Models/INR_Models/MFSR-CD">MFSR-CD</a></li>
                <li><a href="Extension_Research_Models/INR_Models/COIN++">COIN++</a></li>
                <li><a href="Extension_Research_Models/INR_Models/REFINE-MORE">REFINE-MORE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DINo">DINo</a></li>
                <li><a href="Extension_Research_Models/INR_Models/HUVR">HUVR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DInf-Grid">DInf-Grid</a></li>
                <li><a href="Extension_Research_Models/INR_Models/ASSR">ASSR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/LGE_INR">LGE_INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CNS_INR">CNS_INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/SIR-SNC">SIR-SNC</a></li>
                <li><a href="Extension_Research_Models/INR_Models/NINT">NINT</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CAFE">CAFE</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Lipschitz-INR">Lipschitz-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/TorchLip">TorchLip</a></li>
                <li><a href="Extension_Research_Models/INR_Models/TeCoNeRV">TeCoNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/Equivariant-ASISR">Equivariant-ASISR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/EIDGAN">EIDGAN</a></li>
                <li><a href="Extension_Research_Models/INR_Models/CF-INR">CF-INR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/RFML4MRI">RFML4MRI</a></li>
                <li><a href="Extension_Research_Models/INR_Models/TinyNeRV">TinyNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_Models/EnFold-MRI">EnFold-MRI</a></li>
                <li><a href="Extension_Research_Models/INR_Models/ISO-FNO">ISO-FNO</a></li>
                <li><a href="Extension_Research_Models/INR_Models/FLAIR">FLAIR</a></li>
                <li><a href="Extension_Research_Models/INR_Models/DD-INR">DD-INR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/LoRA">LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/PEVC">PEVC</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/MLoRE">MLoRE</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/LoRA-IR">LoRA-IR</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/CLIP-LoRA">CLIP-LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/LoME">LoME</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/MoRA">MoRA</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/DeLoRA">DeLoRA</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/ASI">ASI</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/DyLoRA-MoE">DyLoRA-MoE</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/SR-LoRA">SR-LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/LoRA-ViT">LoRA-ViT</a></li>
                <li><a href="Extension_Research_Models/Low_Rank_Decomposition_Models/BiLaLoRA">BiLaLoRA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FNO">Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/UNO">UNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/U-FNO">U-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/G-FNO">G-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/AFNO">AFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/HiNOTE">HiNOTE</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SRNO">SRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FourierImaging">Fourier Imaging</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/Galerkin">Galerkin-type Self-attention</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FFNO">Factorized Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NeuralOP">NeuralOP</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/LatentNeuralOperator">Latent Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/WNO">Wavelet Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DQFNO">Derived-Quanities Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FNOMultiSizedImages">FNO Multi Sized Images</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/AAAI_FNO_Tutorial">AAAI FNO Tutorial</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/MAWNO">MAWNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NeurOp-Diff">NeurOp-Diff</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/AMG">AMG</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/CoNO">CoNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FNOReg">FNOReg</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/CGH-SFO">CGH-SFO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/AFFNet">AFFNet</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DAFNO">DAFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/Geo-FNO">Geo-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/ResFNO">ResFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/GNOT">GNOT</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SSRNO">SSRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/INN">INN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FNIN">FNIN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/UNet-FNO">UNet-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/WDNO">WDNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SFNO">SFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/GANO">GANO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NewtonFNO">NewtonFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PITA">PITA</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DPOT">DPOT</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PINO">PINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/VINO">VINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/GKN">GKN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PAC-FNO">PAC-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PDIO">PDIO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/MambaNO">MambaNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/muTransfer-FNO">muTransfer-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/RNO">RNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/iFNO">iFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/UNOI">UNOI</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/CNO">CNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/HNOSeg-XS">HNOSeg-XS</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/JPNeO">JPNeO-XS</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NeurOLight">NeurOLight</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PastNet">PastNet</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PINN_FP64">PINN_FP64</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/TANTE">TANTE</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/Tucker-FNO">Tucker-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/AFNO">AFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DINOZAUR">DINOZAUR</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/STVSR-NO">STVSR-NO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/INO">INO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/OpFlow">OpFlow</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PODNO">PODNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/MINO">MINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/HyPINO">HyPINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/RINN">RINN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SPL_OFM">SPL_OFM</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/MoEPOT">MoEPOT</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SoFoNO">SoFoNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FNOPE">FNOPE</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/FlowGINO">FlowGINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/MMGN">MMGN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/Sonnet">Sonnet</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NeuralKoopmanSVD">NeuralKoopmanSVD</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PINN">PINN</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/PINNs-Torch">PINNs-Torch</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/R-FNO">R-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SONIC">SONIC</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/IRNO">IRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DisentangO">DisentangO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/DRIFT-Net">DRIFT-Net</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/NOIR">NOIR</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/EqGINO">EqGINO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/delNO">delNO</a></li>
                <li><a href="Extension_Research_Models/Neural_Operator_Models/SirenFNO">SirenFNO</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SpikeYOLO">SpikeYOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/EMS-YOLO">EMS-YOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SpikingYOLO">SpikingYOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/EOLO">EOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/RENet">RENet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/DSEC-MOS">DSEC-MOS</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/RVT">RVT</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/LEOD">LEOD</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SSM">SSM</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SMamba">SMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/eTraM">eTraM</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SpikingYOLOX">SpikingYOLOX</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/MvHeatDET">MvHeatDET</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/CvHeat-DET">CvHeat-DET</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/DAGr">DAGr</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SU-YOLO">SU-YOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/CREST">CREST</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/DEOE">DEOE</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/SNNTracker">SNNTracker</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/Advanced-SpikingYOLOX">Advanced-SpikingYOLOX</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Detection_Models/TDE">TDE</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/LISNN">LISNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/E-SpikeFormer">E-SpikeFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Meta-SpikeFormer">Meta-SpikeFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Spike-driven">Spike-driven</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Spikformer">Spikformer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/QKFormer">QKFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SEW_ResNet">SEW_ResNet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpikingResformer">SpikingResformer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SEMM">SEMM</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Spikingformer">Spikingformer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/CML">CML</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SGLFormer">SGLFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/NDA_SNN">NDA_SNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/MS-ResNet">MS-ResNet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SWformer">SWformer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/BKDSNN">BKDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Event_Modality_Application">Event_Modality_Application</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EventCLIP">EventCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EST">EST</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EDP">EDP</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/n_imagenet">n_imagenet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpikeCLIP">SpikeCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/NeuroCLIP">NeuroCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SLAYER">SLAYER</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/T-RevSNN">T-RevSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/GAC">GAC</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpikeVideoFormer">SpikeVideoFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpikMamba">SpikMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/ExACT">ExACT</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EventBind">EventBind</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/NVS2Graph">NVS2Graph</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/ANN2SNN_NQ">ANN2SNN_NQ</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/MambaPAR">MambaPAR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SNN-PAR">SNN-PAR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/VTB">VTB</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EventMamba">EventMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/TET">TET</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/STMixer">STMixer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/MTT">MTT</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SML">SML</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/MaxFormer">MaxFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Masked-Spiking-Transformer">Masked-Spiking-Transformer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/AT-LIF">AT-LIF</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/TRSG">TRSG</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/CKDSNN">CKDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/Mamba-Spike">Mamba-Spike</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpiLiFormer">SpiLiFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/BSDSNN">BSDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/PseudoSNN">PseudoSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/DSD">DSD</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/STOD">STOD</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SGC">SGC</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/BSD">BSD</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/DeepEISNN">DeepEISNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/EnhSNN">EnhSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/MSViT">MSViT</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/CMSF">CMSF</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/SpikeInit">SpikeInit</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Recognition_Models/TEFormer">TEFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/e2vid">e2vid</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/EvINR">EvINR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/ESDNet">ESDNet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/SpikerIR">SpikerIR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/NER">NER</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/EventGAN">EventGAN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/EHN">EHN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/EVSNN">EVSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/SpikeCLIP">SpikeCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/INR-Event-VSR">INR-Event-VSR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/PRE-Mamba">PRE-Mamba</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/DepthAnyEvent">DepthAnyEvent</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/e2depth">e2depth</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/Spike-FlowNet">Spike-FlowNet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/RSIR">RSIR</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/e2flow">e2flow</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/SpikeDiffusion">SpikeDiffusion</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/EvEnhancer">EvEnhancer</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/DehazeSNN">DehazeSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Recognition_Models/VLIF">VLIF</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/Gen1_toolbox">Gen1_toolbox</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/V2CE">V2CE</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/V2E">V2E</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/Vid2E">Vid2E</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/S2R">S2R</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/IEBCS">IEBCS</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Toolbox/I2E">I2E</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Anomaly_Detection_Models/UCF-Crime-DVS">UCF-Crime-DVS</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Anomaly_Detection_Models/ISSAFE">ISSAFE</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Anomaly_Detection_Models/EventAD">EventAD</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Segmentation_Models/Spike2Former">Spike2Former</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Segmentation_Models/SLTNet">SLTNet</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Segmentation_Models/ECOS">ECOS</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Segmentation_Models/Spiking-DeepLab">Spiking-DeepLab</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Object_Segmentation_Models/MambaSeg">MambaSeg</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Action_Recognition_Models/HARDVS">HARDVS</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Action_Recognition_Models/HARDVSv2">HARDVSv2</a></li>
                <li><a href="Extension_Research_Models/SNN_Event_Models/Point_Cloud_Models/SPM">SPM</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Efficient_Models/mincam">mincam</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/NdLinear">NdLinear</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/GlobustVP">GlobustVP</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/minGRU">minGRU</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/OFA">OFA</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileOne">MobileOne</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileIE">MobileIE</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Turbo-VAED">Turbo-VAED</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/PreciseRoIPooling">PreciseRoIPooling</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/ortho-residual">ortho-residual</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileNetV4">MobileNetV4</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileNetV3">MobileNetV3</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileNetV2">MobileNetV2</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileNetV1">MobileNetV1</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/RepViT">RepViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/EfficientFormer">EfficientFormer</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/LeViT">LeViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileViG">MobileViG</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/MobileViGv2">MobileViGv2</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/GreedyViG">GreedyViG</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Mobile-U-ViT">Mobile-U-ViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/AB-BNN">AB-BNN</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/BPAM">BPAM</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/LGM">LGM</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Lottery-Ticket-Hypothesis">Lottery-Ticket-Hypothesis</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/mHC">mHC</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Titans">Titans</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Flash-KMeans">Flash-KMeans</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/UniRepLKNet">UniRepLKNet</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/FastViT">FastViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/FasterViT">FasterViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/LowFormer">LowFormer</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/TransXNet">TransXNet</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/StarNet">StarNet</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/FasterNet">FasterNet</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/RepVGG">RepVGG</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Temperature_Scaling">Temperature_Scaling</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/NamedCurves">NamedCurves</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/LiBrA-Net">LiBrA-Net</a></li>
                <li><a href="Extension_Research_Models/Efficient_Models/Slimmable-ConvNeXt">Slimmable-ConvNeXt</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/VLM_Models/CoOp">CoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CLIP_Surgery">CLIP_Surgery</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PromptKD">PromptKD</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MCM">MCM</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CLIP-KD">CLIP-KD</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MaPLe">MaPLe</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/DPLQ">DPLQ</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/DiffCLIP">DiffCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/DualPrompt">DualPrompt</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PromptSRC">PromptSRC</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/DualCoOp">DualCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MMRL">MMRL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/Safe-CLIP">Safe-CLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CLIP-Refine">CLIP-Refine</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/Tip-Adapter">Tip-Adapter</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/TrustVLM">TrustVLM</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/Bayesian-Prompt-Learning">Bayesian-Prompt-Learning</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/Nano-vLLM">Nano-vLLM</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PointCLIP">PointCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PromptFix">PromptFix</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CAT-Seg">CAT-Seg</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/DMN">DMN</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/FCPrompt">FCPrompt</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/SurPL">SurPL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MITLOp-CCAug">MITLOp-CCAug</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/GCSCoOp">GCSCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PromptAlign">PromptAlign</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CasPL">CasPL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/TAP">TAP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/AToken">AToken</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/KgCoOp">KgCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/ProGrad">ProGrad</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/CLIPstyler">CLIPstyler</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MSAE">MSAE</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/Mobile-VideoGPT">Mobile-VideoGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/NXTP">NXTP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/minGPT">minGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/iGPT">iGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/AdaptVision">AdaptVision</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/ICL">ICL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/SuperCLIP">SuperCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/MoPD">MoPD</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PAE">PAE</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/PDA">PDA</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/TIPS">TIPS</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/KDPL">KDPL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/LOREAL">LOREAL</a></li>
                <li><a href="Extension_Research_Models/VLM_Models/RainbowPrompt">RainbowPrompt</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/CLODE">CLODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/DeepRUOT">DeepRUOT</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/NeuralODE">NeuralODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/SDE">SDE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/SNOpt">SNOpt</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/RNODE">RNODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/ANODE">ANODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/GAINS">GAINS</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/DCN">DCN</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/NODE-ImgNet">NODE-ImgNet</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/DMF">DMF</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/NODEO">NODEO</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/NODE-GP">NODE-GP</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/CLE">CLE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/ResNet_NODE">ResNet_NODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/HomoODE">HomoODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/MetaNODE">MetaNODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/Bi-flow">Bi-flow</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/HazeFlow">HazeFlow</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/DEQ">DEQ</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_Models/FLUX-IR">FLUX-IR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/OT_Models/Hyperbolic_Alignment">Hyperbolic Alignment</a></li>
                <li><a href="Extension_Research_Models/OT_Models/PLOT">PLOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DUCT">DUCT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OTFusion">OT Fusion</a></li>
                <li><a href="Extension_Research_Models/OT_Models/TransformerFusion">TransformerFusion</a></li>
                <li><a href="Extension_Research_Models/OT_Models/Intra-Fusion">Intra-Fusion</a></li>
                <li><a href="Extension_Research_Models/OT_Models/LAVCap">LAVCap</a></li>
                <li><a href="Extension_Research_Models/OT_Models/Tree-Diffusion-Schrodinger-Bridge">Tree-based Diffusion Schrödinger Bridge</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OTCE">OTCE</a></li>
                <li><a href="Extension_Research_Models/OT_Models/TSA-MLT">TSA-MLT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OT-VP">OT-VP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/LICO">LICO</a></li>
                <li><a href="Extension_Research_Models/OT_Models/SinkhornAutoDiff">SinkhornAutoDiff</a></li>
                <li><a href="Extension_Research_Models/OT_Models/FedOTP">FedOTP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/ALIGN">ALIGN</a></li>
                <li><a href="Extension_Research_Models/OT_Models/TCL">TCL</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DebiasedOT">DebiasedOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/MGPATH">MGPATH</a></li>
                <li><a href="Extension_Research_Models/OT_Models/COT">COT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/APTP">APTP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/HiWA">HiWA</a></li>
                <li><a href="Extension_Research_Models/OT_Models/MM-Align">MM-Align</a></li>
                <li><a href="Extension_Research_Models/OT_Models/AWT">AWT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/PROP">PROP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/L2RM">L2RM</a></li>
                <li><a href="Extension_Research_Models/OT_Models/KPG-RL">KPG-RL</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OSSL">OSSL</a></li>
                <li><a href="Extension_Research_Models/OT_Models/Prompt-OT">Prompt-OT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/NLPrompt">NLPrompt</a></li>
                <li><a href="Extension_Research_Models/OT_Models/AffScoreDeep">AffScoreDeep</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OTSeg">OTSeg</a></li>
                <li><a href="Extension_Research_Models/OT_Models/Sinkformers">Sinkformers</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DiffusionSchrodingerBridge">DiffusionSchrodingerBridge</a></li>
                <li><a href="Extension_Research_Models/OT_Models/LightSB-Matching">LightSB-Matching</a></li>
                <li><a href="Extension_Research_Models/OT_Models/LightSB">LightSB</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OTTER">OTTER</a></li>
                <li><a href="Extension_Research_Models/OT_Models/EntropicOTBenchmark">EntropicOTBenchmark</a></li>
                <li><a href="Extension_Research_Models/OT_Models/RAM">RAM</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DbTSW">DbTSW</a></li>
                <li><a href="Extension_Research_Models/OT_Models/ASOT">ASOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/TOT">TOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DoRA">DoRA</a></li>
                <li><a href="Extension_Research_Models/OT_Models/MFSWB">MFSWB</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DA-RCOT">DA-RCOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/TRIP">TRIP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/REMOTE">REMOTE</a></li>
                <li><a href="Extension_Research_Models/OT_Models/UniOT">UniOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/UNSB">UNSB</a></li>
                <li><a href="Extension_Research_Models/OT_Models/CUNSB-RFIE">CUNSB-RFIE</a></li>
                <li><a href="Extension_Research_Models/OT_Models/DehazeSB">DehazeSB</a></li>
                <li><a href="Extension_Research_Models/OT_Models/LaCoOT">LaCoOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/CatSBench">CatSBench</a></li>
                <li><a href="Extension_Research_Models/OT_Models/BaryIR">BaryIR</a></li>
                <li><a href="Extension_Research_Models/OT_Models/RCOT">RCOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/SOT-GLP">SOT-GLP</a></li>
                <li><a href="Extension_Research_Models/OT_Models/SOTA">SOTA</a></li>
                <li><a href="Extension_Research_Models/OT_Models/OT_DETECTOR">OT_DETECTOR</a></li>
                <li><a href="Extension_Research_Models/OT_Models/PolyStep">PolyStep</a></li>
                <li><a href="Extension_Research_Models/OT_Models/FlashSinkhorn">FlashSinkhorn</a></li>
                <li><a href="Extension_Research_Models/OT_Models/VSFOT">VSFOT</a></li>
                <li><a href="Extension_Research_Models/OT_Models/SinkhornCPD">SinkhornCPD</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/LieFlow">LieFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/FM">FM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/PnP-Flow">PnP-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/SemFlow">SemFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/Data-Uncertainty">Data-Uncertainty</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/FlowMatching">FlowMatching</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/EqM">EqM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/FFM">FFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/DeltaFM">DeltaFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/Flow-Guidance">Flow-Guidance</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/MeanFlow">MeanFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/TCCM">TCCM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/RFM">RFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/ATFM">ATFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/Restora-Flow">Restora-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/BFM">BFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/PCFM">PCFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/FERNN">FERNN</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/SoFlow">SoFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/flowNP">flowNP</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/PBFM">PBFM</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/DegFlow">DegFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/Self-Flow">Self-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/VGG-Flow">VGG-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_Matching_Models/MC-RFM">MC-RFM</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Visualization_Models/Finer-CAM">Finer-CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/VCC">VCC</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/MoXI">MoXI</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Prompt_CAM">Prompt_CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/CLIP-dissect">CLIP-dissect</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/WWW">WWW</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Transformer-Explainability">Transformer-Explainability</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/MAMMI">MAMMI</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/SIC">SIC</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/NWHead">NWHead</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/ProTeCt">ProTeCt</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/B-cos-v2">B-cos-v2</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/VPS">VPS</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/FIxLIP">FIxLIP</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Grad-Eclip">Grad-Eclip</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/exCLIP">exCLIP</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Lucent">Lucent</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/TAM">TAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/CE-FAM">CE-FAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/FG-VCE">FG-VCE</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/EPIC">EPIC</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/VX-CODE">VX-CODE</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/ConceptAttention">ConceptAttention</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/MaskInversion">MaskInversion</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Fusion-CAM">Fusion-CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/FaCT">FaCT</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/DEX-AR">DEX-AR</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/SCAN">SCAN</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/CanViT">CanViT</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/Fooling-Feature-Visualization">Fooling-Feature-Visualization</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/RefineCAM">RefineCAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/RISE">RISE</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/MHE">MHE</a></li>
                <li><a href="Extension_Research_Models/Visualization_Models/DAVE">DAVE</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/describe-anything">describe-anything</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/KGG">KGG</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/kg-gen">kg-gen</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/Cosmos-Reason2">Cosmos-Reason2</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/CogVLM2">CogVLM2</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/CogVLM">CogVLM</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/Qwen3-VL">Qwen3-VL</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/LLaVA">LLaVA</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/LLaVA-3D">LLaVA-3D</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/LLaVA-Grounding">LLaVA-Grounding</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/MiniGPT-4">MiniGPT-4</a></li>
                <li><a href="Extension_Research_Models/Image_Captioning_Models/Youtu-VL">Youtu-VL</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/mean-teacher">mean-teacher</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/MM-DistillNet">MM-DistillNet</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/CSKD">CSKD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/Manifold_Distillation">Manifold_Distillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/CMT">CMT</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/DM2TNet">DM2TNet</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/LMT-GP">LMT-GP</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/PMT">PMT</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/Stable-Mean-Teacher">Stable-Mean-Teacher</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/Self-Distillation">Self-Distillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/SDViT">SDViT</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/SLA">SLA</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/DLB">DLB</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/LSD">LSD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/TSDA">TSDA</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/S3D">S3D</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/MSDA">MSDA</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/LGD">LGD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/GeoDistill">GeoDistill</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/DualSelfDistillation">DualSelfDistillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/Dataset-Distillation">Dataset-Distillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/GLSD">GLSD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/AdaSim">AdaSim</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/MixSKD">MixSKD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/LightRA">LightRA</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/PS-KD">PS-KD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/FRAMER">FRAMER</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/TISC">TISC</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/MosaicKD">MosaicKD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/DIAS">DIAS</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/SelfPLL">SelfPLL</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/EmbeddingDistillation">EmbeddingDistillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_Distillation_Models/Bayesian-KD">Bayesian-KD</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/L2D">L2D</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DANN">DANN</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/D_GCD">D_GCD</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/PromptStyler">PromptStyler</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DPStyler">DPStyler</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DFCF">DFCF</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/PromptTA">PromptTA</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/SFDG">SFDG</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/RobustNet">RobustNet</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/ABA">ABA</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/ALT">ALT</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DDG">DDG</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/NeuralLio">NeuralLio</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/ASRNorm">ASRNorm</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/PDEN">PDEN</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/SFDA2">SFDA2</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/MJDOT">MJDOT</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/Fish">Fish</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/ZeroShotDG">ZeroShotDG</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/M-ADA">M-ADA</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/Single-DGOD">Single-DGOD</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/RandDG">RandDG</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DomED">DomED</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/RISE">RISE</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/PAKD">PAKD</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/TAF-Cal">TAF-Cal</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/SSG">SSG</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/GradNorm">GradNorm</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/PEER">PEER</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/IBN-Net">IBN-Net</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/SW">SW</a></li>
                <li><a href="Extension_Research_Models/Domain_Generalization_Models/DGvGS">DGvGS</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/AFA">AFA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/HybridAugment++">HybridAugment++</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/SPM">SPM</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/VIPAug">VIPAug</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/DCAug">DCAug</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/PhysAug">PhysAug</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/RainNet">RainNet</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/InstanceAugmentation">InstanceAugmentation</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/Polar_Augment">Polar_Augment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/NMCB">NMCB</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/FastAutoAugment">FastAutoAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/RandAugment">RandAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/Analytical-Learning-Theory">Analytical-Learning-Theory</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/On-Device-DG">On-Device-DG</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/MixStyle">MixStyle</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/XDomainMix">XDomainMix</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/RandConv">RandConv</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/SagNet">SagNet</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/C-UDA">C-UDA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/LEAwareSGD">LEAwareSGD</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/ConDiSR">ConDiSR</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/StyDeSty">StyDeSty</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/ADA">ADA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/EFDM">EFDM</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/AdaIN">AdaIN</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/CCDR">CCDR</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/FIXED">FIXED</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/OpenDG-DAML">OpenDG-DAML</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/LearnAug-UDA">LearnAug-UDA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/TeachAugment">TeachAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/TTSA">TTSA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/TSA">TSA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/DoGE">DoGE</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/MIXALL">MIXALL</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/FreeSDG">FreeSDG</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/UDA-norm-VAE">UDA-norm-VAE</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/OnlineAugment">OnlineAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/FATA">FATA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/Mixup_for_UDA">Mixup_for_UDA</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/AugMix">AugMix</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/GridMask">GridMask</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/StyleAugmentation">StyleAugmentation</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/IL2A">IL2A</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/DSU">DSU</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/SHADE">SHADE</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/QuadAug">QuadAug</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/AdvStyle">AdvStyle</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/StyleMix">StyleMix</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/HiLo">HiLo</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/Generalize-Unseen-Domains">Generalize-Unseen-Domains</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/TrivialAugment">TrivialAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/OA-CutMix">OA-CutMix</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/DCON">DCON</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/Pro-RandConv">Pro-RandConv</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/KeepAugment">KeepAugment</a></li>
                <li><a href="Extension_Research_Models/Data_Augmentation_Models/ResizeMix">ResizeMix</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Object_Detection_Models/UniFS">UniFS</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/OLN">OLN</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/SR4IR">SR4IR</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/DEIMv2">DEIMv2</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/Rex-Omni">Rex-Omni</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/AutoEval">AutoEval</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/GroundingDINO">GroundingDINO</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/Open-GroundingDino">Open-GroundingDino</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/GSDet">GSDet</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/DEQDet">DEQDet</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/YOLO-Master">YOLO-Master</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/YOLO-IOD">YOLO-IOD</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/SAM-DETR">SAM-DETR</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/WildDet3D">WildDet3D</a></li>
                <li><a href="Extension_Research_Models/Object_Detection_Models/Bridge">Bridge</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Physics_Models/PSF-Estimation">PSF-Estimation</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/MANO">MANO</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhyDNet">PhyDNet</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/vHeat">vHeat</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/CIConv">CIConv</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhyDAE">PhyDAE</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/NLOS-LPP">NLOS-LPP</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/QuantiPhy">QuantiPhy</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhysVideoGenerator">PhysVideoGenerator</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhysCtrl">PhysCtrl</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhysDreamer">PhysDreamer</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/NCLaw">NCLaw</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/NewtonGen">NewtonGen</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/STDDN">STDDN</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/Goal-Force">Goal-Force</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhysicEdit">PhysicEdit</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/RealWonder">RealWonder</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/UWEnhancer">UWEnhancer</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/CAML">CAML</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/GS_Physics">GS_Physics</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PECA">PECA</a></li>
                <li><a href="Extension_Research_Models/Physics_Models/PhysInOne">PhysInOne</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/ICon">ICon</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/MRL">MRL</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/Pigment_Representation">Pigment_Representation</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/VAC">VAC</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/RotationContinuity">RotationContinuity</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/UAE">UAE</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/Platonic">Platonic</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/KnowCoL">KnowCoL</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/Midway">Midway</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/FlowFeat">FlowFeat</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/Compositional-Generalization">Compositional-Generalization</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/SteerViT">SteerViT</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/PMLR">PMLR</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/LCH">LCH</a></li>
                <li><a href="Extension_Research_Models/Representation_Learning_Models/IAM-CL2R">IAM-CL2R</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/RWKV_Models/CRWKV">CRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/Vision-RWKV">Vision-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/Restore-RWKV">Restore-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKV-UNet">RWKV-UNet</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/VisualRWKV">VisualRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKV-IR">RWKV-IR</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/Diffusion-RWKV">Diffusion-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/LION">LION</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/OmniRWKVSR">OmniRWKVSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/URWKV">URWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/LALIC">LALIC</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/FEAT">FEAT</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/ScaleMatch">ScaleMatch</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/HFE-RWKV">HFE-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/OccRWKV">OccRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/LF-RWKVNet">LF-RWKVNet</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/EventPAR">EventPAR</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/ChangeRWKV">ChangeRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKV-VG">RWKV-VG</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RetinexRWKV">RetinexRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKVFusion">RWKVFusion</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RSRWKV">RSRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/GDSR">GDSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKVSR">RWKVSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/FRWKV">FRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/Fourier-RWKV">Fourier-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/Hi-RWKV">Hi-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/TS-Pan">TS-Pan</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/DIN-GEBC">DIN-GEBC</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RawRWKV">RawRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/DRWKV">DRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/MMSegRWKV">MMSegRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/LHIC-BP">LHIC-BP</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/LineRWKV">LineRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/PointRWKV">PointRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/RWKV-PCSSC">RWKV-PCSSC</a></li>
                <li><a href="Extension_Research_Models/RWKV_Models/SCRWKV">SCRWKV</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/RMOT">RMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/CRMOT">CRMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/TempRMOT">TempRMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/RexSeek">RexSeek</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/3EED">3EED</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/WildRefer">WildRefer</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/Dynamic-MDETR">Dynamic-MDETR</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/TransVG">TransVG</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/RefDrone">RefDrone</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/SOREC">SOREC</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/Zeroshot_REC">Zeroshot_REC</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/RefCrowd">RefCrowd</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/OmniSegNet">OmniSegNet</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/ReLA">ReLA</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/GPT4RoI">GPT4RoI</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/DKGTrack">DKGTrack</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/FlowRVS">FlowRVS</a></li>
                <li><a href="Extension_Research_Models/Referring_Expression_Models/SAM3-I">SAM3-I</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/SARD">SARD</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/EGTR">EGTR</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/USGG">USGG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/sg2im">sg2im</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Graph-RCNN">Graph-RCNN</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/OvSGTR">OvSGTR</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Seg-Grounded">Seg-Grounded</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/OpenPSG">OpenPSG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Pair-Net">Pair-Net</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/VLPrompt">VLPrompt</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/HiLo">HiLo</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Fair-PSGG">Fair-PSGG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Scene-Graph-ViT">Scene-Graph-ViT</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/REACT">REACT</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/PL">PL</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/TDE">TDE</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/PENET">PENET</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/TopoNet">TopoNet</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/T2SG">T2SG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/Salience-SGG">Salience-SGG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/SG_Risk">SG_Risk</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/HSG">HSG</a></li>
                <li><a href="Extension_Research_Models/Scene_Graph_Generation_Models/BUSSARD">BUSSARD</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/World_Models/I-JEPA">I-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/V-JEPA">V-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/V-JEPAv2">V-JEPAv2</a></li>
                <li><a href="Extension_Research_Models/World_Models/JEPA_SSL">JEPA_SSL</a></li>
                <li><a href="Extension_Research_Models/World_Models/PiLaMIM">PiLaMIM</a></li>
                <li><a href="Extension_Research_Models/World_Models/SCOTT">SCOTT</a></li>
                <li><a href="Extension_Research_Models/World_Models/D-JEPA">D-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/Point-JEPA">Point-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/CNN-JEPA">CNN-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/NWM-NWM">CNN-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/DINO-Foresight">DINO-Foresight</a></li>
                <li><a href="Extension_Research_Models/World_Models/FUTURIST">FUTURIST</a></li>
                <li><a href="Extension_Research_Models/World_Models/JEPA-T">JEPA-T</a></li>
                <li><a href="Extension_Research_Models/World_Models/seq-JEPA">seq-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/LeJEPA">LeJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/FlowJEPA">FlowJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/DMT-JEPA">DMT-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/JEPA-Physics">JEPA-Physics</a></li>
                <li><a href="Extension_Research_Models/World_Models/DSeq-JEPA">DSeq-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/TesserAct">TesserAct</a></li>
                <li><a href="Extension_Research_Models/World_Models/NanoJEPA">NanoJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/LingBot-World">LingBot-World</a></li>
                <li><a href="Extension_Research_Models/World_Models/EB_JEPA">EB_JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/Rectified_LpJEPA">Rectified_LpJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/EchoJEPA">EchoJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/Newton-Kepler">Newton-Kepler</a></li>
                <li><a href="Extension_Research_Models/World_Models/Infinite-World">Infinite-World</a></li>
                <li><a href="Extension_Research_Models/World_Models/C-JEPA">C-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/NeoVerse">NeoVerse</a></li>
                <li><a href="Extension_Research_Models/World_Models/WorldGrow">WorldGrow</a></li>
                <li><a href="Extension_Research_Models/World_Models/Social-JEPA">Social-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/BiJEPA">BiJEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/HJEPA-MoE">HJEPA-MoE</a></li>
                <li><a href="Extension_Research_Models/World_Models/SAT-JEPA-DIFF">SAT-JEPA-DIFF</a></li>
                <li><a href="Extension_Research_Models/World_Models/LeWorldModel">LeWorldModel</a></li>
                <li><a href="Extension_Research_Models/World_Models/ZWM">ZWM</a></li>
                <li><a href="Extension_Research_Models/World_Models/Sub-JEPA">Sub-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/Le-MuMo-JEPA">Le-MuMo-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/IA-JEPA">IA-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/JEPA-Guided">JEPA-Guided</a></li>
                <li><a href="Extension_Research_Models/World_Models/LeJEPA_Identifiability">LeJEPA_Identifiability</a></li>
                <li><a href="Extension_Research_Models/World_Models/VISReg">VISReg</a></li>
                <li><a href="Extension_Research_Models/World_Models/FlowPokeTransformer">FlowPokeTransformer</a></li>
                <li><a href="Extension_Research_Models/World_Models/DVD-JEPA">DVD-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_Models/S-JEPA">S-JEPA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Hopfield_Models/Hopfield-Layers">Hopfield-Layers</a></li>
                <li><a href="Extension_Research_Models/Hopfield_Models/Associative-Transformer">Associative-Transformer</a></li>
                <li><a href="Extension_Research_Models/Hopfield_Models/iMixer">iMixer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/Neurons">Neurons</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/ZEBRA">ZEBRA</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/Lp-Convolution">Lp-Convolution</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/PhiNets">PhiNets</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/PhiNetv2">PhiNetv2</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/NICE">NICE</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/Brain-inspired-Replay">Brain-inspired-Replay</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/SPM">SPM</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/Pixel-Reg">Pixel-Reg</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/NeuroBridge">NeuroBridge</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/TMNs">TMNs</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/FSPT-FSCIL">FSPT-FSCIL</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/ML-BVAE">ML-BVAE</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/BraVL">BraVL</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/IterativeVAE">IterativeVAE</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/HiCL">HiCL</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/VCFlow">VCFlow</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/NeuroPictor">NeuroPictor</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/TRIBEv2">TRIBEv2</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/HyFI">HyFI</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/UHN">UHN</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/SQHN">SQHN</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/MTG">MTG</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/LSHN">LSHN</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/DAM">DAM</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/Hopfield-Boosting">Hopfield-Boosting</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/NeuroFlow">NeuroFlow</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/EEG-MoCE">EEG-MoCE</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/WSCL">WSCL</a></li>
                <li><a href="Extension_Research_Models/Brain_Inspired_Models/SyncBrain">SyncBrain</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_Editing_Models/InstructPix2Pix">InstructPix2Pix</a></li>
                <li><a href="Extension_Research_Models/Image_Editing_Models/Insert-Anything">Insert-Anything</a></li>
                <li><a href="Extension_Research_Models/Image_Editing_Models/ChordEdit">ChordEdit</a></li>
                <li><a href="Extension_Research_Models/Image_Editing_Models/OPCD">OPCD</a></li>
                <li><a href="Extension_Research_Models/Image_Editing_Models/DGINStyle">DGINStyle</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_Similarity_Models/VisDiff">VisDiff</a></li>
                <li><a href="Extension_Research_Models/Image_Similarity_Models/relsim">relsim</a></li>
                <li><a href="Extension_Research_Models/Image_Similarity_Models/MARCO">MARCO</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Active_Learning_Models/APL">APL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PCB">PCB</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PCoreSet">PCoreSet</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CALD">CALD</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/AL">AL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/LabelBench">LabelBench</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/EADA">EADA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/TQS">TQS</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/SDM">SDM</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/RIPU">RIPU</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/A2LC">A2LC</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ALC">ALC</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/SUGFW">SUGFW</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/NAS">NAS</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/TypiClust">TypiClust</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/SCAN">SCAN</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/UHerding">UHerding</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ISOAL">ISOAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CSQ">CSQ</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/BADGE">BADGE</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PT4AL">PT4AL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PPAL">PPAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/BLAD">BLAD</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/BGADL">BGADL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/AL-consistency">AL-consistency</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ALFA-Mix">ALFA-Mix</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PAL">PAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CAMPAL">CAMPAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/LADA">LADA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/SAAL">SAAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/DiaNA">DiaNA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/AutoAugment">AutoAugment</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/UWE">UWE</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ActGen">ActGen</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CAL">CAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/AccuACL">AccuACL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/E2OAL">E2OAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/BalancedEntropy">BalancedEntropy</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/HALO">HALO</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/DG_AL">DG_AL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CEG">CEG</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ELDA">ELDA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ASTE-AL">ASTE-AL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/WAAL">WAAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/TAAL">TAAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/TA-VAAL">TA-VAAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ARAL">ARAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/AAL">AAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ASDA">ASDA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/FEDALV">FEDALV</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/C_PEAL">C_PEAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ActiveTeacher">ActiveTeacher</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/IPGPT-SFADA">IPGPT-SFADA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ActMAD">ActMAD</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/EAOA">EAOA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/EOAL">EOAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/LfOSA">LfOSA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ELPT">ELPT</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/EATTA">EATTA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ADA-LDM">ADA-LDM</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/LCADA">LCADA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/DeepAL">DeepAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/Epiplexity">Epiplexity</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CLUE">CLUE</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/Category_Aware_DA">Category_Aware_DA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PALM">PALM</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/PALF">PALF</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/DSL">DSL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/UGTST">UGTST</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/HiCAL">HiCAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/BADA">BADA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/EAGC">EAGC</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/FairPAL">FairPAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/Clearning-The-Pool">Clearning-The-Pool</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/Fast_Fishing">Fast_Fishing</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ATTA">ATTA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/JODA">JODA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ALLs">ALLs</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/SaE">SaE</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/LAMDA">LAMDA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ALUS">ALUS</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/ViewAL">ViewAL</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/CCMA">CCMA</a></li>
                <li><a href="Extension_Research_Models/Active_Learning_Models/GAL4Personalization">GAL4Personalization</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Upsample_Models/UpsampleAnything">UpsampleAnything</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/LoftUp">LoftUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/FeatUp">FeatUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/AnyUp">AnyUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/NAF">NAF</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/FeatSharp">FeatSharp</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/LiFT">LiFT</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/UPLiFT">UPLiFT</a></li>
                <li><a href="Extension_Research_Models/Upsample_Models/ViT-Up">ViT-Up</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Autoregressive_Models/AutoregressiveSegmentation">AutoregressiveSegmentation</a></li>
                <li><a href="Extension_Research_Models/Autoregressive_Models/RestoreVAR">RestoreVAR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/CSZL_Models/TOMCAT">TOMCAT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Segmentation_Models/RD-Net">RD-Net</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/SegEarth-OV-3">SegEarth-OV-3</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/CrowdSAM">CrowdSAM</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/SAM-DAQ">SAM-DAQ</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/SAM3-Adapter">SAM3-Adapter</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/DINO-AugSeg">DINO-AugSeg</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/FSSDINO">FSSDINO</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/VidEoMT">VidEoMT</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/RNS">RNS</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/Fractal-Pretraining">Fractal-Pretraining</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/SegRCDB">SegRCDB</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/SegDINO_v2">SegDINO_v2</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/M2C">M2C</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/RSRM">RSRM</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/C2AM-Seg">C2AM-Seg</a></li>
                <li><a href="Extension_Research_Models/Segmentation_Models/HFS-SAM2">HFS-SAM2</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/ICL_Models/AWRaCLe">AWRaCLe</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/CL_Models/CL_all-in-one">CL_all-in-one</a></li>
                <li><a href="Extension_Research_Models/CL_Models/LEAR">LEAR</a></li>
                <li><a href="Extension_Research_Models/CL_Models/ARCL-ViT">ARCL-ViT</a></li>
                <li><a href="Extension_Research_Models/CL_Models/QGPM">QGPM</a></li>
                <li><a href="Extension_Research_Models/CL_Models/GPM">GPM</a></li>
                <li><a href="Extension_Research_Models/CL_Models/FIRE">FIRE</a></li>
                <li><a href="Extension_Research_Models/CL_Models/ACL">ACL</a></li>
                <li><a href="Extension_Research_Models/CL_Models/PILOT">PILOT</a></li>
                <li><a href="Extension_Research_Models/CL_Models/Low-rank-CL">Low-rank-CL</a></li>
                <li><a href="Extension_Research_Models/CL_Models/InfLoRA">InfLoRA</a></li>
                <li><a href="Extension_Research_Models/CL_Models/SD-LoRA">SD-LoRA</a></li>
                <li><a href="Extension_Research_Models/CL_Models/CL-LoRA">CL-LoRA</a></li>
                <li><a href="Extension_Research_Models/CL_Models/AFEC">AFEC</a></li>
                <li><a href="Extension_Research_Models/CL_Models/IDER">IDER</a></li>
                <li><a href="Extension_Research_Models/CL_Models/SNV">SNV</a></li>
                <li><a href="Extension_Research_Models/CL_Models/Continual_Distillation">Continual_Distillation</a></li>
                <li><a href="Extension_Research_Models/CL_Models/AREA">AREA</a></li>
                <li><a href="Extension_Research_Models/CL_Models/E2-LoRA">E2-LoRA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Toolbox/SGG-Annotate">SGG-Annotate</a></li>
                <li><a href="Extension_Research_Models/Toolbox/CrowdSAM">CrowdSAM</a></li>
                <li><a href="Extension_Research_Models/Toolbox/LabelAny3D">LabelAny3D</a></li>
                <li><a href="Extension_Research_Models/Toolbox/SAM3D-Body-Rerun">SAM3D-Body-Rerun</a></li>
                <li><a href="Extension_Research_Models/Toolbox/Paper2Rebuttal">Paper2Rebuttal</a></li>
                <li><a href="Extension_Research_Models/Toolbox/AutoFigure">AutoFigure</a></li>
                <li><a href="Extension_Research_Models/Toolbox/Intrinsic">Intrinsic</a></li>
                <li><a href="Extension_Research_Models/Toolbox/Autoresearch">Autoresearch</a></li>
                <li><a href="Extension_Research_Models/Toolbox/VisionCaptioner">VisionCaptioner</a></li>
                <li><a href="Extension_Research_Models/Toolbox/AI-Scientist-v2">AI-Scientist-v2</a></li>
                <li><a href="Extension_Research_Models/Toolbox/AutoResearchClaw">AutoResearchClaw</a></li>
                <li><a href="Extension_Research_Models/Toolbox/Nougat">Nougat</a></li>
                <li><a href="Extension_Research_Models/Toolbox/OmniParse">OmniParse</a></li>
                <li><a href="Extension_Research_Models/Toolbox/LocateAnything">LocateAnything</a></li>
                <li><a href="Extension_Research_Models/Toolbox/Unlimited-OCR">Unlimited-OCR</a></li>
                <li><a href="Extension_Research_Models/Toolbox/CleanVision">CleanVision</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Energy_Models/EBMs_Jarzynski">EBMs_Jarzynski</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/TEA">TEA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/Hyper-SET">Hyper-SET</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EBT">EBT</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EBTSA">EBTSA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/ET">ET</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EBM_CL">EBM_CL</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EBCLR">EBCLR</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/JEM">JEM</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/GeneralizedEBM">GeneralizedEBM</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/Discrete-EBM">Discrete-EBM</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EB_GFN">EB_GFN</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/HeatOOD">HeatOOD</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EnergyOOD">EnergyOOD</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/OEST">OEST</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EOW">EOW</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/VERA">VERA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/DAEN">DAEN</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EGC">EGC</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/BPT-EBM">BPT-EBM</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EnergyMatching">EnergyMatching</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EBGAN">EBGAN</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/APT">APT</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EagleNet">EagleNet</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/EnergyKD">EnergyKD</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/AEA">AEA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/PTTEA">PTTEA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/CRETTA">CRETTA</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/Glocal">Glocal</a></li>
                <li><a href="Extension_Research_Models/Energy_Models/Detective">Detective</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Reinforcement_Learning_Models/RL-AWB">RL-AWB</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Quantum_Models/TorchQuantum">TorchQuantum</a></li>
                <li><a href="Extension_Research_Models/Quantum_Models/QuantumGS">QuantumGS</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Test_Time_Models/ViTTT">ViTTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTT">TTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTT-MAE">TTT-MAE</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTT-Video">TTT-Video</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTT_Denoising">TTT_Denoising</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/NCTTT">NCTTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTT_Matching_VOS">TTT_Matching_VOS</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/ITTT">ITTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/DATTT">DATTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTA">TTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/Learning-Loss-TTA">Learning-Loss-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTAch">TTAch</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/MTA">MTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/GPS-TTA">GPS-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/CI-TTA">CI-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/FSL">FSL</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/FSL-Rectifier">FSL-Rectifier</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/DynTTA">DynTTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/MEMO">MEMO</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/MV-ATTA">MV-ATTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/RLCF">RLCF</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/tttLRM">tttLRM</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TPT">TPT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/O-TPT">O-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/C-TPT">C-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/Vision-TTT">Vision-TTT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/A-TPT">A-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/IMSE">IMSE</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/ZipMap">ZipMap</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTAC">TTAC</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTAC2">TTAC2</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/S4T-mtl">S4T-mtl</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TTD">TTD</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/MOON">MOON</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/BayesianTTA">BayesianTTA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/TDA">TDA</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/DPE-CLIP">DPE-CLIP</a></li>
                <li><a href="Extension_Research_Models/Test_Time_Models/U-TTT">U-TTT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Superpixel_Models/SSN">SSN</a></li>
                <li><a href="Extension_Research_Models/Superpixel_Models/SPAM">SPAM</a></li>
            </ul>
        </td>
        </tr>
    </tbody>
</table>
 
## 4. Citation
If you find our work useful, please cite the following:
```
@misc{Chi2023,
  author       = {Chi Tran},
  title        = {OpenAnomaly: An Open Source Implementation of Anomaly Detection Methods},
  publisher    = {GitHub},
  booktitle    = {GitHub repository},
  howpublished = {https://github.com/SKKUAutoLab/OpenAnomaly},
  year         = {2023}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com) or [tdc2000@skku.edu](tdc2000@skku.edu)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.