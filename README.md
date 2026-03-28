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

#### 1.1.2 Using requirements.txt
```bash
conda create --name anomaly python=3.10.12
conda activate anomaly
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r envs/requirements.txt
pip install -e .
```

### 1.2 Using uv
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

### 2.5. Surveillance Traffic Anomaly Detection Datasets
For the TUMTrafficQA dataset, please download it from this [repository](https://github.com/TraffiX-VideoQA/TraffiX-Qwen)

For the WWC dataset, please download it from this [repository](https://github.com/VICA-Lab-HKUST-GZ/WWC-Predictor)

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
 
### 3.1.2. Supported Models for 3D Industrial Anomaly Detection
| Models        | MVTec3D            | Eyecandies         | MadSim             | Anomaly-ShapNet    | Real3D-AD          | BrokenChairs | PD-Real | MulSen-AD          | infra 3DALv2       | MIP                | ToysAD-8K          |
|---------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------|---------|--------------------|--------------------|--------------------|--------------------|
| BTF           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CFM           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CPMF          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Looking3D     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| M3DM          | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PAD           | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Shape-Guided  | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| SplatPose     | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CRD           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CMDIAD        | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MulSen-AD     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :heavy_check_mark: | :x:                | :x:                | :x:                |
| FPFHI         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :heavy_check_mark: | :x:                | :x:                |
| EasyNet       | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Real3D-AD     | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| 3DSR          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PO3AD         | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MC4AD         | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| R3D-AD        | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| C3DAD         | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :heavy_check_mark: | :x:                | :x:                | :x:                |
| Reg2Inv       | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MVR           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| MC3D-AD       | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PIAD          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :heavy_check_mark: | :x:                |
| GLFM          | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| ISMP          | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| SplatPosePlus | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CIF           | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| PASDF         | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| RIF           | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CASL          | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| Odd-One-Out   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :heavy_check_mark: |
| BridgeNet     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| GS-CLIP       | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| CMDR-IAD      | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |
| BTP-3DAD      | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:          | :x:     | :x:                | :x:                | :x:                | :x:                |

### 3.1.3. Supported Models for Anomaly Generative Industrial Anomaly Detection
| Models          | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD | MVTec3D            |
|-----------------|--------------------|--------------------|--------------------|------------|--------------------|------|----------|--------------------|
| Defect Spectrum | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| DualAnoDiff     | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| MAGIC           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| SeaS            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:      | :heavy_check_mark: |
| AnoStyler       | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |
| O2MAG           | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:      | :x:                |

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

### 3.1.8. Supported Models for Multi-class Industrial Anomaly Detection
| Models       | MVTec              | BTAD               | VisA               | MVTec LOCO | MPDD               | WFDD | Real-IAD           | MVTec3D            | ITDD               |
|--------------|--------------------|--------------------|--------------------|------------|--------------------|------|--------------------|--------------------|--------------------|
| CRAD         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| Real-IAD     | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |
| UniAD        | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |
| HVQ          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| MSTAD        | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| MambaAD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                |
| ViTAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                |
| InvAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                |
| Dinomaly     | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |
| RLR          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| UniFormaly   | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| OneNIP       | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| IUF          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| MoEAD        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| PMAD         | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| UniAS        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| Omni-AD      | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |
| LGC          | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |
| RAS          | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| Wave-MambaAD | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :x:                | :x:  | :x:                | :heavy_check_mark: | :x:                |
| CRAS         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:        | :heavy_check_mark: | :x:  | :x:                | :heavy_check_mark: | :heavy_check_mark: |
| AnomalyMoE   | :heavy_check_mark: | :x:                | :x:                | :x:        | :x:                | :x:  | :x:                | :x:                | :x:                |
| MVAD         | :x:                | :x:                | :x:                | :x:        | :x:                | :x:  | :heavy_check_mark: | :x:                | :x:                |

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
| Models              | MVTec              | BTAD               | VisA               | MVTec LOCO         | MPDD               | WFDD               | Real-IAD | ELPV               | SDD                | AITEX              | BrainMRI           | HeadCT             | DAGM               | DTD                | Br35H              | ISIC               | ColonDB            | ClinicDB           | TN3K               | Real3D             | Eyecandies         | MVTec3D            | MVTec-FS           | KSDD2              | MTD                | GoodsAD            | LiverCT            | RESC               | HIS                | ChestXray          | OCT17              |
|---------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|----------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| WinCLIP             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PromptAD            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| InCTRL              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyCLIP         | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| APRIL-GAN           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaCLIP             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PointAD             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AMI-Net             | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ResAD               | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MVP-PCLIP           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VCP-PCLIP           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnoVL               | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MVREC               | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Segment-Any-Anomaly | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CLIP-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GPT-4V              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FADE                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AnomalyDINO         | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FiLo                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MetaUAS             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| FoundAD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| NAGL                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CoDeGraph           | :heavy_check_mark: | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AD-DINOv3           | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| ReMP-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| FAPrompt            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RareCLIP            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MuSc                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DictAS              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Bayes-PFL           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MultiADS            | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AF-CLIP             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TPS                 | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RegAD               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AA-CLIP             | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| INP-Former          | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LogSAD              | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| IIPAD               | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LAFT                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniVAD              | :heavy_check_mark: | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Anomagic            | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| AdaptCLIP           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PCSNet              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DIVAD               | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TF-IDG              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UniADC              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Tipsomaly           | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MRAD                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SubspaceAD          | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| D24FAD              | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| VisualAD            | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GATE-AD             | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:      | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.2. Supported Models for Dashcam Traffic Anomaly Detection
| Models        | ROL                | DoTA               | CCD                | DAD                | A3D                | CTA                | DADA-2000          | W3DA               | VRU                | RoadSocial         | TrafficGaze        | MASKER_MD          | DADA-Seg           | DADA-1000          | CUVA               | Russia Crash       | TSOD10K            | MM-AU              | CrashChat          | AV-TAU             |
|---------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| AMNet         | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Baseline GRU  | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DSTA          | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| UString       | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| XAI-Accident  | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CTA           | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| TTHF          | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| W3DA          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RARE          | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| BADAS         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DriveCLIP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DriveCLIP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| PromptTAD     | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| RoadSocial    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Hawk          | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CGNAM         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| SaliencyMamba | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| DRIVE         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| GSC           | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| MMUDA         | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LOTVS-CAP     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| LOTVS-CAAD    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| Graph-Graph   | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CRASH         | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CUVA          | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                |
| Ctrl-Crash    | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| Tramba        | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| LOTVS-MM-AU   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| CrashChat     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :heavy_check_mark: | :x:                |
| EchoTraffic   | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.3. Supported Models for Unsupervised Anomaly Detection
| Models                      | Avenue             | ShanghaiTech       | Ped2               | Arbitrary Video    | PAB                | HumanSAM           | Vad-R1             |
|-----------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
| 3DNet                       | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                |
| GMM_DAE                     | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| Jigsaw-VAD                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| OGAM-MRAM                   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| SwinAnomaly                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                |
| GenerateAnomaliesFromNormal | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| F2LM                        | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| VADMamba                    | :heavy_check_mark: | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| AED-MAE                     | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| CMP                         | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                |
| DDL                         | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |
| HumanSAM                    | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                |
| TAO                         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                |
| Vad-R1                      | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :heavy_check_mark: |

### 3.4. Supported Models for Surveillance Traffic Anomaly Detection
| Models        | UCF-Traffic        | TAD                | TSP6K              | SO-TAD             | TUMTrafficQA       | WWC |
|---------------|--------------------|--------------------|--------------------|--------------------|--------------------|-----|
| TA-NET        | :heavy_check_mark: | :heavy_check_mark: | :x:                | :x:                | :x:                | :x: |
| TSP6K         | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x: |
| SO-TAD        | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x: |
| TraffiX-Qwen  | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x: |
| WWC-Predictor | :x:                | :x:                | :x:                | :x:                | :x:                | :x: |

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
| PiDiNet                       | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: |
| SAM-Audio                     | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :heavy_check_mark: | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                | :x:                |

### 3.7. Supported Models for AI City Challenge
| Models   | Iowa DOT           |
|----------|--------------------|
| CETCVLAB | :heavy_check_mark: |
| SIS_Lab  | :heavy_check_mark: |

### 3.7. Supported Models for Extension Research
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
                <b>Dataset Distillation Models</b>
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
        <tr valign="top">
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Attention_modules/A2">A2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/ACmix">ACmix Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/AFT">AFT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Agent Attention">Agent Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/AoA">Attention on Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Axial">Axial Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Axial Attention">Axial Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/BAM">BAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/BidirectionalCrossAttention">Bidirectional Cross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CBAM">CBAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CoAtNet">CoAtNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CompositionalAttention">Compositional Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Coord">Coord Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CoT">CoT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CrissCross">CrissCross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Crossformer">Crossformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/DANet">DANet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/DAT">DAT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/DeformableAttention">Deformable Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/ECA">ECA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/EMSA">EMSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/External">External Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/FlashTransformer">Flash Transformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/GFNet">GFNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/GSA">GSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Halo">Halo Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Hamburger">Hamburger Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/KroneckerAttention">Kronecker Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/LambdaNet">LambdaNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Linformer">Linformer Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/LocalAttention">Local Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MOATransformer">MOA Transformer</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MobileViT">MobileViT Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MobileViTv2">MobileViTv2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MUSE">MUSE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Nonlocal">Non-local Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/NystromAttention">Nystrom Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Outlook">Outlook Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/ParNet">ParNet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/PolarizedSelfAttention">PolarizedSelf Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/PSA">PSA Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Residual">Residual Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/S2">S2 Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SCSE">SCSE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SelfAttention">Self-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SGE">SGE Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Shuffle">Shuffle Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SimAM">SimAM Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SimplifiedSelfAttention">Simplified Self-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SK">SK Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SlotAttention">Slot Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/TaylorLinearAttention">Taylor Linear Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Triplet">Triplet Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/UFO">UFO Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/ViP">ViP Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/RingAttention">Ring Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/HierarchicalAttention">Hierarchical Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/RectifiedLinearAttention">Rectified Linear Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/STAM">Space Time Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/HaloAttentiono">Halo Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CoordinateDescentAttention">Coordinate Descent Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MemoryCompressedAttention">Memory Compressed Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Mega">Mega Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Tranception">Tranception Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/InfiniAttention">Infini Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/SparseAttention">Sparse Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MultimodalCrossAttention">Multi-modal Cross Attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/AttentionConvolution">Attention with Convolution</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MoA">MoA</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/LinearAttentionTransformer">LinearAttentionTransformer</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Performer">Performer</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/ISAB">ISAB</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MemoryEfficientAttention">MemoryEfficientAttention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/Perceiver">Perceiver</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CoLT5-attention">CoLT5-attention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/GatedSlotAttention">GatedSlotAttention</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/JVP">JVP</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/CBSA">CBSA</a></li>
                <li><a href="Extension_Research_Models/Attention_modules/MCA">MCA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Convolution_architectures/DynamicConvolution">Dynamic Convolution</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/WTConv">Wavelet Convolution</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/InvertibleResidualNetworks">Invertible Residual Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/FrEIA">FrEIA</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/FFF">Fast Feedforward Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/GLOM">GLOM</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/Firefly">Firefly</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/BEM">Bayesian Enhancement Model</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/mincam">Minimalist Vision with Freeform Pixels</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/AKOrN">AKOrN</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/Any-Precision-DNNs">Any-Precision Deep Neural Network</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/FDConv">FDConv</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/OverLoCK">OverLoCK</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/ConvNeXt">ConvNeXt</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/HSNet">HSNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/homeostatic-neural-networks">Homeostatic Neural Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/VAT">VAT</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/NC-Net">NC-Net</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/ANCNet">ANCNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/CVNet">CVNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/ConverseNet">ConverseNet</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/ConvNeXt-V2">ConvNeXt-V2</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/FFC">FFC</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/Contextual-Convolutional-Networks">Contextual-Convolutional-Networks</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/CoConv">CoConv</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/LogViG">LogViG</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/Optical-FCNN">Optical-FCNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/FDAM">FDAM</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/SN-Net">SN-Net</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/Nested-Learning">Nested-Learning</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/CycleMLP">CycleMLP</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/SimSiam">SimSiam</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/TRL">TRL</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/DDL">DDL</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/V4D">V4D</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/BayesianCNN">BayesianCNN</a></li>
                <li><a href="Extension_Research_Models/Convolution_architectures/F-Conv">F-Conv</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Generative_models/EMA">EMA</a></li>
                <li><a href="Extension_Research_Models/Generative_models/VectorQuantization">Vector Quantization</a></li>
                <li><a href="Extension_Research_Models/Generative_models/VQ-VAE">VQ-VAE</a></li>
                <li><a href="Extension_Research_Models/Generative_models/TiTok">TiTok</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiT">DiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Fast-DiT">Fast DiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DTR">DTR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/ANT">ANT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/R3GAN">R3GAN</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DALL-E">DALL-E</a></li>
                <li><a href="Extension_Research_Models/Generative_models/VAR">VAR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CleanDIFT">CleanDIFT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Vec2Vec">Vec2Vec</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FoD">FoD</a></li>
                <li><a href="Extension_Research_Models/Generative_models/taming-transformers">taming-transformers</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiS">DiS</a></li>
                <li><a href="Extension_Research_Models/Generative_models/MAR">MAR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiCo">DiCo</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Tweedie">Tweedie</a></li>
                <li><a href="Extension_Research_Models/Generative_models/EnergyMatching">EnergyMatching</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DDPM">DDPM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Guided-Diffusion">Guided-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Improved-Diffusion">Improved-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FSDDIM">FSDDIM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DDIM">DDIM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/RectifiedFlow">RectifiedFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_models/GMFlow">GMFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_models/ScoreSDE">ScoreSDE</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FlowGrad">FlowGrad</a></li>
                <li><a href="Extension_Research_Models/Generative_models/InstaFlow">InstaFlow</a></li>
                <li><a href="Extension_Research_Models/Generative_models/EDTR">EDTR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/TorchCFM">TorchCFM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/RAE">RAE</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CSBM">CSBM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/REPA">REPA</a></li>
                <li><a href="Extension_Research_Models/Generative_models/InDI">InDI</a></li>
                <li><a href="Extension_Research_Models/Generative_models/U-ViT">U-ViT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiffMoE">DiffMoE</a></li>
                <li><a href="Extension_Research_Models/Generative_models/MDM">MDM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/JiT">JiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DDT">DDT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/SiT">SiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FractalGen">FractalGen</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CoDe">CoDe</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DAAM">DAAM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Glance">Glance</a></li>
                <li><a href="Extension_Research_Models/Generative_models/ViBT">ViBT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/TREAD">TREAD</a></li>
                <li><a href="Extension_Research_Models/Generative_models/MaskDiT">MaskDiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/EDM">EDM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/ToMe">ToMe</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Open-Sora">Open-Sora</a></li>
                <li><a href="Extension_Research_Models/Generative_models/REPA-E">REPA-E</a></li>
                <li><a href="Extension_Research_Models/Generative_models/LearningWithNoise">LearningWithNoise</a></li>
                <li><a href="Extension_Research_Models/Generative_models/S-NODE">S-NODE</a></li>
                <li><a href="Extension_Research_Models/Generative_models/LTX-Video">LTX-Video</a></li>
                <li><a href="Extension_Research_Models/Generative_models/MobileI2V">MobileI2V</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Next-Patch-Prediction">Next-Patch-Prediction</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DimensionX">DimensionX</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Cosmos-Predict2.5">Cosmos-Predict2.5</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DeepVerse">DeepVerse</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Cosmos-Transfer2.5">Cosmos-Transfer2.5</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Wan2.2">Wan2.2</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CogVideo">CogVideo</a></li>
                <li><a href="Extension_Research_Models/Generative_models/LTX-2">LTX-2</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiT-Extrapolation">DiT-Extrapolation</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Self-Refining">Self-Refining</a></li>
                <li><a href="Extension_Research_Models/Generative_models/UniVideo">UniVideo</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FlashWorld">FlashWorld</a></li>
                <li><a href="Extension_Research_Models/Generative_models/PixelGen">PixelGen</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FastVMT">FastVMT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/REPA-G">REPA-G</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DummyForcing">DummyForcing</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CSBM">CSBM</a></li>
                <li><a href="Extension_Research_Models/Generative_models/MVAR">MVAR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/EAR">EAR</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Drifting">Drifting</a></li>
                <li><a href="Extension_Research_Models/Generative_models/LongLive">LongLive</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Self-Forcing">Self-Forcing</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Driftin">Driftin</a></li>
                <li><a href="Extension_Research_Models/Generative_models/FlowCache">FlowCache</a></li>
                <li><a href="Extension_Research_Models/Generative_models/LayerSync">LayerSync</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Spacetime-Diffusion">Spacetime-Diffusion</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Mobile-O">Mobile-O</a></li>
                <li><a href="Extension_Research_Models/Generative_models/RollingSink">RollingSink</a></li>
                <li><a href="Extension_Research_Models/Generative_models/CHORDS">CHORDS</a></li>
                <li><a href="Extension_Research_Models/Generative_models/Helios">Helios</a></li>
                <li><a href="Extension_Research_Models/Generative_models/DiffiT">DiffiT</a></li>
                <li><a href="Extension_Research_Models/Generative_models/WAE">WAE</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Geometric_models/geoopt">geoopt</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HCL">HCL</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyperbolicCV">Hyperbolic CV</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyperbolicEmbedding">Hyperbolic Embedding</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyperbolicVisionTransformer">Hyperbolic Vision Transformer</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/meru">meru</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HCL">HCL</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/PoincareResnet">Poincare Resnet</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypMix">HypMix</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CurvatureGeneration">Curvature Generation</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CO-SNE">CO-SNE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypAD">HypAD</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HVT">HVT</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HiHPQ">HiHPQ</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/GM-VAE">GM-VAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/GHSW">GHSW</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/DOHSC-DO2HSC">DOHSC-DO2HSC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/GW-Regularization">GW-Regularization</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hypformer">Hypformer</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HAE">HAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HYSP">HYSP</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HR-MLP">HR-MLP</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/hierahyp">hierahyp</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypLiLoc">HypLiLoc</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/UnHyperML">UnHyperML</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypStructure">HypStructure</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HierarchyCPCC">HierarchyCPCC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CIDER">CIDER</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HYPE">HYPE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/DMT-HI">DMT-HI</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HHCH">HHCH</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/hypvl">hypvl</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/hycoclip">hycoclip</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyProMeta">HyProMeta</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/LResNet">LResNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/hyperbolic_wrapped_distribution">Hyperbolic Wrapped Distribution</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/pvae">pvae</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/DistilVPR">DistilVPR</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypHC">HypHC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/L3DMC">L3DMC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/GPr-Net">GPr-Net</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/longtail_segmentation">Longtail Segmentation</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hyperbolic_Feature_Augmentation">Hyperbolic Feature Augmentation</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/hyperfuture">hyperfuture</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CurAML">CurAML</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/BREEDS">BREEDS</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/OTCPCC">OTCPCC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HPEC">HPEC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/QCNN">QCNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CLIP-M3">CLIP-M3</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hyp-OW">Hyp-OW</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/AngularGap">AngularGap</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/vMF">vMF</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Rotation_Trick">Rotation_Trick</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HySAC">HySAC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/QRNN">QRNN</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/QCLNet">QCLNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Quaternion-Capsule-Networks">Quaternion-Capsule-Networks</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/QGAN">QGAN</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HSN">HSN</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyCoRe">HyCoRe</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HLFormer">HLFormer</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HOVER">HOVER</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyperCLIP">HyperCLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypCD">HypCD</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HS-DTE">HS-DTE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypGCD">HypGCD</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/H-CLIP">H-CLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/ATMG">ATMG</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Geo-Sign">Geo-Sign</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HAMN">HAMN</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HexFormer">HexFormer</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/PHyCLIP">PHyCLIP</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hyperbolic_Interpretability">Hyperbolic_Interpretability</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HIVE">HIVE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HoroPCA">HoroPCA</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypModalAlign">HypModalAlign</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypPNet">HypPNet</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HIDISC">HIDISC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HDD">HDD</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypDAE">HypDAE</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HyperRL">HyperRL</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Graph-Prompt">Graph-Prompt</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HypLL">HypLL</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hyperbolic-Vim">Hyperbolic-Vim</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/phier">phier</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/MGIoU">MGIoU</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/Hyperbolic_Action">Hyperbolic_Action</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/CQH">CQH</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/WMDD">WMDD</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HybCBC">HybCBC</a></li>
                <li><a href="Extension_Research_Models/Geometric_models/HBNN">HBNN</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Losses/OT-CLIP">OT CLIP</a></li>
                <li><a href="Extension_Research_Models/Losses/GradNorm">GradNorm</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Token_mixer_models/gMLP">gMLP</a></li>
                <li><a href="Extension_Research_Models/Token_mixer_models/MLP-Mixer">MLP Mixer</a></li>
                <li><a href="Extension_Research_Models/Token_mixer_models/Segformer">Segformer</a></li>
                <li><a href="Extension_Research_Models/Token_mixer_models/ResMLP">ResMLP</a></li>
                <li><a href="Extension_Research_Models/Token_mixer_models/ActiveMLP">ActiveMLP</a></li>
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
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Transformer_models/RotaryEmbedding">Rotary Embedding</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/TNT">Transformer in Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SplineTransformers">Spline-based Transformers</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DyT">DyT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Diff-Transformer">Diff-Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ELN">ELN</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MAE">MAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/PVT">PVT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Vision_GNN">Vision_GNN</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DINO">DINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DINOv2">DINOv2</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/RC-MAE">RC-MAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/VGP">VGP</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/EoMT">EoMT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SwinIRDehaze">SwinIRDehaze</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/FasterViT">FasterViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Twins">Twins</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Deformable-DETR">Deformable-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DETR">DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ConditionalDETR">ConditionalDETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MetaFormer">MetaFormer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/HcNet">HcNet</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DINOv3">DINOv3</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/HyperMAE">HyperMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/RoPE-ViT">RoPE-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SparK">SparK</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SBT">SBT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/PH-Reg">PH-Reg</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/JAFAR">JAFAR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MultiMAE">MultiMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Vision-LSTM">Vision-LSTM</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DinoUNet">DinoUNet</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SegDINO">SegDINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MedDINOv3">MedDINOv3</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/REOrder">REOrder</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/LIT">LIT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MedicalMAE">MedicalMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Attention-Transfer">Attention-Transfer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Test-Time-Register">Test-Time-Register</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/PolaFormer">PolaFormer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/BRIXEL">BRIXEL</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ThinkingViT">ThinkingViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Franca">Franca</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SMCA-DETR">SMCA-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ChangeDINO">ChangeDINO</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/CRATE">CRATE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DReX">DReX</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/RADIO">RADIO</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/CARE_Transformer">CARE_Transformer</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/FGTS">FGTS</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SnapViT">SnapViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/FractalDB">FractalDB</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Derf">Derf</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/POA">POA</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Pixio">Pixio</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/NEPA">NEPA</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Co2S">Co2S</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ALViT">ALViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DAB-DETR">DAB-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MDETR">MDETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DN-DETR">DN-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MobileViT">MobileViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/SSMAE">SSMAE</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/CAS-ViT">CAS-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/EdgeNeXt">EdgeNeXt</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/LoRA-DETR">LoRA-DETR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DIFFSSR">DIFFSSR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ViT-Plasticity">ViT-Plasticity</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DINOv3_Animal">DINOv3_Animal</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/AdaptOVCD">AdaptOVCD</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/Raptor">Raptor</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/ViT-5">ViT-5</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/MLLA">MLLA</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/LAST-ViT">LAST-ViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/DiT4SR">DiT4SR</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/LocAtViT">LocAtViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/FastDINOv2">FastDINOv2</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/BinaryAttention">BinaryAttention</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/AdapterTune">AdapterTune</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/HAViT">HAViT</a></li>
                <li><a href="Extension_Research_Models/Transformer_models/PlaneCycle">PlaneCycle</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/UNet_models/Uformer">Uformer</a></li>
                <li><a href="Extension_Research_Models/UNet_models/ReversibleUnet">Reversible U-Net</a></li>
                <li><a href="Extension_Research_Models/UNet_models/X-UNet">X-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_models/MIMO-UNet">MIMO-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_models/UNetv2">UNetv2</a></li>
                <li><a href="Extension_Research_Models/UNet_models/SAM3-UNet">SAM3-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_models/SAM2-UNet">SAM2-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_models/SAM-UNet">SAM-UNet</a></li>
                <li><a href="Extension_Research_Models/UNet_models/SAM2-UNeXT">SAM2-UNeXT</a></li>
                <li><a href="Extension_Research_Models/UNet_models/Hyperbolic-U-Net">Hyperbolic-U-Net</a></li>
            </ul>
        </td>
         <td>
            <ul>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/2D_Gaussian_Splatting">2D Gaussian Splatting</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/3D_Gaussian_Splatting">3D Gaussian Splatting</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/iNeRF">iNeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/NeRF">NeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/SyncDreamer">SyncDreamer</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/D2GV">D2GV</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GaussianImage">GaussianImage</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Easi3R">Easi3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/VGGSfM">VGGSfM</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/DUSt3R">DUSt3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Fast3R">Fast3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MBPR">MBPR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GroupRec">GroupRec</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/PromptHMR">PromptHMR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/CrowdRec">CrowdRec</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/2DGS_Inpaint">2DGS_Inpaint</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GSASR">GSASR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/ContinuousSR">ContinuousSR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/DeblurGS">DeblurGS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Deblur-NeRF">Deblur-NeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/BAD-Gaussians">BAD-Gaussians</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/CLIPNeRF">CLIPNeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/CLIP2NeRF">CLIP2NeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/nerfstudio">nerfstudio</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/SpikeNeRF">SpikeNeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/gsplat">gsplat</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/NerfAcc">NerfAcc</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Human3R">Human3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/4DGaussians">4DGaussians</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MapAnything">MapAnything</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/UFM">UFM</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MuM">MuM</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/SAM-Body4D">SAM-Body4D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/HSMR">HSMR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Sharp">Sharp</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Multi-View-Foundation-Models">Multi-View-Foundation-Models</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Pi3">Pi3</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/DVGT">DVGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/EgoX">EgoX</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/PedGen">PedGen</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/VGGT4D">VGGT4D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/InfiniteVGGT">InfiniteVGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/DroneSplat">DroneSplat</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/WorldGen">WorldGen</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LiftImage3D">LiftImage3D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MASt3R">MASt3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Gen3R">Gen3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/UniSH">UniSH</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/instant-ngp">instant-ngp</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/EasyVolcap">EasyVolcap</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/4K4D">4K4D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/V-DPM">V-DPM</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Geo4D">Geo4D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MonST3R">MonST3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MVGGT">MVGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LongSplat">LongSplat</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GaussianSR">GaussianSR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Instant-GI">Instant-GI</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LIG">LIG</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/IGGT">IGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/VeGaS">VeGaS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GSVC">GSVC</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/4DGS">4DGS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GaussianToken">GaussianToken</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/D-NeRF">D-NeRF</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Deformable-3D-Gaussians">Deformable-3D-Gaussians</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/SAT-HMR">SAT-HMR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/FastGS">FastGS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/DashGaussian">DashGaussian</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/FastVGGT">FastVGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Shape-of-Motion">Shape-of-Motion</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Deblur4DGS">Deblur4DGS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Any4D">Any4D</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/JOSH">JOSH</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MotionCrafter">MotionCrafter</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/und3rstand">und3rstand</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GaussianImage++">GaussianImage++</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/ODE-GS">ODE-GS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LangSplat">LangSplat</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LangSplatV2">LangSplatV2</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/4DLangSplat">4DLangSplat</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/UBS">UBS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/OmniVGGT">OmniVGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/AMB3R">AMB3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Flow3r">Flow3r</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/CUT3R">CUT3R</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/GT2-GS">GT2-GS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MoE-GS">MoE-GS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/SGI">SGI</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LoGeR">LoGeR</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/Mobile-GS">Mobile-GS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/LASER">LASER</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/NanoGS">NanoGS</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/MatryoshkaGaussianSplatting">MatryoshkaGaussianSplatting</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/QuantVGGT">QuantVGGT</a></li>
                <li><a href="Extension_Research_Models/3d_reconstruction_models/PAGE4D">PAGE4D</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/MoE_models/MoE-CNN">MoE CNN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Soft-MoE">Soft MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ViT-Soft-MoE">ViT Soft MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/PEER">PEER</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ST-MoE">ST MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Sparse-MoE">Sparse MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/LIMoE">LIMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoD">MoD</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-Mamba">MoE Mamba</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MHMoE">Multi-Head MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoNE">MoNE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-Fusion">MoE-Fusion</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE">MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SinkhornRouter">Sinkhorn Router</a></li>
                <li><a href="Extension_Research_Models/MoE_models/BlackMamba">BlackMamba</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoU">MoU</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Switch-MoE">Switch MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DiT-MoE">DiT MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/TC-MoA">TC-MoA</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Tutel">Tutel</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoSE">MoSE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/M4oE">M4oE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/NID">NID</a></li>
                <li><a href="Extension_Research_Models/MoE_models/FAME">FAME</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MVMoE">MVMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Switch-DiT">Switch-DiT</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ScatterMoE">ScatterMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SFAN">SFAN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SMoE">SMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DeepMoE">DeepMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/GMoE">GMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoFME">MoFME</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ShiftAddViT">ShiftAddViT</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoA">MoA</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE_PromptCL">MoE-PromptCL</a></li>
                <li><a href="Extension_Research_Models/MoE_models/CRMoE">CRMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-Jetpack">MoE Jetpack</a></li>
                <li><a href="Extension_Research_Models/MoE_models/muMoE">muMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/CLIP-MoE">CLIP-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoH">MoH</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-Adapters4CL">MoE-Adapters4CL</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DAMEX">DAMEX</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Neural-Experts">Neural-Experts</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ConsistentMoE">ConsistentMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Mod-Squad">Mod-Squad</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DMP">DMP</a></li>
                <li><a href="Extension_Research_Models/MoE_models/AMIR">AMIR</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Efficient-WEMoE">Efficient-WEMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/PnD">PnD</a></li>
                <li><a href="Extension_Research_Models/MoE_models/weight-ensembling_MoE">weight-ensembling_MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MEGAN">MEGAN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DeMo">DeMo</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SM3Det">SM3Det</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MEASNet">MEASNet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SeemoRe">SeemoRe</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoCE-IR">MoCE-IR</a></li>
                <li><a href="Extension_Research_Models/MoE_models/AdaptIR">AdaptIR</a></li>
                <li><a href="Extension_Research_Models/MoE_models/pFedMoAP">pFedMoAP</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Earth-Adapter">Earth-Adapter</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DM-Adapter">DM-Adapter</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MicarVLMoE">MicarVLMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MFG-HMoE">MFG-HMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/QuARF">QuARF</a></li>
                <li><a href="Extension_Research_Models/MoE_models/LFME">LFME</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MixBridge">MixBridge</a></li>
                <li><a href="Extension_Research_Models/MoE_models/EMOE">EMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/LiveEdit">LiveEdit</a></li>
                <li><a href="Extension_Research_Models/MoE_models/beyond-param-count">Beyond Parameter Count</a></li>
                <li><a href="Extension_Research_Models/MoE_models/VMoBA">VMoBA</a></li>
                <li><a href="Extension_Research_Models/MoE_models/I2MoE">I2MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoPE">MoPE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Diff-Retinex++">Diff-Retinex++</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Inter2Former">Inter2Former</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-FFD">MoE-FFD</a></li>
                <li><a href="Extension_Research_Models/MoE_models/BMFH-Net">BMFH-Net</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MFRNet">MFRNet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/BIG-MoE">BIG-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Robust-MoE-CNN">Robust-MoE-CNN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/JTDMoE">JTDMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Diff-MoE">Diff-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ICL">ICL</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Face-MoGLE">Face-MoGLE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/AMoED">AMoED</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Switch-NeRF">Switch-NeRF</a></li>
                <li><a href="Extension_Research_Models/MoE_models/T-MoENet">T-MoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE-Adapters++">MoE-Adapters++</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MiN">MiN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MINGLE">MINGLE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/HotMoE">HotMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/R-MoE">R-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DynMoE">DynMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/EMoE">EMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoME">MoME</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoE_CL">MoE_CL</a></li>
                <li><a href="Extension_Research_Models/MoE_models/AVMOE">AVMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DeMoE">DeMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SMoPE">SMoPE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/GM-MoE">GM-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SPMTrack">SPMTrack</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SFAN">SFAN</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Forensic-MoE">Forensic-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/HyMoENet">HyMoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/APMoENet">APMoENet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MCMoE">MCMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoMKE">MoMKE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/CoMoE">CoMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/LPMoE">LPMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/EfficientMoE">EfficientMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/STAR">STAR</a></li>
                <li><a href="Extension_Research_Models/MoE_models/GeoMoE">GeoMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/AMOE">AMOE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DMDNet">DMDNet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/D-MoLE">D-MoLE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/PA-MoE">PA-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Gazeformer">Gazeformer</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Meta-Expert">Meta-Expert</a></li>
                <li><a href="Extension_Research_Models/MoE_models/ProMoE">ProMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MoETTA">MoETTA</a></li>
                <li><a href="Extension_Research_Models/MoE_models/WMNet">WMNet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/DMFusion">DMFusion</a></li>
                <li><a href="Extension_Research_Models/MoE_models/GRAM">GRAM</a></li>
                <li><a href="Extension_Research_Models/MoE_models/Point-MoE">Point-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/GazeMoE">GazeMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/CA-MoE">CA-MoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MEGADance">MEGADance</a></li>
                <li><a href="Extension_Research_Models/MoE_models/LE-ProtoPNet">LE-ProtoPNet</a></li>
                <li><a href="Extension_Research_Models/MoE_models/MedMoE">MedMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/M4Fuse">M4Fuse</a></li>
                <li><a href="Extension_Research_Models/MoE_models/StyleExpert">StyleExpert</a></li>
                <li><a href="Extension_Research_Models/MoE_models/SpecMoE">SpecMoE</a></li>
                <li><a href="Extension_Research_Models/MoE_models/VidPrism">VidPrism</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Mamba_models/MambaVision">MambaVision</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/TinyViM">TinyViM</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VM-UNet">VM-UNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/TransMamba">TransMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Vim">Vim</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Vamba">Vamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MobileMamba">MobileMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/DAMamba">DAMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/CLIMB-ReID">CLIMB-ReID</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Mamba-Reg">Mamba-Reg</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MambaOut">MambaOut</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MAP">MAP</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/ARM">ARM</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/LFTramba">LFTramba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/PoinTramba">PoinTramba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Tran-Mamba">Tran-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Dimba">Dimba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MxT">MxT</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/HMT-Unet">HMT-Unet</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VM-UNetv2">VM-UNetv2</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/AiM">AiM</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/CWNet">CWNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/FreqMamba">FreqMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/LocalMamba">LocalMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VMamba">VMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/RetinexRawMamba">RetinexRawMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/W-Mamba">W-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VMatcher">VMatcher</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/SP-Mamba">SP-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/M3amba">M3amba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Mamba3D">Mamba3D</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/HSRMamba">HSRMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/WalMaFa">WalMaFa</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Wave-Mamba">Wave-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/IRSRMamba">IRSRMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MambaIR">MambaIR</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/DFSSM">DFSSM</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/FreMamba">FreMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MambaLLIE">MambaLLIE</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VCMamba">VCMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/V2M">V2M</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/BMTNet">BMTNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/PRISM">PRISM</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/AdaSFFuse">AdaSFFuse</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MambaMoE">MambaMoE</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/SpectralVMamba">SpectralVMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/EAMamba">EAMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/WDMamba">WDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/CAB">CAB</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MambaHoME">MambaHoME</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/VSSD">VSSD</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/AFM-Net">AFM-Net</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/SFMFusion">SFMFusion</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/ReIDMamba">ReIDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MatMamba">MatMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/TPCM-SegNet">TPCM-SegNet</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/NDMamba">NDMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/MM-DETR">MM-DETR</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/TranSamba">TranSamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/ACMamba">ACMamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/CMFS">CMFS</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/UVM-Net">UVM-Net</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Dino-Mamba">Dino-Mamba</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/TimeViper">TimeViper</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/Mamba-3">Mamba-3</a></li>
                <li><a href="Extension_Research_Models/Mamba_models/EQ-VMamba">EQ-VMamba</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/KAN_models/EfficientKAN">Efficient KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/Vision-KAN">Vision KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/Fast-KAN">Fast KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/FourierKAN">Fourier KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/MoE-KAN">MoE KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KAT">KAT</a></li>
                <li><a href="Extension_Research_Models/KAN_models/GR_KAN">GR_KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KAN-COIN">KAN COIN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KM_UNet">KM_UNet</a></li>
                <li><a href="Extension_Research_Models/KAN_models/U-KAN">U-KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/MedViTV2">MedViTV2</a></li>
                <li><a href="Extension_Research_Models/KAN_models/RKAN">RKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/FewSound">FewSound</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KAN-AD">KAN-AD</a></li>
                <li><a href="Extension_Research_Models/KAN_models/LDSB">LDSB</a></li>
                <li><a href="Extension_Research_Models/KAN_models/cmKAN">cmKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KAC">KAC</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KArAt">KArAt</a></li>
                <li><a href="Extension_Research_Models/KAN_models/QKAN">QKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/KAN-IDIR">KAN-IDIR</a></li>
                <li><a href="Extension_Research_Models/KAN_models/JacobiKAN">JacobiKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/ChebyKAN">ChebyKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/GroupKAN">GroupKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/ABFR-KAN">ABFR-KAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/FasterKAN">FasterKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/DGKAN">DGKAN</a></li>
                <li><a href="Extension_Research_Models/KAN_models/ViK">ViK</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/INR_models/siren">siren</a></li>
                <li><a href="Extension_Research_Models/INR_models/CoLIE">CoLIE</a></li>
                <li><a href="Extension_Research_Models/INR_models/WIRE">WIRE</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeRCo">NeRCo</a></li>
                <li><a href="Extension_Research_Models/INR_models/GKAN">GKAN</a></li>
                <li><a href="Extension_Research_Models/INR_models/SCONE">SCONE</a></li>
                <li><a href="Extension_Research_Models/INR_models/LINR">LINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/FR-INR">FR-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeRD">NeRD</a></li>
                <li><a href="Extension_Research_Models/INR_models/INRAD">INRAD</a></li>
                <li><a href="Extension_Research_Models/INR_models/SineKAN">SineKAN</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeRP">NeRP</a></li>
                <li><a href="Extension_Research_Models/INR_models/MINER">MINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/O-INR">O-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/ImpSq">ImpSq</a></li>
                <li><a href="Extension_Research_Models/INR_models/ANR">ANR</a></li>
                <li><a href="Extension_Research_Models/INR_models/Partition-INR">Partition INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/COIN">COIN</a></li>
                <li><a href="Extension_Research_Models/INR_models/LoE">LoE</a></li>
                <li><a href="Extension_Research_Models/INR_models/FreSh">FreSh</a></li>
                <li><a href="Extension_Research_Models/INR_models/RFF">RFF</a></li>
                <li><a href="Extension_Research_Models/INR_models/FINER">FINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/FINER++">FINER++</a></li>
                <li><a href="Extension_Research_Models/INR_models/xinc">xinc</a></li>
                <li><a href="Extension_Research_Models/INR_models/Deblur_INR">Deblur-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/Trans-INR">Trans-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/GINR-IPC">GINR-IPC</a></li>
                <li><a href="Extension_Research_Models/INR_models/Local-Global-INR">Local-Global-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeRV">NeRV</a></li>
                <li><a href="Extension_Research_Models/INR_models/HNeRV">HNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_models/NerPE">NerPE</a></li>
                <li><a href="Extension_Research_Models/INR_models/IRL-INR">IRL-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/DI-INR">DI-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/LIIF">LIIF</a></li>
                <li><a href="Extension_Research_Models/INR_models/awesome">awesome</a></li>
                <li><a href="Extension_Research_Models/INR_models/QIREN">QIREN</a></li>
                <li><a href="Extension_Research_Models/INR_models/COMBINER">COMBINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/RECOMBINER">RECOMBINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/TAF">TAF</a></li>
                <li><a href="Extension_Research_Models/INR_models/nir">nir</a></li>
                <li><a href="Extension_Research_Models/INR_models/INR-Harmonization">INR-Harmonization</a></li>
                <li><a href="Extension_Research_Models/INR_models/BDINR">BDINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/VideoINR">VideoINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/PFNR">PFNR</a></li>
                <li><a href="Extension_Research_Models/INR_models/DINER">DINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/TSINR">TSINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/CQ-NIR">CQ-NIR</a></li>
                <li><a href="Extension_Research_Models/INR_models/Sobolev_INRs">Sobolev_INRs</a></li>
                <li><a href="Extension_Research_Models/INR_models/Poly_INR">Poly_INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/INR-ST">INR-ST</a></li>
                <li><a href="Extension_Research_Models/INR_models/O2V">O2V</a></li>
                <li><a href="Extension_Research_Models/INR_models/SketchINR">SketchINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/INR-ECGAN">INR-ECGAN</a></li>
                <li><a href="Extension_Research_Models/INR_models/LTE">LTE</a></li>
                <li><a href="Extension_Research_Models/INR_models/HNET">HNET</a></li>
                <li><a href="Extension_Research_Models/INR_models/LTEW">LTEW</a></li>
                <li><a href="Extension_Research_Models/INR_models/PartSDF">PartSDF</a></li>
                <li><a href="Extension_Research_Models/INR_models/Spherical-INR">Spherical-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/UncertaINR">UncertaINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/LosslessINR">LosslessINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/DDiF">DDiF</a></li>
                <li><a href="Extension_Research_Models/INR_models/SAPE">SAPE</a></li>
                <li><a href="Extension_Research_Models/INR_models/Symmetric-Power-Transformation-INR">Symmetric-Power-Transformation-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/MWT">MWT</a></li>
                <li><a href="Extension_Research_Models/INR_models/HiNeRV">HiNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_models/NG">NG</a></li>
                <li><a href="Extension_Research_Models/INR_models/NFN">NFN</a></li>
                <li><a href="Extension_Research_Models/INR_models/EVOS-INR">EVOS-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/INT">INT</a></li>
                <li><a href="Extension_Research_Models/INR_models/FAOR">FAOR</a></li>
                <li><a href="Extension_Research_Models/INR_models/SNF">SNF</a></li>
                <li><a href="Extension_Research_Models/INR_models/RPP">RPP</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeuralSVG">NeuralSVG</a></li>
                <li><a href="Extension_Research_Models/INR_models/STINR">STINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/FeINFN">FeINFN</a></li>
                <li><a href="Extension_Research_Models/INR_models/Spener">Spener</a></li>
                <li><a href="Extension_Research_Models/INR_models/Moner">Moner</a></li>
                <li><a href="Extension_Research_Models/INR_models/VI3NR">VI3NR</a></li>
                <li><a href="Extension_Research_Models/INR_models/PhysINR">PhysINR</a></li>
                <li><a href="Extension_Research_Models/INR_models/IGA-INR">IGA-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/MetricGrids">MetricGrids</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeuRBF">NeuRBF</a></li>
                <li><a href="Extension_Research_Models/INR_models/HyperNVD">HyperNVD</a></li>
                <li><a href="Extension_Research_Models/INR_models/SL2A-INR">SL2A-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeRT">NeRT</a></li>
                <li><a href="Extension_Research_Models/INR_models/Implicit-Zoo">Implicit-Zoo</a></li>
                <li><a href="Extension_Research_Models/INR_models/INRFlow">INRFlow</a></li>
                <li><a href="Extension_Research_Models/INR_models/Transformer-NFN">Transformer-NFN</a></li>
                <li><a href="Extension_Research_Models/INR_models/SIREN-square">SIREN-square</a></li>
                <li><a href="Extension_Research_Models/INR_models/INF">INF</a></li>
                <li><a href="Extension_Research_Models/INR_models/FFN">FFN</a></li>
                <li><a href="Extension_Research_Models/INR_models/INR-benchmark">INR-benchmark</a></li>
                <li><a href="Extension_Research_Models/INR_models/Residual-INR">Residual-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/STAF">STAF</a></li>
                <li><a href="Extension_Research_Models/INR_models/INCODE">INCODE</a></li>
                <li><a href="Extension_Research_Models/INR_models/FFNeRV">FFNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_models/Neural-Knitworks">Neural-Knitworks</a></li>
                <li><a href="Extension_Research_Models/INR_models/IDIR">IDIR</a></li>
                <li><a href="Extension_Research_Models/INR_models/MetaSeg">MetaSeg</a></li>
                <li><a href="Extension_Research_Models/INR_models/CLIT">CLIT</a></li>
                <li><a href="Extension_Research_Models/INR_models/STRAINER">STRAINER</a></li>
                <li><a href="Extension_Research_Models/INR_models/SUICA">SUICA</a></li>
                <li><a href="Extension_Research_Models/INR_models/INR-Bench">INR-Bench</a></li>
                <li><a href="Extension_Research_Models/INR_models/NeuroQuant">NeuroQuant</a></li>
                <li><a href="Extension_Research_Models/INR_models/CST-Net">CST-Net</a></li>
                <li><a href="Extension_Research_Models/INR_models/LINR-PCGC">LINR-PCGC</a></li>
                <li><a href="Extension_Research_Models/INR_models/Split-Layer">Split-Layer</a></li>
                <li><a href="Extension_Research_Models/INR_models/MFSR-CD">MFSR-CD</a></li>
                <li><a href="Extension_Research_Models/INR_models/COIN++">COIN++</a></li>
                <li><a href="Extension_Research_Models/INR_models/REFINE-MORE">REFINE-MORE</a></li>
                <li><a href="Extension_Research_Models/INR_models/DINo">DINo</a></li>
                <li><a href="Extension_Research_Models/INR_models/HUVR">HUVR</a></li>
                <li><a href="Extension_Research_Models/INR_models/DInf-Grid">DInf-Grid</a></li>
                <li><a href="Extension_Research_Models/INR_models/ASSR">ASSR</a></li>
                <li><a href="Extension_Research_Models/INR_models/LGE_INR">LGE_INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/CNS_INR">CNS_INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/SIR-SNC">SIR-SNC</a></li>
                <li><a href="Extension_Research_Models/INR_models/NINT">NINT</a></li>
                <li><a href="Extension_Research_Models/INR_models/CAFE">CAFE</a></li>
                <li><a href="Extension_Research_Models/INR_models/Lipschitz-INR">Lipschitz-INR</a></li>
                <li><a href="Extension_Research_Models/INR_models/TorchLip">TorchLip</a></li>
                <li><a href="Extension_Research_Models/INR_models/TeCoNeRV">TeCoNeRV</a></li>
                <li><a href="Extension_Research_Models/INR_models/Equivariant-ASISR">Equivariant-ASISR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/LoRA">LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/PEVC">PEVC</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/MLoRE">MLoRE</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/LoRA-IR">LoRA-IR</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/CLIP-LoRA">CLIP-LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/LoME">LoME</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/MoRA">MoRA</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/DeLoRA">DeLoRA</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/ASI">ASI</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/DyLoRA-MoE">DyLoRA-MoE</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/SR-LoRA">SR-LoRA</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/LoRA-ViT">LoRA-ViT</a></li>
                <li><a href="Extension_Research_Models/Low_rank_decomposition_models/BiLaLoRA">BiLaLoRA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Neural_operator_models/FNO">Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/UNO">UNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/U-FNO">U-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/G-FNO">G-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/AFNO">AFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/HiNOTE">HiNOTE</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SRNO">SRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FourierImaging">Fourier Imaging</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/Galerkin">Galerkin-type Self-attention</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FFNO">Factorized Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NeuralOP">NeuralOP</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/LatentNeuralOperator">Latent Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/WNO">Wavelet Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DQFNO">Derived-Quanities Fourier Neural Operator</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FNOMultiSizedImages">FNO Multi Sized Images</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/AAAI_FNO_Tutorial">AAAI FNO Tutorial</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/MAWNO">MAWNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NeurOp-Diff">NeurOp-Diff</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/AMG">AMG</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/CoNO">CoNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FNOReg">FNOReg</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/CGH-SFO">CGH-SFO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/AFFNet">AFFNet</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DAFNO">DAFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/Geo-FNO">Geo-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/ResFNO">ResFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/GNOT">GNOT</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SSRNO">SSRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/INN">INN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FNIN">FNIN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/UNet-FNO">UNet-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/WDNO">WDNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SFNO">SFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/GANO">GANO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NewtonFNO">NewtonFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PITA">PITA</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DPOT">DPOT</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PINO">PINO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/VINO">VINO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/GKN">GKN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PAC-FNO">PAC-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PDIO">PDIO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/MambaNO">MambaNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/muTransfer-FNO">muTransfer-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/RNO">RNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/iFNO">iFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/UNOI">UNOI</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/CNO">CNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/HNOSeg-XS">HNOSeg-XS</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/JPNeO">JPNeO-XS</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NeurOLight">NeurOLight</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PastNet">PastNet</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PINN_FP64">PINN_FP64</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/TANTE">TANTE</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/Tucker-FNO">Tucker-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/AFNO">AFNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DINOZAUR">DINOZAUR</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/STVSR-NO">STVSR-NO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/INO">INO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/OpFlow">OpFlow</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PODNO">PODNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/MINO">MINO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/HyPINO">HyPINO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/RINN">RINN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SPL_OFM">SPL_OFM</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/MoEPOT">MoEPOT</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SoFoNO">SoFoNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FNOPE">FNOPE</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/FlowGINO">FlowGINO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/MMGN">MMGN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/Sonnet">Sonnet</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NeuralKoopmanSVD">NeuralKoopmanSVD</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PINN">PINN</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/PINNs-Torch">PINNs-Torch</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/R-FNO">R-FNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/SONIC">SONIC</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/IRNO">IRNO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DisentangO">DisentangO</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/DRIFT-Net">DRIFT-Net</a></li>
                <li><a href="Extension_Research_Models/Neural_operator_models/NOIR">NOIR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SpikeYOLO">SpikeYOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/EMS-YOLO">EMS-YOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SpikingYOLO">SpikingYOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/EOLO">EOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/RENet">RENet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/DSEC-MOS">DSEC-MOS</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/RVT">RVT</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/LEOD">LEOD</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SSM">SSM</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SMamba">SMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/eTraM">eTraM</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SpikingYOLOX">SpikingYOLOX</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/MvHeatDET">MvHeatDET</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/CvHeat-DET">CvHeat-DET</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/DAGr">DAGr</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SU-YOLO">SU-YOLO</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/CREST">CREST</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/DEOE">DEOE</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/SNNTracker">SNNTracker</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/Advanced-SpikingYOLOX">Advanced-SpikingYOLOX</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_detection/TDE">TDE</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/LISNN">LISNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/E-SpikeFormer">E-SpikeFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Meta-SpikeFormer">Meta-SpikeFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Spike-driven">Spike-driven</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Spikformer">Spikformer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/QKFormer">QKFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SEW_ResNet">SEW_ResNet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SpikingResformer">SpikingResformer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SEMM">SEMM</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Spikingformer">Spikingformer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/CML">CML</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SGLFormer">SGLFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/NDA_SNN">NDA_SNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/MS-ResNet">MS-ResNet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SWformer">SWformer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/BKDSNN">BKDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Event_Modality_Application">Event_Modality_Application</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EventCLIP">EventCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EST">EST</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EDP">EDP</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/n_imagenet">n_imagenet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SpikeCLIP">SpikeCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/NeuroCLIP">NeuroCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SLAYER">SLAYER</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/T-RevSNN">T-RevSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/GAC">GAC</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SpikeVideoFormer">SpikeVideoFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SpikMamba">SpikMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/ExACT">ExACT</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EventBind">EventBind</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/NVS2Graph">NVS2Graph</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/ANN2SNN_NQ">ANN2SNN_NQ</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/MambaPAR">MambaPAR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SNN-PAR">SNN-PAR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/VTB">VTB</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EventMamba">EventMamba</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/TET">TET</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/STMixer">STMixer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/MTT">MTT</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SML">SML</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/MaxFormer">MaxFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Masked-Spiking-Transformer">Masked-Spiking-Transformer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/AT-LIF">AT-LIF</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/TRSG">TRSG</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/CKDSNN">CKDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/Mamba-Spike">Mamba-Spike</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SpiLiFormer">SpiLiFormer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/BSDSNN">BSDSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/PseudoSNN">PseudoSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/DSD">DSD</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/STOD">STOD</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/SGC">SGC</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/BSD">BSD</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/DeepEISNN">DeepEISNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/EnhSNN">EnhSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_recognition/MSViT">MSViT</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/e2vid">e2vid</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/EvINR">EvINR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/ESDNet">ESDNet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/SpikerIR">SpikerIR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/NER">NER</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/EventGAN">EventGAN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/EHN">EHN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/EVSNN">EVSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/SpikeCLIP">SpikeCLIP</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/INR-Event-VSR">INR-Event-VSR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/PRE-Mamba">PRE-Mamba</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/DepthAnyEvent">DepthAnyEvent</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/e2depth">e2depth</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/Spike-FlowNet">Spike-FlowNet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/RSIR">RSIR</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/e2flow">e2flow</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/SpikeDiffusion">SpikeDiffusion</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/EvEnhancer">EvEnhancer</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/DehazeSNN">DehazeSNN</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Reconstruction/VLIF">VLIF</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/Gen1_toolbox">Gen1_toolbox</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/V2CE">V2CE</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/V2E">V2E</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/Vid2E">Vid2E</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/S2R">S2R</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/IEBCS">IEBCS</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Toolbox/I2E">I2E</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Anomaly_detection/UCF-Crime-DVS">UCF-Crime-DVS</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Anomaly_detection/ISSAFE">ISSAFE</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Anomaly_detection/EventAD">EventAD</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_segmentation/Spike2Former">Spike2Former</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_segmentation/SLTNet">SLTNet</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_segmentation/ECOS">ECOS</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_segmentation/Spiking-DeepLab">Spiking-DeepLab</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Object_segmentation/MambaSeg">MambaSeg</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Action_recognition/HARDVS">HARDVS</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/Action_recognition/HARDVSv2">HARDVSv2</a></li>
                <li><a href="Extension_Research_Models/SNN_event_models/PointCloud/SPM">SPM</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Efficient_models/mincam">mincam</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/NdLinear">NdLinear</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/GlobustVP">GlobustVP</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/minGRU">minGRU</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/OFA">OFA</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileOne">MobileOne</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileIE">MobileIE</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/Turbo-VAED">Turbo-VAED</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/PreciseRoIPooling">PreciseRoIPooling</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/ortho-residual">ortho-residual</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileNetV4">MobileNetV4</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileNetV3">MobileNetV3</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileNetV2">MobileNetV2</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileNetV1">MobileNetV1</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/RepViT">RepViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/EfficientFormer">EfficientFormer</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/LeViT">LeViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileViG">MobileViG</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/MobileViGv2">MobileViGv2</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/GreedyViG">GreedyViG</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/Mobile-U-ViT">Mobile-U-ViT</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/AB-BNN">AB-BNN</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/BPAM">BPAM</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/LGM">LGM</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/Lottery-Ticket-Hypothesis">Lottery-Ticket-Hypothesis</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/mHC">mHC</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/Titans">Titans</a></li>
                <li><a href="Extension_Research_Models/Efficient_models/Flash-KMeans">Flash-KMeans</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/VLM_models/CoOp">CoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CLIP_Surgery">CLIP_Surgery</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PromptKD">PromptKD</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MCM">MCM</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CLIP-KD">CLIP-KD</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MaPLe">MaPLe</a></li>
                <li><a href="Extension_Research_Models/VLM_models/DPLQ">DPLQ</a></li>
                <li><a href="Extension_Research_Models/VLM_models/DiffCLIP">DiffCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/DualPrompt">DualPrompt</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PromptSRC">PromptSRC</a></li>
                <li><a href="Extension_Research_Models/VLM_models/DualCoOp">DualCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MMRL">MMRL</a></li>
                <li><a href="Extension_Research_Models/VLM_models/Safe-CLIP">Safe-CLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CLIP-Refine">CLIP-Refine</a></li>
                <li><a href="Extension_Research_Models/VLM_models/Tip-Adapter">Tip-Adapter</a></li>
                <li><a href="Extension_Research_Models/VLM_models/TrustVLM">TrustVLM</a></li>
                <li><a href="Extension_Research_Models/VLM_models/Bayesian-Prompt-Learning">Bayesian-Prompt-Learning</a></li>
                <li><a href="Extension_Research_Models/VLM_models/Nano-vLLM">Nano-vLLM</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PointCLIP">PointCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PromptFix">PromptFix</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CAT-Seg">CAT-Seg</a></li>
                <li><a href="Extension_Research_Models/VLM_models/DMN">DMN</a></li>
                <li><a href="Extension_Research_Models/VLM_models/FCPrompt">FCPrompt</a></li>
                <li><a href="Extension_Research_Models/VLM_models/SurPL">SurPL</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MITLOp-CCAug">MITLOp-CCAug</a></li>
                <li><a href="Extension_Research_Models/VLM_models/GCSCoOp">GCSCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PromptAlign">PromptAlign</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CasPL">CasPL</a></li>
                <li><a href="Extension_Research_Models/VLM_models/TAP">TAP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/AToken">AToken</a></li>
                <li><a href="Extension_Research_Models/VLM_models/KgCoOp">KgCoOp</a></li>
                <li><a href="Extension_Research_Models/VLM_models/ProGrad">ProGrad</a></li>
                <li><a href="Extension_Research_Models/VLM_models/CLIPstyler">CLIPstyler</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MSAE">MSAE</a></li>
                <li><a href="Extension_Research_Models/VLM_models/Mobile-VideoGPT">Mobile-VideoGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_models/NXTP">NXTP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/minGPT">minGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_models/iGPT">iGPT</a></li>
                <li><a href="Extension_Research_Models/VLM_models/AdaptVision">AdaptVision</a></li>
                <li><a href="Extension_Research_Models/VLM_models/ICL">ICL</a></li>
                <li><a href="Extension_Research_Models/VLM_models/SuperCLIP">SuperCLIP</a></li>
                <li><a href="Extension_Research_Models/VLM_models/MoPD">MoPD</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PAE">PAE</a></li>
                <li><a href="Extension_Research_Models/VLM_models/PDA">PDA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Neural_ODE_models/CLODE">CLODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/DeepRUOT">DeepRUOT</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/NeuralODE">NeuralODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/SDE">SDE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/SNOpt">SNOpt</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/RNODE">RNODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/ANODE">ANODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/GAINS">GAINS</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/DCN">DCN</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/NODE-ImgNet">NODE-ImgNet</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/DMF">DMF</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/NODEO">NODEO</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/NODE-GP">NODE-GP</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/CLE">CLE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/ResNet_NODE">ResNet_NODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/HomoODE">HomoODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/MetaNODE">MetaNODE</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/Bi-flow">Bi-flow</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/HazeFlow">HazeFlow</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/DEQ">DEQ</a></li>
                <li><a href="Extension_Research_Models/Neural_ODE_models/FLUX-IR">FLUX-IR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/OT_models/Hyperbolic_Alignment">Hyperbolic Alignment</a></li>
                <li><a href="Extension_Research_Models/OT_models/PLOT">PLOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/DUCT">DUCT</a></li>
                <li><a href="Extension_Research_Models/OT_models/OTFusion">OT Fusion</a></li>
                <li><a href="Extension_Research_Models/OT_models/TransformerFusion">TransformerFusion</a></li>
                <li><a href="Extension_Research_Models/OT_models/Intra-Fusion">Intra-Fusion</a></li>
                <li><a href="Extension_Research_Models/OT_models/LAVCap">LAVCap</a></li>
                <li><a href="Extension_Research_Models/OT_models/Tree-Diffusion-Schrodinger-Bridge">Tree-based Diffusion Schrödinger Bridge</a></li>
                <li><a href="Extension_Research_Models/OT_models/OTCE">OTCE</a></li>
                <li><a href="Extension_Research_Models/OT_models/TSA-MLT">TSA-MLT</a></li>
                <li><a href="Extension_Research_Models/OT_models/OT-VP">OT-VP</a></li>
                <li><a href="Extension_Research_Models/OT_models/LICO">LICO</a></li>
                <li><a href="Extension_Research_Models/OT_models/SinkhornAutoDiff">SinkhornAutoDiff</a></li>
                <li><a href="Extension_Research_Models/OT_models/FedOTP">FedOTP</a></li>
                <li><a href="Extension_Research_Models/OT_models/ALIGN">ALIGN</a></li>
                <li><a href="Extension_Research_Models/OT_models/TCL">TCL</a></li>
                <li><a href="Extension_Research_Models/OT_models/DebiasedOT">DebiasedOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/MGPATH">MGPATH</a></li>
                <li><a href="Extension_Research_Models/OT_models/COT">COT</a></li>
                <li><a href="Extension_Research_Models/OT_models/APTP">APTP</a></li>
                <li><a href="Extension_Research_Models/OT_models/HiWA">HiWA</a></li>
                <li><a href="Extension_Research_Models/OT_models/MM-Align">MM-Align</a></li>
                <li><a href="Extension_Research_Models/OT_models/AWT">AWT</a></li>
                <li><a href="Extension_Research_Models/OT_models/PROP">PROP</a></li>
                <li><a href="Extension_Research_Models/OT_models/L2RM">L2RM</a></li>
                <li><a href="Extension_Research_Models/OT_models/KPG-RL">KPG-RL</a></li>
                <li><a href="Extension_Research_Models/OT_models/OSSL">OSSL</a></li>
                <li><a href="Extension_Research_Models/OT_models/Prompt-OT">Prompt-OT</a></li>
                <li><a href="Extension_Research_Models/OT_models/NLPrompt">NLPrompt</a></li>
                <li><a href="Extension_Research_Models/OT_models/AffScoreDeep">AffScoreDeep</a></li>
                <li><a href="Extension_Research_Models/OT_models/OTSeg">OTSeg</a></li>
                <li><a href="Extension_Research_Models/OT_models/Sinkformers">Sinkformers</a></li>
                <li><a href="Extension_Research_Models/OT_models/DiffusionSchrodingerBridge">DiffusionSchrodingerBridge</a></li>
                <li><a href="Extension_Research_Models/OT_models/LightSB-Matching">LightSB-Matching</a></li>
                <li><a href="Extension_Research_Models/OT_models/LightSB">LightSB</a></li>
                <li><a href="Extension_Research_Models/OT_models/OTTER">OTTER</a></li>
                <li><a href="Extension_Research_Models/OT_models/EntropicOTBenchmark">EntropicOTBenchmark</a></li>
                <li><a href="Extension_Research_Models/OT_models/RAM">RAM</a></li>
                <li><a href="Extension_Research_Models/OT_models/DbTSW">DbTSW</a></li>
                <li><a href="Extension_Research_Models/OT_models/ASOT">ASOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/TOT">TOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/DoRA">DoRA</a></li>
                <li><a href="Extension_Research_Models/OT_models/MFSWB">MFSWB</a></li>
                <li><a href="Extension_Research_Models/OT_models/DA-RCOT">DA-RCOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/TRIP">TRIP</a></li>
                <li><a href="Extension_Research_Models/OT_models/REMOTE">REMOTE</a></li>
                <li><a href="Extension_Research_Models/OT_models/UniOT">UniOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/UNSB">UNSB</a></li>
                <li><a href="Extension_Research_Models/OT_models/CUNSB-RFIE">CUNSB-RFIE</a></li>
                <li><a href="Extension_Research_Models/OT_models/DehazeSB">DehazeSB</a></li>
                <li><a href="Extension_Research_Models/OT_models/LaCoOT">LaCoOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/CatSBench">CatSBench</a></li>
                <li><a href="Extension_Research_Models/OT_models/BaryIR">BaryIR</a></li>
                <li><a href="Extension_Research_Models/OT_models/RCOT">RCOT</a></li>
                <li><a href="Extension_Research_Models/OT_models/SOT-GLP">SOT-GLP</a></li>
                <li><a href="Extension_Research_Models/OT_models/SOTA">SOTA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Flow_matching_models/LieFlow">LieFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/FM">FM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/PnP-Flow">PnP-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/SemFlow">SemFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/Data-Uncertainty">Data-Uncertainty</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/FlowMatching">FlowMatching</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/EqM">EqM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/FFM">FFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/DeltaFM">DeltaFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/Flow-Guidance">Flow-Guidance</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/MeanFlow">MeanFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/TCCM">TCCM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/RFM">RFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/ATFM">ATFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/Restora-Flow">Restora-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/BFM">BFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/PCFM">PCFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/FERNN">FERNN</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/SoFlow">SoFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/flowNP">flowNP</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/PBFM">PBFM</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/DegFlow">DegFlow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/Self-Flow">Self-Flow</a></li>
                <li><a href="Extension_Research_Models/Flow_matching_models/VGG-Flow">VGG-Flow</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Visualization_models/Finer-CAM">Finer-CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/VCC">VCC</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/MoXI">MoXI</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/Prompt_CAM">Prompt_CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/CLIP-dissect">CLIP-dissect</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/WWW">WWW</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/Transformer-Explainability">Transformer-Explainability</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/MAMMI">MAMMI</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/SIC">SIC</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/NWHead">NWHead</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/ProTeCt">ProTeCt</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/B-cos-v2">B-cos-v2</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/VPS">VPS</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/FIxLIP">FIxLIP</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/Grad-Eclip">Grad-Eclip</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/exCLIP">exCLIP</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/Lucent">Lucent</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/TAM">TAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/CE-FAM">CE-FAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/FG-VCE">FG-VCE</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/EPIC">EPIC</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/VX-CODE">VX-CODE</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/ConceptAttention">ConceptAttention</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/MaskInversion">MaskInversion</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/Fusion-CAM">Fusion-CAM</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/FaCT">FaCT</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/DEX-AR">DEX-AR</a></li>
                <li><a href="Extension_Research_Models/Visualization_models/SCAN">SCAN</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_captioning_models/describe-anything">describe-anything</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/KGG">KGG</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/kg-gen">kg-gen</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/Cosmos-Reason2">Cosmos-Reason2</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/CogVLM2">CogVLM2</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/CogVLM">CogVLM</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/Qwen3-VL">Qwen3-VL</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/LLaVA">LLaVA</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/LLaVA-3D">LLaVA-3D</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/LLaVA-Grounding">LLaVA-Grounding</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/MiniGPT-4">MiniGPT-4</a></li>
                <li><a href="Extension_Research_Models/Image_captioning_models/Youtu-VL">Youtu-VL</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/mean-teacher">mean-teacher</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/MM-DistillNet">MM-DistillNet</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/CSKD">CSKD</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/Manifold_Distillation">Manifold_Distillation</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/CMT">CMT</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/DM2TNet">DM2TNet</a></li>
                <li><a href="Extension_Research_Models/Knowledge_distillation_models/LMT-GP">LMT-GP</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Domain_generalization_models/L2D">L2D</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/DANN">DANN</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/D_GCD">D_GCD</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/PromptStyler">PromptStyler</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/DPStyler">DPStyler</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/DFCF">DFCF</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/PromptTA">PromptTA</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/SFDG">SFDG</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/RobustNet">RobustNet</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/ABA">ABA</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/ALT">ALT</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/DDG">DDG</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/NeuralLio">NeuralLio</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/ASRNorm">ASRNorm</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/PDEN">PDEN</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/SFDA2">SFDA2</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/MJDOT">MJDOT</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/Fish">Fish</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/ZeroShotDG">ZeroShotDG</a></li>
                <li><a href="Extension_Research_Models/Domain_generalization_models/M-ADA">M-ADA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Data_augmentation_models/AFA">AFA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/HybridAugment++">HybridAugment++</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/SPM">SPM</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/VIPAug">VIPAug</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/DCAug">DCAug</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/PhysAug">PhysAug</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/RainNet">RainNet</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/InstanceAugmentation">InstanceAugmentation</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/Polar_Augment">Polar_Augment</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/NMCB">NMCB</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/FastAutoAugment">FastAutoAugment</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/RandAugment">RandAugment</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/Analytical-Learning-Theory">Analytical-Learning-Theory</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/On-Device-DG">On-Device-DG</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/MixStyle">MixStyle</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/XDomainMix">XDomainMix</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/RandConv">RandConv</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/SagNet">SagNet</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/C-UDA">C-UDA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/LEAwareSGD">LEAwareSGD</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/ConDiSR">ConDiSR</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/StyDeSty">StyDeSty</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/ADA">ADA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/EFDM">EFDM</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/AdaIN">AdaIN</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/CCDR">CCDR</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/FIXED">FIXED</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/OpenDG-DAML">OpenDG-DAML</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/LearnAug-UDA">LearnAug-UDA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/TeachAugment">TeachAugment</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/TTSA">TTSA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/TSA">TSA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/DoGE">DoGE</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/MIXALL">MIXALL</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/FreeSDG">FreeSDG</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/UDA-norm-VAE">UDA-norm-VAE</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/OnlineAugment">OnlineAugment</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/FATA">FATA</a></li>
                <li><a href="Extension_Research_Models/Data_augmentation_models/Mixup_for_UDA">Mixup_for_UDA</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Object_detection_models/UniFS">UniFS</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/OLN">OLN</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/SR4IR">SR4IR</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/DEIMv2">DEIMv2</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/Rex-Omni">Rex-Omni</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/AutoEval">AutoEval</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/GroundingDINO">GroundingDINO</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/Open-GroundingDino">Open-GroundingDino</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/GSDet">GSDet</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/DEQDet">DEQDet</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/YOLO-Master">YOLO-Master</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/YOLO-IOD">YOLO-IOD</a></li>
                <li><a href="Extension_Research_Models/Object_detection_models/SAM-DETR">SAM-DETR</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Physics_models/PSF-Estimation">PSF-Estimation</a></li>
                <li><a href="Extension_Research_Models/Physics_models/MANO">MANO</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhyDNet">PhyDNet</a></li>
                <li><a href="Extension_Research_Models/Physics_models/vHeat">vHeat</a></li>
                <li><a href="Extension_Research_Models/Physics_models/CIConv">CIConv</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhyDAE">PhyDAE</a></li>
                <li><a href="Extension_Research_Models/Physics_models/NLOS-LPP">NLOS-LPP</a></li>
                <li><a href="Extension_Research_Models/Physics_models/QuantiPhy">QuantiPhy</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhysVideoGenerator">PhysVideoGenerator</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhysCtrl">PhysCtrl</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhysDreamer">PhysDreamer</a></li>
                <li><a href="Extension_Research_Models/Physics_models/NCLaw">NCLaw</a></li>
                <li><a href="Extension_Research_Models/Physics_models/NewtonGen">NewtonGen</a></li>
                <li><a href="Extension_Research_Models/Physics_models/STDDN">STDDN</a></li>
                <li><a href="Extension_Research_Models/Physics_models/Goal-Force">Goal-Force</a></li>
                <li><a href="Extension_Research_Models/Physics_models/PhysicEdit">PhysicEdit</a></li>
                <li><a href="Extension_Research_Models/Physics_models/RealWonder">RealWonder</a></li>
                <li><a href="Extension_Research_Models/Physics_models/UWEnhancer">UWEnhancer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Representation_learning_models/ICon">ICon</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/MRL">MRL</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/Pigment_Representation">Pigment_Representation</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/VAC">VAC</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/RotationContinuity">RotationContinuity</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/UAE">UAE</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/Platonic">Platonic</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/KnowCoL">KnowCoL</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/Midway">Midway</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/FlowFeat">FlowFeat</a></li>
                <li><a href="Extension_Research_Models/Representation_learning_models/Compositional-Generalization">Compositional-Generalization</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/RWKV_models/CRWKV">CRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/Vision-RWKV">Vision-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/Restore-RWKV">Restore-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RWKV-UNet">RWKV-UNet</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/VisualRWKV">VisualRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RWKV-IR">RWKV-IR</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/Diffusion-RWKV">Diffusion-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/LION">LION</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/OmniRWKVSR">OmniRWKVSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/URWKV">URWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/LALIC">LALIC</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/FEAT">FEAT</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/ScaleMatch">ScaleMatch</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/HFE-RWKV">HFE-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/OccRWKV">OccRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/LF-RWKVNet">LF-RWKVNet</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/EventPAR">EventPAR</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/ChangeRWKV">ChangeRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RWKV-VG">RWKV-VG</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RetinexRWKV">RetinexRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RWKVFusion">RWKVFusion</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RSRWKV">RSRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/GDSR">GDSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RWKVSR">RWKVSR</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/FRWKV">FRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/Fourier-RWKV">Fourier-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/Hi-RWKV">Hi-RWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/TS-Pan">TS-Pan</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/DIN-GEBC">DIN-GEBC</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/RawRWKV">RawRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/DRWKV">DRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/MMSegRWKV">MMSegRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/LHIC-BP">LHIC-BP</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/LineRWKV">LineRWKV</a></li>
                <li><a href="Extension_Research_Models/RWKV_models/PointRWKV">PointRWKV</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Referring_expression_models/RMOT">RMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/CRMOT">CRMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/TempRMOT">TempRMOT</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/RexSeek">RexSeek</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/3EED">3EED</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/WildRefer">WildRefer</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/Dynamic-MDETR">Dynamic-MDETR</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/TransVG">TransVG</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/RefDrone">RefDrone</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/SOREC">SOREC</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/Zeroshot_REC">Zeroshot_REC</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/RefCrowd">RefCrowd</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/OmniSegNet">OmniSegNet</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/ReLA">ReLA</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/GPT4RoI">GPT4RoI</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/DKGTrack">DKGTrack</a></li>
                <li><a href="Extension_Research_Models/Referring_expression_models/FlowRVS">FlowRVS</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/SARD">SARD</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/EGTR">EGTR</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/USGG">USGG</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/sg2im">sg2im</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Graph-RCNN">Graph-RCNN</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/OvSGTR">OvSGTR</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Seg-Grounded">Seg-Grounded</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/OpenPSG">OpenPSG</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Pair-Net">Pair-Net</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/VLPrompt">VLPrompt</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/HiLo">HiLo</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Fair-PSGG">Fair-PSGG</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Scene-Graph-ViT">Scene-Graph-ViT</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/REACT">REACT</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/PL">PL</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/TDE">TDE</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/PENET">PENET</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/TopoNet">TopoNet</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/T2SG">T2SG</a></li>
                <li><a href="Extension_Research_Models/Scene_graph_generation_models/Salience-SGG">Salience-SGG</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/World_models/I-JEPA">I-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/V-JEPA">V-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/V-JEPAv2">V-JEPAv2</a></li>
                <li><a href="Extension_Research_Models/World_models/JEPA_SSL">JEPA_SSL</a></li>
                <li><a href="Extension_Research_Models/World_models/PiLaMIM">PiLaMIM</a></li>
                <li><a href="Extension_Research_Models/World_models/SCOTT">SCOTT</a></li>
                <li><a href="Extension_Research_Models/World_models/D-JEPA">D-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/Point-JEPA">Point-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/CNN-JEPA">CNN-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/NWM-NWM">CNN-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/DINO-Foresight">DINO-Foresight</a></li>
                <li><a href="Extension_Research_Models/World_models/FUTURIST">FUTURIST</a></li>
                <li><a href="Extension_Research_Models/World_models/JEPA-T">JEPA-T</a></li>
                <li><a href="Extension_Research_Models/World_models/seq-JEPA">seq-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/LeJEPA">LeJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/FlowJEPA">FlowJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/DMT-JEPA">DMT-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/JEPA-Physics">JEPA-Physics</a></li>
                <li><a href="Extension_Research_Models/World_models/DSeq-JEPA">DSeq-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/TesserAct">TesserAct</a></li>
                <li><a href="Extension_Research_Models/World_models/NanoJEPA">NanoJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/LingBot-World">LingBot-World</a></li>
                <li><a href="Extension_Research_Models/World_models/EB_JEPA">EB_JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/Rectified_LpJEPA">Rectified_LpJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/EchoJEPA">EchoJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/Newton-Kepler">Newton-Kepler</a></li>
                <li><a href="Extension_Research_Models/World_models/Infinite-World">Infinite-World</a></li>
                <li><a href="Extension_Research_Models/World_models/C-JEPA">C-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/NeoVerse">NeoVerse</a></li>
                <li><a href="Extension_Research_Models/World_models/WorldGrow">WorldGrow</a></li>
                <li><a href="Extension_Research_Models/World_models/Social-JEPA">Social-JEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/BiJEPA">BiJEPA</a></li>
                <li><a href="Extension_Research_Models/World_models/HJEPA-MoE">HJEPA-MoE</a></li>
                <li><a href="Extension_Research_Models/World_models/SAT-JEPA-DIFF">SAT-JEPA-DIFF</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Hopfield_models/Hopfield-Layers">Hopfield-Layers</a></li>
                <li><a href="Extension_Research_Models/Hopfield_models/Associative-Transformer">Associative-Transformer</a></li>
                <li><a href="Extension_Research_Models/Hopfield_models/iMixer">iMixer</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Brain_inspired_models/Neurons">Neurons</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/ZEBRA">ZEBRA</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/Lp-Convolution">Lp-Convolution</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/PhiNets">PhiNets</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/PhiNetv2">PhiNetv2</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/NICE">NICE</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/Brain-inspired-Replay">Brain-inspired-Replay</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/SPM">SPM</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/Pixel-Reg">Pixel-Reg</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/NeuroBridge">NeuroBridge</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/TMNs">TMNs</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/FSPT-FSCIL">FSPT-FSCIL</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/ML-BVAE">ML-BVAE</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/BraVL">BraVL</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/IterativeVAE">IterativeVAE</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/HiCL">HiCL</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/VCFlow">VCFlow</a></li>
                <li><a href="Extension_Research_Models/Brain_inspired_models/NeuroPictor">NeuroPictor</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_editing_models/InstructPix2Pix">InstructPix2Pix</a></li>
                <li><a href="Extension_Research_Models/Image_editing_models/Insert-Anything">Insert-Anything</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Image_similarity_models/VisDiff">VisDiff</a></li>
                <li><a href="Extension_Research_Models/Image_similarity_models/relsim">relsim</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Active_learning_models/APL">APL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/PCB">PCB</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/PCoreSet">PCoreSet</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/CALD">CALD</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/AL">AL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/LabelBench">LabelBench</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/EADA">EADA</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/TQS">TQS</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/SDM">SDM</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/RIPU">RIPU</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/A2LC">A2LC</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ALC">ALC</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/SUGFW">SUGFW</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/NAS">NAS</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/TypiClust">TypiClust</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/SCAN">SCAN</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/UHerding">UHerding</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ISOAL">ISOAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/CSQ">CSQ</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/BADGE">BADGE</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/PT4AL">PT4AL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/PPAL">PPAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/BLAD">BLAD</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/BGADL">BGADL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/AL-consistency">AL-consistency</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ALFA-Mix">ALFA-Mix</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/PAL">PAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/CAMPAL">CAMPAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/LADA">LADA</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/SAAL">SAAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/DiaNA">DiaNA</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/AutoAugment">AutoAugment</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/UWE">UWE</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ActGen">ActGen</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/CAL">CAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/AccuACL">AccuACL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/E2OAL">E2OAL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/BalancedEntropy">BalancedEntropy</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/HALO">HALO</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/DG_AL">DG_AL</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/CEG">CEG</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ELDA">ELDA</a></li>
                <li><a href="Extension_Research_Models/Active_learning_models/ASTE-AL">ASTE-AL</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Upsample_models/UpsampleAnything">UpsampleAnything</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/LoftUp">LoftUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/FeatUp">FeatUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/AnyUp">AnyUp</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/NAF">NAF</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/FeatSharp">FeatSharp</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/LiFT">LiFT</a></li>
                <li><a href="Extension_Research_Models/Upsample_models/UPLiFT">UPLiFT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Autoregressive_models/AutoregressiveSegmentation">AutoregressiveSegmentation</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/CSZL_models/TOMCAT">TOMCAT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Segmentation_models/RD-Net">RD-Net</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/SegEarth-OV-3">SegEarth-OV-3</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/CrowdSAM">CrowdSAM</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/SAM-DAQ">SAM-DAQ</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/SAM3-Adapter">SAM3-Adapter</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/DINO-AugSeg">DINO-AugSeg</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/FSSDINO">FSSDINO</a></li>
                <li><a href="Extension_Research_Models/Segmentation_models/VidEoMT">VidEoMT</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/ICL_models/AWRaCLe">AWRaCLe</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/CL_models/CL_all-in-one">CL_all-in-one</a></li>
                <li><a href="Extension_Research_Models/CL_models/LEAR">LEAR</a></li>
                <li><a href="Extension_Research_Models/CL_models/ARCL-ViT">ARCL-ViT</a></li>
                <li><a href="Extension_Research_Models/CL_models/QGPM">QGPM</a></li>
                <li><a href="Extension_Research_Models/CL_models/GPM">GPM</a></li>
                <li><a href="Extension_Research_Models/CL_models/FIRE">FIRE</a></li>
                <li><a href="Extension_Research_Models/CL_models/ACL">ACL</a></li>
                <li><a href="Extension_Research_Models/CL_models/PILOT">PILOT</a></li>
                <li><a href="Extension_Research_Models/CL_models/Low-rank-CL">Low-rank-CL</a></li>
                <li><a href="Extension_Research_Models/CL_models/InfLoRA">InfLoRA</a></li>
                <li><a href="Extension_Research_Models/CL_models/SD-LoRA">SD-LoRA</a></li>
                <li><a href="Extension_Research_Models/CL_models/CL-LoRA">CL-LoRA</a></li>
                <li><a href="Extension_Research_Models/CL_models/AFEC">AFEC</a></li>
                <li><a href="Extension_Research_Models/CL_models/IDER">IDER</a></li>
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
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Energy_models/EBMs_Jarzynski">EBMs_Jarzynski</a></li>
                <li><a href="Extension_Research_Models/Energy_models/TEA">TEA</a></li>
                <li><a href="Extension_Research_Models/Energy_models/Hyper-SET">Hyper-SET</a></li>
                <li><a href="Extension_Research_Models/Energy_models/EBT">EBT</a></li>
                <li><a href="Extension_Research_Models/Energy_models/EBTSA">EBTSA</a></li>
                <li><a href="Extension_Research_Models/Energy_models/ET">ET</a></li>
                <li><a href="Extension_Research_Models/Energy_models/EBM_CL">EBM_CL</a></li>
                <li><a href="Extension_Research_Models/Energy_models/EBCLR">EBCLR</a></li>
                <li><a href="Extension_Research_Models/Energy_models/JEM">JEM</a></li>
                <li><a href="Extension_Research_Models/Energy_models/GeneralizedEBM">GeneralizedEBM</a></li>
                <li><a href="Extension_Research_Models/Energy_models/Discrete-EBM">Discrete-EBM</a></li>
                <li><a href="Extension_Research_Models/Energy_models/EB_GFN">EB_GFN</a></li>
                <li><a href="Extension_Research_Models/Energy_models/HeatOOD">HeatOOD</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Dataset_distillation_models/Dataset-Distillation">Dataset-Distillation</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Reinforcement_learning_models/RL-AWB">RL-AWB</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Quantum_models/TorchQuantum">TorchQuantum</a></li>
                <li><a href="Extension_Research_Models/Quantum_models/QuantumGS">QuantumGS</a></li>
            </ul>
        </td>
        <td>
            <ul>
                <li><a href="Extension_Research_Models/Test_time_models/ViTTT">ViTTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTT">TTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTT-MAE">TTT-MAE</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTT-Video">TTT-Video</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTT_Denoising">TTT_Denoising</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/NCTTT">NCTTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTT_Matching_VOS">TTT_Matching_VOS</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/ITTT">ITTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/DATTT">DATTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTA">TTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/Learning-Loss-TTA">Learning-Loss-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TTAch">TTAch</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/MTA">MTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/GPS-TTA">GPS-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/CI-TTA">CI-TTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/FSL">FSL</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/FSL-Rectifier">FSL-Rectifier</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/DynTTA">DynTTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/MEMO">MEMO</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/MV-ATTA">MV-ATTA</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/RLCF">RLCF</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/tttLRM">tttLRM</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/TPT">TPT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/O-TPT">O-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/C-TPT">C-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/Vision-TTT">Vision-TTT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/A-TPT">A-TPT</a></li>
                <li><a href="Extension_Research_Models/Test_time_models/IMSE">IMSE</a></li>
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
  howpublished = {https://github.com/SKKUAutoLab/ETSS-06-Anomaly},
  year         = {2023}
}
```

## 5. Contact
If you have any questions, feel free to contact `Chi Tran` 
([ctran743@gmail.com](ctran743@gmail.com) or [tdc2000@skku.edu](tdc2000@skku.edu)).

## 6. Acknowledgement
Our framework is built using multiple open source, thanks for their great contributions.