# Quantitative Evaluation 📊

We provide instructions on how to evaluate Mobile-VideoGPT models on MVBench, PerceptionTest, NextQA, MLVU, EgoSchema, and ActNet-QA. Please follow the instructions below:

## Download the Mobile-VideoGPT Models
Mobile-VideoGPT models are available on [Mobile-VideoGPT](https://huggingface.co/collections/Abdelrahman-shaker/mobile-videogpt-fast-and-accurate-video-understanding-langu-67dc745074f8dd68d93b6b92). Please follow the instructions below to download,

Save the downloaded models under `Checkpoints` directory.

```bash

mkdir Checkpoints
cd Checkpoints
git lfs install
git clone https://huggingface.co/Amshaker/Mobile-VideoGPT-0.5B
git clone https://huggingface.co/Amshaker/Mobile-VideoGPT-1.5B
```

## Lmms_eval based evaluation

First, clone the LMMS_eval repository as follows:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
```

Second, you need to integrate the MobileVideoGPT model into the LMMS_eval framework into lmms_eval as follows:

- Copy [eval/mobile_videogpt.py]() to lmms-eval/lmms_eval/models
- Update the available models of lmms_eval to include MobileVideoGPT to lmms_eval/models/__init__.py as follows:
  
```bash
 "mobile_videogpt": "MobileVideoGPT",
```

Third, copy the evaluation scripts to lmms-eval repository to run the evaluation for all benchmarks as follows:

### Run Evaluation for Mobile-VideoGPT Models (0.5B and 1.5B)
We provide [Mobile-VideoGPT-evaluation.sh](Mobile-VideoGPT-evaluation.sh) script to run inference on multiple GPUs for Mobile-VideoGPT-0.5B or Mobile-VideoGPT-1.5B:

```bash
bash Mobile-VideoGPT-evaluation.sh Checkpoints/Mobile-VideoGPT-0.5B
bash Mobile-VideoGPT-evaluation.sh Checkpoints/Mobile-VideoGPT-1.5B
```

Where `Checkpoints/Mobile-VideoGPT-0.5B` is the path of Mobile-VideoGPT-0.5B model and `Checkpoints/Mobile-VideoGPT-1.5B` is the path of Mobile-VideoGPT-1.5B model.
