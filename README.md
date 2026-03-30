<h1 align="center"><b><img src="./assets/unils_logo.png" width="420"/></b></h1>
<h1 align="center"><b>UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking</b></h1>
<h3 align="center">
    <a href='https://arxiv.org/abs/2512.09327'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp;
    <a href='https://xg-chu.site/project_unils/'><img src='https://img.shields.io/badge/Project-Page-blue'></a> &nbsp;
    <a href='https://huggingface.co/xg-chu/UniLS'><img src='https://img.shields.io/badge/HuggingFace-Weights-yellow'></a> &nbsp;
    <a href='https://huggingface.co/datasets/xg-chu/UniLSTalkDataset'><img src='https://img.shields.io/badge/HuggingFace-Dataset-yellow'></a> &nbsp;
</h3>

<h5 align="center">
    <a href="https://xg-chu.site">Xuangeng Chu</a><sup>*1</sup>&emsp;
    <a href="https://ruicongliu.github.io">Ruicong Liu</a><sup>*1&dagger;</sup>&emsp;
    <a href="https://hyf015.github.io">Yifei Huang</a><sup>1</sup>&emsp;
    <a href="https://scholar.google.com/citations?user=5mbpi0kAAAAJ&hl=zh-TW">Yun Liu</a><sup>2</sup>&emsp;
    <a href="https://puckikk1202.github.io">Yichen Peng</a><sup>3</sup>&emsp;
    <a href="http://www.bozheng-lab.com">Bo Zheng</a><sup>2</sup>
    <br>
    <sup>1</sup>Shanda AI Research Tokyo, The University of Tokyo,
    <sup>2</sup>Shanda AI Research Tokyo,
    <sup>3</sup>Institute of Science Tokyo
    <br>
    <sup>*</sup>Equal contribution,
    <sup>&dagger;</sup>Corresponding author
</h5>

<h3 align="center">
🤩 CVPR 2026 🤩
</h3>

<div align="center">
    <b>
        UniLS generates diverse and natural listening and speaking motions from audio.
    </b>
</div>

## Installation
### Clone the project
```
git clone --recurse-submodules git@github.com:xg-chu/UniLS.git
cd UniLS
```

### Build environment
```
conda env create -f environment.yml
conda activate unils
```
Or install manually:
```
pip install torch torchvision torchaudio
pip install accelerate transformers peft einops omegaconf lmdb tqdm scipy wandb
```

### Pretrained Models
Download the pretrained models from [HuggingFace](https://huggingface.co/xg-chu/UniLS).

### Data
Download the dataset from [UniLS-Talk Dataset](https://huggingface.co/datasets/xg-chu/UniLSTalkDataset).

## Training

UniLS follows a three-stage training pipeline:

**Stage 1: Motion Codec (VAE)**
```
python train.py -c unils_codec
```

**Stage 2: Audio-Free Autoregressive Generator**

Modify `VAE_PATH` path in the config file to point to the Stage 1 checkpoint, then run:
```
python train.py -c unils_freegen
```

**Stage 3: Audio-Conditioned LoRA Fine-tuning**

Modify `PRETRAIN_PATH` path in the config file to point to the Stage 2 checkpoint, then run:
```
python train.py -c unils_loragen
```

## Evaluation
Run evaluation with multi-GPU support via Accelerate:
```
accelerate launch eval.py -r /path/to/checkpoint --tau 1.0 --cfg 1.5
```
You can also pass an external dataset config to override the checkpoint's dataset:
```
accelerate launch eval.py -r /path/to/checkpoint --dataset configs/dataset.yaml
```

## Inference

### From Dataset
Generate visualizations from the dataset:
```
python infer_dataset.py -r /path/to/checkpoint --clip_length 20 --tau 1.0 --cfg 1.5 --num_samples 32
```
- `--resume_path, -r`: Path to the trained model checkpoint.
- `--dataset`: Path to a dataset YAML config (optional, uses checkpoint config by default).
- `--clip_length`: Duration of the generated clip in seconds (default: 20).
- `--tau`: Temperature for sampling (default: 1.0).
- `--cfg`: Classifier-free guidance scale (default: 1.5).
- `--num_samples, -n`: Number of samples to generate (default: 32).
- `--dump_dir, -d`: Output directory (default: `./render_results`).

### From Audio Files
Generate visualizations directly from audio files, supporting one or two speakers:
```
# Single speaker
python infer_audio.py -r /path/to/checkpoint -a speaker0.wav

# Two speakers (dyadic conversation)
python infer_audio.py -r /path/to/checkpoint -a speaker0.wav --audio2 speaker1.wav
```
- `--resume_path, -r`: Path to the trained model checkpoint.
- `--audio, -a`: Path to speaker 0 audio file.
- `--audio2`: Path to speaker 1 audio file (optional; if omitted, only speaker 0 motion is generated).
- `--tau`: Temperature for sampling (default: 1.0).
- `--cfg`: Classifier-free guidance scale (default: 1.5).
- `--dump_dir, -d`: Output directory (default: `./render_results`).


## Acknowledgements

Some part of our work is built based on FLAME. We also thank the following projects:
- **FLAME**: https://flame.is.tue.mpg.de
- **EMICA**: https://github.com/radekd91/inferno

## Citation
If you find our work useful in your research, please consider citing:
<!-- @inproceedings{chu2025unils,
    title     = {UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking},
    author    = {Chu, Xuangeng and Liu, Ruicong and Huang, Yifei and Liu, Yun and Peng, Yichen and Zheng, Bo},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2026}
} -->
```bibtex
@misc{chu2025unils,
      title={UniLS: End-to-End Audio-Driven Avatars for Unified Listening and Speaking}, 
      author={Xuangeng Chu and Ruicong Liu and Yifei Huang and Yun Liu and Yichen Peng and Bo Zheng},
      year={2025},
      eprint={2512.09327},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.09327}, 
}
```
