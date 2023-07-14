# ExpressiveVC Expressive Voice Conversion

This is a repository based on so-vits-svc singing voice conversion model with focus on expressive speech conversion.

Implementation plans that differs from original so-vits-svc

- [ ] (WIP) Add energy conditioning
  - [x] Raw conditioning (failed, already excluded branch)
  - [ ] Quantized energy with lookup embedding aggregation  
- [ ] Add option to SSL representation (default is ContentVec or Hubert)
- [ ] Release pre-trained model 
- [ ] Change vocoder to MB-ISTFT-VITS for better inference-time
- [ ] Add KNN-VC for better cross-lingual

Experiments plans

- [ ] Synthetic cross-speaker expressive speech dataset for TTS modeling
- [ ] Cross-language Cross-speaker emotional speech transfer


Below you have the steps to run a training with this model:

## Required downloads
+ Download ContentVec model:[checkpoint_best_legacy_500.pt](https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/checkpoint_best_legacy_500.pt)
  + Place under `hubert`.
+ Download pretrained models [G_0.pth](https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth) and [D_0.pth](https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth)
  + Place under `logs/44k`.
  + Pretrained models are required, because from experiments, training from scratch can be rather unpredictable to say the least, and training with a pretrained model can greatly improve training speeds.
  + The pretrained model includes云灏, 即霜, 辉宇·星AI, 派蒙, and 绫地宁宁, covering the common ranges of both male and female voices, and so it can be seen as a rather universal pretrained model.

```shell
wget -P logs/44k/ https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/G_0.pth
wget -P logs/44k/ https://huggingface.co/therealvul/so-vits-svc-4.0-init/resolve/main/D_0.pth
```

## Dataset preparation
All that is required is that the data be put under the `dataset_raw` folder in the structure format provided below.
```shell
dataset_raw
├───speaker0
│   ├───xxx1-xxx1.wav
│   ├───...
│   └───Lxx-0xx8.wav
└───speaker1
    ├───xx2-0xxx2.wav
    ├───...
    └───xxx7-xxx007.wav
```

## Data pre-processing.
1. Resample to 44100hz

```shell
python resample.py
 ```
2. Automatically sort out training set, validation set, test set, and automatically generate configuration files.
```shell
python preprocess_flist_config.py
```
3. Generate hubert and F0 features/
```shell
python preprocess_hubert_f0.py
```
After running the step above, the `dataset` folder will contain all the pre-processed data, you can delete the `dataset_raw` folder after that.

## Training.
```shell
python train.py -c configs/config.json -m 44k
```
Note: The old model will be automatically cleared during training, and only the latest 5 models will be kept. If you want to prevent overfitting, you need to manually back up the model record points, or modify the configuration file keep_ckpts 0 to never clear.

To train a cluster model, train a so-vits-svc 4.0 model first (as above), then execute `python cluster/train_cluster.py`.

## Inference
For instructions on using the GUI see the `eff` [branch](https://github.com/effusiveperiscope/so-vits-svc/tree/eff)
Otherwise use [inference_main.py](inference_main.py)
Command line support has been added for inference

```shell
# Example
python inference_main.py -m "logs/44k/G_30400.pth" -c "configs/config.json" -n "君の知らない物語-src.wav" -t 0 -s "nen"
```

Required fields
+ -m, --model_path: model path
+ -c, --config_path: configuration file path
+ -n, --clean_names: list of wav file names placed in `raw` folder
+ -t, --trans: pitch transpose (semitones)
+ -s, --spk_list: target speaker names

Optional fields
+ -a, --auto_predict_f0:Automatic pitch prediction; do not enable when converting singing or it will be out of tune.
+ -cm, --cluster_model_path:Path of cluster model
+ -cr, --cluster_infer_ratio:Ratio of clustering to use

# Optional fields
### Automatic f0 prediction
The 4.0 model training process will train an f0 predictor. For voice conversion
you can enable automatic pitch prediction. Do not enable this function when
converting singing voices unless you want it to be out of tune.
### Cluster timbre leakage
Clustering is used to make the model trained more like the target timbre at the cost of articulation/intelligibility. The model can linearly control the proportion of non-clustering scheme (more intelligible, 0) vs. clustering scheme (more speaker-like, 1).

## Onnx export
Use [onnx_export.py](onnx_export.py)
+ Create a new folder:`checkpoints` and open it
+ Create a new folder in the `checkpoints` folder and name it after your project such as `aziplayer`
+ Rename your model to `model.pth`，rename the config file to `config.json`，and place it in the project folder (`aziplayer` )
+ In [onnx_export.py](onnx_export.py) change `path = "NyaruTaffy"` to your project name e.g. `path = "aziplayer"`
+ Run [onnx_export.py](onnx_export.py) 
+ After execution is completed，A `model.onnx` file will be generated in your project folder, which is the exported model
### Onnx UI
   + [MoeSS](https://github.com/NaruseMioShirakana/MoeSS)

