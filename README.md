# BGAI

Broadcast-Gated Attention with Identity Adaptive Integration for Efficient Image Super-Resolution

Qian Wang, Yanyu Mao, Ruilong Guo, Mengyang Wang, Jing Wei, Han Pan

## 💻Environment

- [PyTorch >= 1.9](https://pytorch.org/)
- [Python 3.7](https://www.python.org/downloads/)
- [Numpy](https://numpy.org/)
- [BasicSR >= 1.3.4.9](https://github.com/XPixelGroup/BasicSR)

## 🔧Installation

```python
pip install -r requirements.txt
```

## 📜Data Preparation

The trainset uses the DIV2K (800). In order to effectively improve the training speed, images are cropped to 480 * 480 images by running script extract_subimages.py, and the dataloader will further randomly crop the images to the GT_size required for training. GT_size defaults to 128/192/256 (×2/×3/×4). 

```python
python extract_subimages.py
```

The input and output paths of cropped pictures can be modify in this script. Default location: ./datasets/DIV2K.

## 📜Implementation details

Patches of 64 × 64 pixels are randomly cropped from LR images as input. The model is optimized by minimizing the L1 loss through the Adam optimizer with β1 = 0.9, β2 = 0.999. The initial learning rate is set to be 5×10−4 with a multistep scheduler in 500k and is reduced by half at the (250k,400k,450k,475k)-th iterations.

Hyperparameter  | Configuration  
---- | -----
BAG    | 4
BRAB(Each BAG)  | 2
Feature Channel Dimension | 52
Minibatch Size(train) | 64

## 🚀Train

▶️ You can change the training strategy by modifying the configuration file. The default configuration files are included in ./options/train/BGAI. Take one GPU as the example.

```python
### Train ###
### BGAI ###
python train.py -opt ./options/train/BGAI/train_BGAI_x2.yml --auto_resume  # ×2
python train.py -opt ./options/train/BGAI/train_BGAI_x3.yml --auto_resume  # ×3
python train.py -opt ./options/train/BGAI/train_BGAI_x4.yml --auto_resume  # ×4
```

For more training commands, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)

## 🚀Test

▶️ You can modify the configuration file about the test, which is located in ./options/test/BGAI. At the same time, you can change the benchmark datasets and modify the path of the pre-train model. The pre-training weights for our model are saved in the Premodel folder.

```python
### Test ###
### BGAI for Lightweight Image Super-Resolution ###
python basicsr/test.py -opt ./options/test/BGAI/test_BGAI_x2.yml  # ×2
python basicsr/test.py -opt ./options/test/BGAI/test_BGAI_x3.yml  # ×3
python basicsr/test.py -opt ./options/test/BGAI/test_BGAI_x4.yml  # ×4

### BGAI for Large Image Super-Resolution ###
### Flicker2K  Test2K  Test4K  Test8K ###
python basicsr/test.py -opt ./options/test/BGAI/test_BGAI_large.yml  # large image
```

## 🚩Results

The inference results on benchmark datasets will be available at [Google Drive](https://drive.google.com/file/d).

## :mailbox:Contact

If you have any questions, please feel free to contact us wqabby@xupt.edu.cn and [bolttt@stu.xupt.edu.cn](mailto:bolttt@stu.xupt.edu.cn).
