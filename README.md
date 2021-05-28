# Deep Learning - HW3: LeNet

Author: Jay Liao (re6094028@gs.ncku.edu.tw)

This is assignment 3 of Deep Learning, a course at Institute of Data Science, National Cheng Kung University. This project aims to construct LeNet-related models to perform image classification.

## Data

- Images: please go to https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr to download raw image files and put them under the folder `./images/`. There are 64,225 files with 50 subfolders.

- File name lists of images: `./data/train.txt`, `./data/val.txt`, and `./data/test.txt`.

- Preprocessed data of the validation and the testing sets have been produced and put under `./data`.

## Code

- `main.py`: the main program for training LeNet-5 with NumPy/CuPy

- `main_NonCNN.py`: the main program for training baselines with NumPy

- `main_torch.py`: the main program for training LeNet-5 with Pytorch

- Source codes for training LeNet-5:

    -  `./source/args.py`: define the arguments parser

    -  `./source/utils.py`: little tools

    -  `./source/layers.py`: layers for NN model construction, e.g., `ReLU()`, `Sigmoid()`
    
    -  `./source/optimizers.py`: construct the models, e.g., `SGD`, `Adam`.

    -  `./source/models.py`: construct the models
    
    -  `./source/trainer.py`: class for training, predicting, and evaluating the models

-  `requirements.txt`: required packages

- Source codes for training baselines:

    -  `./NonCNN/args_.py`: define the arguments parser

    -  `./NonCNN/utils.py`: little tools

    -  `./NonCNN/feature_extraction.py`: functions for feature extraction

    -  `./NonCNN/layers.py`: layers for NN model construction, e.g., `ReLU()`, `Sigmoid()`
    
    -  `./NonCNN/optimizers.py`: construct the models, e.g., `SGD`, `Adam`.

    -  `./NonCNN/models.py`: construct the models
    
    -  `./NonCNN/trainers.py`: class for training, predicting, and evaluating the models

- Source codes of pytorch version for training LeNet-5: `args.py`, `trainer.py`, `model.py`, and `utils.py` under `./lenet_torch/`.

-  `requirements.txt`: required packages

## Folders

- `./images/` should contain raw image files (please go [here](https://drive.google.com/open?id=1kwYYWL67O0Dcbx3dvZIfbGg9NiHdyisr) to download and put them with subfolders here).

- `./data/` contains .txt files of image lists.

- `./output/` will contain trained models, model performances, and experiments results after running. 

## Requirements

```
numpy==1.16.3
cupy-cuda102==8.6.0      # change this if your cuda version is not 10.2
pandas==0.24.2
tqdm==4.50.0
opencv-python==3.4.2.16
matplotlib==3.1.3
```

## Usage

1. Clone this repo.

```
git clone https://github.com/jayenliao/DL-LeNet-numpy.git
```

2. Set up the required packages.

```
cd DL-LeNet-numpy
pip3 install requirements.txt
```

3. Run the experiments.

```
python3 main.py
python3 main_NonCNN.py
```

It may take much time to run the whole `main.py`. The arguments parser can be used to run several experiments only, such as:

```
python3 main.py --sizes_filter 3 3
python3 main_NonCNN.py --models 'TwoLayerPerceptron' --epochs 100
```

## Reference

1. Liao, J. C. (2021). Deep Learning - Image Classification. GitHub: https://github.com/jayenliao/DL-image-classification.

2. Liao, J. C. (2021). Deep Learning - Computational Graph. GitHub: https://github.com/jayenliao/DL-computational-graph.

3. Lowe, D. G. (1999, September). Object recognition from local scale-invariant features. In Proceedings of the seventh IEEE international conference on computer vision (Vol. 2, pp. 1150-1157). Ieee.

4. Bay, H., Ess, A., Tuytelaars, T., & Van Gool, L. (2008). Speeded-up robust features (SURF). Computer vision and image understanding, 110(3), 346-359.

5. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

6. 斎藤康毅（吳嘉芳譯）（2017）。Deep Learning: 用Python進行深度學習的基礎理論實作。碁峰資訊股份有限公司。ISBN: 9789864764846。GitHub: https://github.com/oreilly-japan/deep-learning-from-scratch。

7. Watt, J., Borhani, R., & Katsaggelos, A. K. (2019). Machine learning refined. ISBN: 9781107123526. GitHub: https://github.com/jermwatt/machine_learning_refined.
