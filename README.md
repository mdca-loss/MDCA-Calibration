# MDCA-Calibration
## Abstract
Deep Neural Networks (DNNs) make overconfident mistakes which can prove to be probematic in deployment in safety critical applications. Calibration is aimed to enhance trust in DNNs. The goal of our proposed Multi-Class Difference in Confidence and Accuracy (MDCA) loss is to align the probability estimates in accordance with accuracy thereby enhancing the trust in DNN decisions. MDCA can be used in case of image classification, image segmentation, and natural language classification tasks.


TODO
@neelabh insert teaser figure

## Training scripts:

Refer to the `scripts` folder to train for every model and dataset. Overall the command to train looks like below where each argument can be changed accordingly on how to train. Also refer to `dataset/__init__.py` and `models/__init__.py` for correct arguments to train with. Argument parser can be found in `utils/argparser.py`.

Train with cross-entropy:
```
python train.py --dataset cifar10 --model resnet56 --schedule-steps 80 120 --epochs 160 --loss cross_entropy 
```

Train with FL+MDCA: Also mention the gamma (for Focal Loss) and beta (Weight assigned to MDCA) to train FL+MDCA with
```
python train.py --dataset cifar10 --model resnet56 --schedule-steps 80 120 --epochs 160 --loss FL+MDCA --gamma 1.0 --beta 1.0 
```

Train with NLL+MDCA:
```
python train.py --dataset cifar10 --model resnet56 --schedule-steps 80 120 --epochs 160 --loss NLL+MDCA --beta 1.0
```

## Post Hoc Calibration:

To do post-hoc calibration, we can use the following command.

`lr` and `patience` value is used for Dirichlet calibration. To change range of grid-search in dirichlet calibration, refer to `posthoc_calibrate.py`.
```
python posthoc_calibrate.py --dataset cifar10 --model resnet56 --lr 0.001 --patience 5 --checkpoint path/to/your/trained/model
```

## Other Experiments (Dataset Drift, Dataset Imbalance):

`experiments` folder contains our experiments on PACS, Rotated MNIST and Imbalanced CIFAR10. Please refer to the scripts provided to run them.

## Pre-Trained models

We provide trained MDCA ResNet-56 models on CIFAR10/100 and SVHN here. Feel free to use them in your experiments.

TODO
@neelabhm @jatin upload trained models


## References:

[1] <a href="https://github.com/bearpaw/pytorch-classification">bearpaw/pytorch-classification</a>
[2] <a href="https://github.com/torrvision/focal_calibration">torrvision/focal_calibration</a>
[3] <a href="https://github.com/Jonathan-Pearce/calibration_library">Jonathan-Pearce/calibration_library</a>
