# MDCA Calibration: ``A Stitch in Time Saves Nine:A Train-Time Regularizing Loss for Improved Neural Network Calibration"

This is the official PyTorch implementation for the paper: "A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration". Paper is published at CVPR'22 as ORAL Presentation.
## Abstract
Despite the wide applicability of Deep Neural Networks(DNNs) as a part of decision pipelines in safety critical applications, modern DNNs are found to be poorly calibrated i.e., the DNNs make overconfident mistakes which can prove to be probematic in deployment in safety critical applications. One of the ways to enhance trust in DNNs is via calibration. Our goal is to mitigate miscalibration by proposing a train-time regulariser termed "Multi-Class Difference in Confidence and Accuracy (MDCA)'' loss that aligns the average probability estimates in accordance with average accuracy thereby enhancing the trust in DNN decisions. We demonstrate use of MDCA loss in case of image classification, image segmentation, and natural language classification tasks. 

Trivia: The title " A stitch in time saves Nine" suggests we output well-calibrated models via a joint optimisation of calibration loss alongside the any classification loss thereby outputting well-calibrated models in one go rather than in a post-hoc fashion.

![Teaser](content/teaser.png)

Above image shows comparison of classwise reliability diagrams of Cross-Entropy vs. our proposed method (FL+MDCA). Commonly used metrics for measuring calibration are: Expected Calibration Error (ECE) and Static Calibration Error (SCE). ECE is a weighted average of all gaps on a winning class  while SCE is a class-wise extension for every class.


## Requirements

* Python 3.8
* PyTorch 1.8

Directly install using pip

```
 pip install -r requirements.txt
```
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

## Citation

If you find our work useful in your research, please cite the following:
```bibtex
@InProceedings{Hebbalaguppe_2022_CVPR,
    author    = {Hebbalaguppe, Ramya and Prakash, Jatin and Madan, Neelabh and Arora, Chetan},
    title     = {A Stitch in Time Saves Nine: A Train-Time Regularizing Loss for Improved Neural Network Calibration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {16081-16090}
}
```

## Contact
For questions about our paper or code, please contact any of the authors ([@neelabh17](https://github.com/neelabh17), [@bicycleman15](https://github.com/bicycleman15), [@rhebbalaguppe](https://github.com/rhebbalaguppe) ) or raise an issue on GitHub.

## References:
The code is adapted from the following repositories:

[1] <a href="https://github.com/bearpaw/pytorch-classification">bearpaw/pytorch-classification</a>
[2] <a href="https://github.com/torrvision/focal_calibration">torrvision/focal_calibration</a>
[3] <a href="https://github.com/Jonathan-Pearce/calibration_library">Jonathan-Pearce/calibration_library</a>
