# Train cifar10
# Just replace resnet20 with other model names such as resnet32, resnet56, resnet110 to train on them
# you can also tweak hyper-parameters, look at utils/argparser.py for more parameters.

# Normal training on CIFAR10

# cross entropy
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss cross_entropy

# focal loss
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss focal_loss --gamma 3.0

# label smoothing
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS --alpha 0.1

# MMCE
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss MMCE --beta 4.0

# DCA
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+DCA --beta 1.0

# FLSD (gamma=3)
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FLSD --gamma 3.0

# brier score
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss brier_loss

# NLL+MDCA
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss NLL+MDCA --beta 1.0

# LS+MDCA
# alpha is for label-smoothing
# beta is weight assigned to MDCA
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss LS+MDCA --alpha 0.1 --beta 1.0

# FL+MDCA
# gamma is for focal-loss
# beta is weight assigned to MDCA
python train.py \
--dataset cifar10 \
--model resnet56 \
--schedule-steps 80 120 \
--epochs 160 \
--loss FL+MDCA --gamma 1.0 --beta 1.0

#################################################

# Post Hoc calibration
python posthoc_calibrate.py \
--dataset cifar10 \
--model resnet56 \
--lr 0.001 \
--patience 5 \
--checkpoint checkpoint/cifar10/08-Aug_resnet56_cross_entropy/model_best.pth