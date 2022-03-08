# cross-entropy
python train.py \
--dataset cifar100 \
--model resnet56 \
--schedule-steps 100 150 \
--epochs 200 \
--loss cross_entropy

# to train cifar100 on other methods, refer to scripts/cifar10.sh