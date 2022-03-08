# Train SVHN
# Resnet20
# cross entropy
python train.py \
--dataset svhn \
--model densenet121 \
--schedule-steps 50 70 \
--epochs 100 \
--loss cross_entropy

# to train svhn on other methods, refer to scripts/cifar10.sh