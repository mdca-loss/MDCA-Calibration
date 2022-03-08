# train tiny-imagenet on cross-entropy

python train.py \
--dataset imagenet \
--model resnet50_imagenet \
--train-batch-size 64 \
--test-batch-size 64 \
--epochs 100 \
--schedule-steps 40 60 \
--loss cross_entropy

# to train tiny-imagenet on other methods, refer to scripts/cifar10.sh