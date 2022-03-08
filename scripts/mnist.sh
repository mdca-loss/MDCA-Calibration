# train on normal mnist
python train.py \
--dataset mnist \
--model resnet_mnist \
--epochs 30 \
--schedule-steps 20 \
--loss cross_entropy

# to train mnist on other methods, refer to scripts/cifar10.sh
# in order to test on rotated MNIST, refer to experiments/rotated_mnist.py