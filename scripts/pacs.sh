# train on photo domain
python train.py \
--dataset pacs \
--model resnet_pacs \
--train-batch-size 256 \
--test-batch-size 256 \
--epochs 30 \
--schedule-steps 20 \
--lr 0.01 \
--loss cross_entropy

# test on other domains
python experiments/ood_pacs.py \
--dataset pacs \
--model resnet_pacs \
--checkpoint checkpoint/pacs/18-Aug_resnet_pacs_cross_entropy/model_best.pth

# to train pacs on other methods, refer to scripts/cifar10.sh