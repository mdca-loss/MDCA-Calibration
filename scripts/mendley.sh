# Resnet50
# train 
python train.py \
--epochs 20 \
--loss FLSD \
--gamma 3.0 \
--optimizer adam \
--lr 0.0001 \
--train-batch-size 8 \
--test-batch-size 8 \
--schedule-steps 10 \
--model resnet50_mendley \
--dataset mendley

# to train mendeley V2 on other methods, refer to scripts/cifar10.sh