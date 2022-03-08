import os

import torch
import torch.optim as optim

from utils import mkdir_p, parse_args
from utils import get_lr, save_checkpoint, create_save_path

from solvers.runners import train, test
from solvers.loss import loss_dict

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from time import localtime, strftime

import logging

if __name__ == "__main__":
    
    args = parse_args()

    current_time = strftime("%d-%b", localtime())
    # prepare save path
    model_save_pth = f"{args.checkpoint}/{args.dataset}_IF={args.imbalance}/{current_time}{create_save_path(args)}"
    checkpoint_dir_name = model_save_pth

    if not os.path.isdir(model_save_pth):
        mkdir_p(model_save_pth)

    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
                            logging.StreamHandler()
                        ])
    logging.info(f"Setting up logging folder : {model_save_pth}")

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    model = model_dict[args.model](num_classes=num_classes)
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)

    logging.info(f"Using imbalanced CIFAR10 with imbalance: {args.imbalance}")
    logging.info(f"Setting up optimizer : {args.optimizer}")

    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), 
                              lr=args.lr, 
                              momentum=args.momentum, 
                              weight_decay=args.weight_decay)

    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)
    
    criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta, loss=args.loss, delta=args.delta)
    test_criterion = loss_dict["cross_entropy"]()
    
    logging.info(f"Step sizes : {args.schedule_steps} | lr-decay-factor : {args.lr_decay_factor}")
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_steps, gamma=args.lr_decay_factor)

    start_epoch = args.start_epoch
    
    best_acc = 0.
    best_acc_stats = {"top1" : 0.0}

    for epoch in range(start_epoch, args.epochs):

        logging.info('Epoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, get_lr(optimizer)))
        
        train_loss, top1_train = train(trainloader, model, optimizer, criterion)
        test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, test_criterion)

        scheduler.step()

        logging.info("End of epoch {} stats: train_loss : {:.4f} | val_loss : {:.4f} | top1_train : {:.4f} | top1 : {:.4f} | SCE : {:.5f} | ECE : {:.5f}".format(
            epoch+1,
            train_loss,
            test_loss,
            top1_train,
            top1,
            cce_score,
            ece_score
        ))

        # save best accuracy model
        is_best = top1 > best_acc
        best_acc = max(best_acc, top1)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict(),
                'dataset' : args.dataset,
                'model' : args.model
            }, is_best, checkpoint=model_save_pth)
        
        # Update best stats
        if is_best:
            best_acc_stats = {
                "top1" : top1,
                "top3" : top3,
                "top5" : top5,
                "SCE" : cce_score,
                "ECE" : ece_score
            }

    logging.info("training completed...")
    logging.info("The stats for best trained model on test set are as below:")
    logging.info(best_acc_stats)