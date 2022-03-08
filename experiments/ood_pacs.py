import os
from utils.misc import AverageMeter

import torch
import torch.optim as optim

from utils import Logger, parse_args

from solvers.runners import test

from models import model_dict
from datasets import corrupted_dataloader_dict, dataset_nclasses_dict, dataset_classname_dict, corrupted_dataset_dict

from calibration_library.ece_loss import ECELoss
from calibration_library.cce_loss import CCELossFast

import logging

if __name__ == "__main__":
    
    args = parse_args()
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            logging.StreamHandler()
                        ])

    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")
    assert args.checkpoint, "Please provide a trained model file"
    assert os.path.isfile(args.checkpoint)
    logging.info(f'Resuming from saved checkpoint: {args.checkpoint}')
   
    checkpoint_folder = os.path.dirname(args.checkpoint)
    saved_model_dict = torch.load(args.checkpoint)

    model = model_dict[args.model](num_classes=num_classes, alpha=args.alpha)
    model.load_state_dict(saved_model_dict['state_dict'])
    model.cuda()

    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")

    # set up metrics
    ece_evaluator = ECELoss(n_classes=num_classes)    
    fastcce_evaluator = CCELossFast(n_classes=num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'ood_test.txt')
    logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))

    logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    # read corruptions
    corruption_list = ["art", "cartoon", "sketch"]
    
    top1_avg = AverageMeter()
    top3_avg = AverageMeter()
    sce_avg = AverageMeter()
    ece_avg = AverageMeter()
    test_nll_avg = AverageMeter()

    for c_type in corruption_list:
        _, _, testloader = corrupted_dataloader_dict[args.dataset](args, target_type=c_type)
        test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, ece_evaluator, fastcce_evaluator, criterion)
        method_name = c_type
        logger.append([method_name, test_loss, top1, top3, top5, cce_score, ece_score])

        top1_avg.update(top1)
        top3_avg.update(top3)
        sce_avg.update(cce_score)
        ece_avg.update(ece_score)
        test_nll_avg.update(test_loss)

    logger.append(["avg_domains", test_nll_avg.avg, top1_avg.avg, top3_avg.avg, top3_avg.avg, sce_avg.avg, ece_avg.avg])

