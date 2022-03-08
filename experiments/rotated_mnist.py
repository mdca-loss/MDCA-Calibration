import os
from utils.misc import AverageMeter

import torch
from utils import Logger, parse_args
from solvers.runners import test

from models import model_dict
from datasets import corrupted_dataloader_dict, dataset_nclasses_dict, dataset_classname_dict, corrupted_dataset_dict
from datasets.mnist import get_rotated_set

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
    # set up dataset
    logging.info(f"Using dataset : {args.dataset}")
    num_classes = dataset_nclasses_dict[args.dataset]
    classes_name_list = dataset_classname_dict[args.dataset]
    
    # prepare model
    logging.info(f"Using model : {args.model}")

    model_path_list = open("tmux_runs/model_paths.txt", "r").readlines()

    # set up metrics
    ece_evaluator = ECELoss(n_classes=num_classes)    
    fastcce_evaluator = CCELossFast(n_classes=num_classes)
    
    criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    path_to_log = "results/ood_test_{}_{}.txt".format(args.dataset, args.model)
    global_logger = Logger(path_to_log, resume=os.path.exists(path_to_log))
    global_logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    for path in model_path_list:
        path = path.rstrip("\n")

        assert path, "Please provide a trained model file"
        try:
            assert os.path.isfile(path)
        except:
            print(f"{path} does not exist.")
            continue
        logging.info(f'Resuming from saved checkpoint: {path}')
    
        checkpoint_folder = os.path.dirname(path)
        saved_model_dict = torch.load(path)

        metric_log_path = os.path.join(checkpoint_folder, "ood_test.txt")
        logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))
        logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

        model = model_dict[args.model](num_classes=num_classes)
        model.load_state_dict(saved_model_dict['state_dict'])
        model.cuda()

        # read corruptions
        corruption_list = [0, 15, 30, 45, 60, 75]

        top1_avg = AverageMeter()
        top3_avg = AverageMeter()
        sce_avg = AverageMeter()
        ece_avg = AverageMeter()
        test_nll_avg = AverageMeter()

        for angle in corruption_list:
            testloader = get_rotated_set(args, angle)
            test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, ece_evaluator, fastcce_evaluator, criterion)
            method_name = f"angle={angle}"
            logger.append([method_name, test_loss, top1, top3, top5, cce_score, ece_score])
            global_logger.append([f"{path}_angle={angle}", test_loss, top1, top3, top5, cce_score, ece_score])

            top1_avg.update(top1)
            top3_avg.update(top3)
            sce_avg.update(cce_score)
            ece_avg.update(ece_score)
            test_nll_avg.update(test_loss)

        logger.append([f"avg_angles", test_nll_avg.avg, top1_avg.avg, top3_avg.avg, top3_avg.avg, sce_avg.avg, ece_avg.avg])
        logger.close()
        global_logger.append([f"{path}_avg_angles", test_nll_avg.avg, top1_avg.avg, top3_avg.avg, top3_avg.avg, sce_avg.avg, ece_avg.avg])

