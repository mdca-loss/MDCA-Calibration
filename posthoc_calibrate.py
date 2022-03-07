import os

import torch
from utils import Logger, parse_args

from solvers.runners import test

from models import model_dict
from datasets import dataloader_dict, dataset_nclasses_dict, dataset_classname_dict

from calibration_library.calibrators import TemperatureScaling, DirichletScaling

import logging

if __name__ == "__main__":
    
    args = parse_args()
    logging.basicConfig(level=logging.INFO, 
                        format="%(levelname)s:  %(message)s",
                        handlers=[
                            # logging.FileHandler(filename=os.path.join(model_save_pth, "train.log")),
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
    trainloader, valloader, testloader = dataloader_dict[args.dataset](args)
    
    # criterion = loss_dict[args.loss](gamma=args.gamma, alpha=args.alpha, beta=args.beta, loss=args.loss)
    criterion = torch.nn.CrossEntropyLoss()

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'temperature.txt')
    logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))

    logger.set_names(['temprature', 'SCE', 'ECE'])

    test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, model, criterion)
    logger.append(["1.0", cce_score, ece_score])

    # Set up temperature scaling
    temperature_model = TemperatureScaling(base_model=model)
    temperature_model.cuda()

    logging.info("Running temp scaling:")
    temperature_model.calibrate(valloader)
    
    test_loss, top1, top3, top5, cce_score, ece_score = test(testloader, temperature_model, criterion)
    logger.append(["{:.2f}".format(temperature_model.T), cce_score, ece_score])
    logger.close()


    # Set up dirichlet scaling
    logging.info("Running dirichlet scaling:")
    lambdas = [0, 0.01, 0.1, 1, 10, 0.005, 0.05, 0.5, 5, 0.0025, 0.025, 0.25, 2.5]
    mus = [0, 0.01, 0.1, 1, 10]

    # set up loggers
    metric_log_path = os.path.join(checkpoint_folder, 'dirichlet.txt')
    logger = Logger(metric_log_path, resume=os.path.exists(metric_log_path))
    logger.set_names(['method', 'test_nll', 'top1', 'top3', 'top5', 'SCE', 'ECE'])

    min_stats = {}
    min_error = float('inf')

    for l in lambdas:
        for m in mus:
            # Set up dirichlet model
            dir_model = DirichletScaling(base_model=model, num_classes=num_classes, optim=args.optimizer, Lambda=l, Mu=m)
            dir_model.cuda()

            # calibrate
            dir_model.calibrate(valloader, lr=args.lr, epochs=args.epochs, patience=args.patience)
            val_nll, _, _, _, _, _ = test(valloader, dir_model, criterion)
            test_loss, top1, top3, top5, sce_score, ece_score = test(testloader, dir_model, criterion)

            if val_nll < min_error:
                min_error = val_nll
                min_stats = {
                    "test_loss" : test_loss,
                    "top1" : top1,
                    "top3" : top3,
                    "top5" : top5,
                    "ece_score" : ece_score,
                    "sce_score" : sce_score,
                    "pair" : (l, m)
                }
            
            logger.append(["Dir=({:.2f},{:.2f})".format(l, m), test_loss, top1, top3, top5, sce_score, ece_score])
    
    logger.append(["Best_Dir={}".format(min_stats["pair"]), 
                                            min_stats["test_loss"], 
                                            min_stats["top1"], 
                                            min_stats["top3"], 
                                            min_stats["top5"], 
                                            min_stats["sce_score"], 
                                            min_stats["ece_score"]])

