import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for calibration')

    # Datasets
    parser.add_argument('-d', '--dataset', default='cifar10', type=str)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=100, type=int,
                        help='seed to use')
    parser.add_argument('--imbalance', default=0.02, type=float,
                        help='Imbalance to use in long tailed CIFAR10/100')
    parser.add_argument('--delta', default=0.25, type=float,
                        help='delta to use in Huber Loss in MDCA')
    # Optimization options
    parser.add_argument('--epochs', default=150, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')

    parser.add_argument('--train-batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch-size', default=100, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--alpha', default=5.0, type=float,
                        metavar='ALPHA', help='alpha to train Label Smoothing with')
    parser.add_argument('--beta', default=10, type=float,
                        metavar='BETA', help='beta to train DCA/MDCA with')
    parser.add_argument('--gamma', default=1, type=float,
                        metavar='GAMMA', help='gamma to train Focal Loss with')

    parser.add_argument('--drop', '--dropout', default=0, type=float,
                        metavar='Dropout', help='Dropout ratio')

    parser.add_argument('--schedule-steps', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, help='LR is multiplied by this on schedule.')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    # Checkpoints
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--loss', default='cross_entropy', type=str, metavar='LNAME')
    parser.add_argument('--model', default='resnet20', type=str, metavar='MNAME')
    parser.add_argument('--optimizer', default='sgd', type=str, metavar='ONAME')

    parser.add_argument('--prefix', default='', type=str, metavar='PRNAME')
    parser.add_argument('--regularizer', default='l2', type=str, metavar='RNAME')
    parser.add_argument('--patience', default=10, type=int)
    
    return parser.parse_args()