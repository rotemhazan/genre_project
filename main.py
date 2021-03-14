import argparse
import torch
import numpy as np
import torch.optim as optim
import os 

from data_loader import MusicDastset, BooksDataset
from model import LeNet, VGG 
from train import train,test 

# Training settings
parser = argparse.ArgumentParser(description='ConvNets for Speech Commands Recognition')
parser.add_argument('--dataset', default='GTZAN', help='choose dataset (GTZAN or LibriSpeech)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='LeNet', help='network architecture: LeNet, VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum, for SGD only')###
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234, metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='num of batches to wait until logging train status')
parser.add_argument('--patience', type=int, default=15, metavar='N', help='how many epochs of no loss improvement should we wait before stop training')

args, unknown = parser.parse_known_args()
args.cuda = args.cuda and torch.cuda.is_available()
print("cuda: {}".format(args.cuda))
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#choose and load dataset
if args.dataset == "GTZAN":
    GTZAN_path = 'music/GTZAN/genres_original.16kHz'
    train_dataset = MusicDastset(root=GTZAN_path + "/train", train = True)
    valid_dataset = MusicDastset(root=GTZAN_path + "/val", train = False)
    test_dataset = MusicDastset(root=GTZAN_path + "/test", train = False)
else:
    train_dataset = BooksDataset("speech2genre", url="train-clean-100", folder_in_archive="", download=False)
    valid_dataset = BooksDataset("speech2genre", url="dev-clean", folder_in_archive="", download=False)
    test_dataset = BooksDataset("speech2genre", url="test-clean", folder_in_archive="", download=False)



num_classes = train_dataset.num_classes
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=0, pin_memory=args.cuda, sampler=None)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=0, pin_memory=args.cuda, sampler=None)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=0, pin_memory=args.cuda, sampler=None)

# build model
if args.arc == 'LeNet':
    model = LeNet(num_classes=num_classes)
elif args.arc.startswith('VGG'):
    model = VGG(args.arc, num_classes=num_classes)
else:
    model = LeNet(num_classes=num_classes)

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.005)

best_valid_loss = np.inf
iteration = 0
epoch = 1

# trainint with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    print('Epoch {0}'.format(str(epoch)))
    valid_loss, acc = test(valid_loader, model, args.cuda, is_valid=True)
    acc_save = 1-acc
    if acc_save > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = acc_save
        state = {
            'net': model.module.state_dict() if args.cuda else model.state_dict(),
            'acc': valid_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}.t7'.format(args.arc))
    epoch += 1

# test model
checkpoint = torch.load('./checkpoint/{}.t7'.format(args.arc),map_location = lambda storage, loc: storage)

if args.arc == 'LeNet':
    test_model = LeNet(num_classes=num_classes)
elif args.arc.startswith('VGG'):
    test_model = VGG(args.arc, num_classes=num_classes)
else:
    test_model = LeNet(num_classes=num_classes)

test_model.load_state_dict(checkpoint["net"])

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    test_model = torch.nn.DataParallel(test_model).cuda()

test(valid_loader, test_model, args.cuda)
test(test_loader, test_model, args.cuda)



