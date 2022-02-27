import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from torchvision import transforms
import videotransforms

import numpy as np

from configs import Config
# from pytorch_i3d import InceptionI3d
# from models.model_transformer import ISLR
from models.cnn_transformer import ISLR
from models.configs import model_config

# from datasets.nslt_dataset import NSLT as Dataset
from nslt_dataset import NSLT as Dataset

import logging

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('-mode', type=str, help='rgb or flow')
parser.add_argument('-save_model', type=str)
parser.add_argument('-root', type=str)
parser.add_argument('--num_class', type=int)
parser.add_argument('-weights', type=str)
parser.add_argument('-train_split', type=str)
parser.add_argument('-config_file', type=str)
parser.add_argument('-max_epoch', type=int)

args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run(configs,
        mode='rgb',
        root='/ssd/Charades_v1_rgb',
        train_split='charades/charades.json',
        save_model='',
        weights=None):
    
    logging.info(configs)

    # setup dataset
    train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
                                           videotransforms.RandomHorizontalFlip(), ])
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(train_split, 'train', root, mode, train_transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                             pin_memory=True)

    val_dataset = Dataset(train_split, 'test', root, mode, test_transforms)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0,
                                                 pin_memory=True)

    dataloaders = {'train': dataloader, 'test': val_dataloader}
    datasets = {'train': dataset, 'test': val_dataset}

    # setup the model
    # if mode == 'flow':
    #     i3d = InceptionI3d(400, in_channels=2)
    #     # TODO: No Pretrained
    #     # i3d.load_state_dict(torch.load('weights/flow_imagenet.pt'))
    # else:
    #     i3d = InceptionI3d(400, in_channels=3)
    #     # TODO: Not Pretrained
    #     # i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

    num_classes = dataset.num_classes
    model_config["num_class"] = num_classes
    # i3d.replace_logits(num_classes)
    i3d = ISLR(**model_config)

    # TimSformer is too big to run it 
    # i3d = TimeSformer(dim=512, num_frames=32, num_classes=num_classes)

    if weights != "None":
        logging.info('loading weights {}'.format(weights))
        i3d.load_state_dict(torch.load(weights))

    logging.info(i3d)
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = configs.init_lr
    weight_decay = configs.adam_weight_decay
    optimizer = optim.Adam(i3d.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(i3d.parameters(), lr=1e-5, weight_decay=weight_decay)

    num_steps_per_update = configs.update_per_step  # accum gradient
    criterion = nn.CrossEntropyLoss()
    steps = 0
    epoch = 0

    best_val_score = 0
    # train it
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True, threshold=1e-5)
    while epoch < args.max_epoch:  # for epoch in range(num_epochs):
        print('Step {}/{}'.format(steps, configs.max_steps))
        print('-' * 10)

        epoch += 1
        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            collected_vids = []

            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode

            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0

            num_iter = 0
            optimizer.zero_grad()

            confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int)
            # Iterate over data.
            for data in dataloaders[phase]:
                num_iter += 1
                # get the inputs
                if data == -1: # bracewell does not compile opencv with ffmpeg, strange errors occur resulting in no video loaded
                    continue

                # inputs, labels, vid, src = data
                inputs, labels, vid = data
                t = inputs.size(2)
                # inputs = inputs.transpose(1, 2).contiguous()

                # wrap them in Variable
                inputs = inputs.cuda()
                # t = inputs.size(2)
                labels = labels.cuda()

                # per_frame_logits = i3d(inputs, pretrained=False)
                logits, per_frame_logit = i3d(inputs)

                per_frame_logits = per_frame_logit.transpose(1, 2).contiguous()
                # upsample to input size
                # per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                # compute localization loss
                loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                tot_loc_loss += loc_loss.data.item()

                # predictions = torch.max(per_frame_logits, dim=2)[0]
                predictions = logits
                gt = torch.max(labels, dim=2)[0]

                # compute classification loss (with max-pooling along time B x C x T)
                # cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0],
                #                                               torch.max(labels, dim=2)[0])
                
                cls_loss = criterion(logits, torch.argmax(labels[:, :, -1], dim=-1))
                tot_cls_loss += cls_loss.data.item()

                for i in range(per_frame_logits.shape[0]):
                    confusion_matrix[torch.argmax(gt[i]).item(), torch.argmax(predictions[i]).item()] += 1

                loss = (0.5 * loc_loss + 0.5 * cls_loss) / num_steps_per_update
                tot_loss += loss.data.item()
                if num_iter == num_steps_per_update // 2:
                    print(epoch, steps, loss.data.item())
                loss.backward()

                if num_iter == num_steps_per_update and phase == 'train':
                    steps += 1
                    num_iter = 0
                    optimizer.step()
                    optimizer.zero_grad()
                    # lr_sched.step()
                    if steps % 10 == 0:
                        acc = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                        logging.info(
                            'Epoch {} Step {} {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(epoch,
                                                                                                                 steps,
                                                                                                                 phase,
                                                                                                                 tot_loc_loss / (10 * num_steps_per_update),
                                                                                                                 tot_cls_loss / (10 * num_steps_per_update),
                                                                                                                 tot_loss / 10,
                                                                                                                 acc))
                        tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'test':
                val_score = float(np.trace(confusion_matrix)) / np.sum(confusion_matrix)
                if val_score > best_val_score:
                    best_val_score = val_score
                    model_name = save_model + "nslt_" + str(num_classes) + "_ordertrans_" + str(epoch).zfill(
                                   3) + '_%3f.pt' % val_score

                    torch.save(i3d.module.state_dict(), model_name)
                    logging.info(model_name)

                logging.info('VALIDATION: {} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f} Accu :{:.4f}'.format(phase,
                                                                                                              tot_loc_loss / num_iter,
                                                                                                              tot_cls_loss / num_iter,
                                                                                                              (tot_loss * num_steps_per_update) / num_iter,
                                                                                                              val_score
                                                                                                              ))

                scheduler.step(tot_loss * num_steps_per_update / num_iter)


if __name__ == '__main__':
    # WLASL setting
    # mode = 'rgb'
    # root = {'word': '../../data/WLASL2000'}

    # save_model = 'checkpoints/'
    # train_split = 'preprocess/nslt_2000.json'

    # # weights = 'archived/asl2000/FINAL_nslt_2000_iters=5104_top1=32.48_top5=57.31_top10=66.31.pt'
    # weights = None
    # config_file = 'configfiles/asl2000.ini'

    # configs = Config(config_file)
    def init_logging(log_file):
        """Init for logging
        """
        logging.basicConfig(level = logging.INFO,
                            format = '%(asctime)s: %(message)s',
                            datefmt = '%m-%d %H:%M:%S',
                            filename = log_file,
                            filemode = 'w')
        # define a Handler which writes INFO message or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    init_logging(os.path.join('log/train_{:s}_{:s}_remove_posemd_log.txt'.format("0905", args.train_split.split("/")[-1])))

    logging.info('Checkpoints save in {}'.format(args.save_model))
    logging.info(args)
    run(configs=Config(args.config_file), mode=args.mode, root={'word': args.root}, save_model=args.save_model, train_split=args.train_split, weights=args.weights)
