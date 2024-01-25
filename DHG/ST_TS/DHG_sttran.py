from torch.utils.hipify.hipify_python import str2bool

from DHG.dataset import get_train_test_data
from data_process.Hand_Dataset import *
import torch.optim as optim
import time
import argparse
import os


from model.AltFormer.ST_GCN_AltFormer import ST_GCN_AltFormer

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=16)  # 16
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=10000, type=int, metavar='N',
                    help='number of total epochs to run')  # 1000

parser.add_argument('--patiences', default=1000, type=int,
                    help='number of epochs to tolerate no improvement of val_loss')  # 1000


parser.add_argument('--test_subject_id', type=int, default=3,
                    help='id of test subject, for cross-validation')

parser.add_argument('--data_cfg', type=int, default=0,
                    help='0 for 14 class, 1 for 28')
parser.add_argument(
    '--nesterov', type=str2bool, default=False, help='use nesterov or not')

parser.add_argument('--dp_rate', type=float, default=0.2,
                    help='dropout rate')  # 1000



def init_data_loader(test_subject_id, data_cfg):

    train_data, test_data = get_train_test_data(test_subject_id, data_cfg)

    train_dataset = Hand_Dataset(train_data, use_data_aug=True, time_len=150, expand=0)

    test_dataset = Hand_Dataset(test_data, use_data_aug=False, time_len=150, expand=0)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    return train_loader, val_loader

def init_model(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, backbone_in_c=128, num_frame=150, num_joints=22, num_class=class_num,
                             style='ST',
                             graph=address,
                             graph_args=graph_args
                             )
    model = torch.nn.DataParallel(model).cuda()

    return model


def model_foreward(sample_batched,model,criterion):


    data = sample_batched["skeleton"].float()
    label = sample_batched["label"]
    label = label.type(torch.LongTensor)
    label = label.cuda()
    label = torch.autograd.Variable(label, requires_grad=False)


    score = model(data)

    loss = criterion(score,label)

    acc = get_acc(score, label)

    return score,loss, acc



def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs==labels)/float(labels.size)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    print("\nhyperparamter......")
    args = parser.parse_args()
    print(args)

    print("test_subject_id: ", args.test_subject_id)

   #folder for saving trained model...
    # change this path to the fold where you want to save your pre-trained model
    # model_fold = "/gpu2/yc984/hand/model/DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(args.test_subject_id,args.dp_rate, args.learning_rate, args.data_cfg)
    model_fold = "/data/zjt/handgesture/ST/DHS_ID-{}_dp-{}_lr-{}_dc-{}/".format(args.test_subject_id, args.dp_rate,
                                                                              args.learning_rate, args.data_cfg)
    try:
        os.mkdir(model_fold)
    except:
        pass




    train_loader, val_loader = init_data_loader(args.test_subject_id,args.data_cfg)
    # print("train_loader:",train_loader)
    # print("val:",val_loader)

    #.........inital model
    print("\ninit model.............")
    model = init_model(args.data_cfg)

    model_solver = optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=args.nesterov,
        weight_decay=0.0001)

    #........set loss
    criterion = torch.nn.CrossEntropyLoss()


    #
    train_data_num = 2660
    test_data_num = 140
    iter_per_epoch = int(train_data_num / args.batch_size)

    #parameters recording training log
    max_acc = 0
    no_improve_epoch = 0
    n_iter = 0

    #***********training#***********
    for epoch in range(args.epochs):
        print("\ntraining.............")
        model.train()
        start_time = time.time()
        train_acc = 0
        train_loss = 0
        for i, sample_batched in enumerate(train_loader):
            n_iter += 1
            # print("training i:",i)
            # print("ss_batch:",sample_batched)
            if i + 1 > iter_per_epoch:
                continue
            score,loss, acc = model_foreward(sample_batched, model, criterion)

            model.zero_grad()
            loss.backward()
            #clip_grad_norm_(model.parameters(), 0.1)
            model_solver.step()


            train_acc += acc
            train_loss += loss

            #print(i)



        train_acc /= float(i + 1)
        train_loss /= float(i + 1)

        print("*** DHS  Epoch: [%2d] time: %4.4f, "
              "cls_loss: %.4f  train_ACC: %.6f ***"
              % (epoch + 1,  time.time() - start_time,
                 train_loss.data, train_acc))


        start_time = time.time()

        # adjust_learning_rate(model_solver, epoch + 1, args)
        # print(print(model.module.encoder.gcn_network[0].edg_weight))
        #
        # ***********evaluation***********
        with torch.no_grad():
            val_loss = 0
            acc_sum = 0
            model.eval()
            for i, sample_batched in enumerate(val_loader):
                #print("testing i:", i)
                label = sample_batched["label"]
                score, loss, acc = model_foreward(sample_batched, model, criterion)
                val_loss += loss

                if i == 0:
                    score_list = score
                    label_list = label
                else:
                    score_list = torch.cat((score_list, score), 0)
                    label_list = torch.cat((label_list, label), 0)


            val_loss = val_loss / float(i + 1)
            val_cc = get_acc(score_list,label_list)


            print("*** DHS  Epoch: [%2d], "
                  "val_loss: %.6f,"
                  "val_ACC: %.6f ***"
                  % (epoch + 1, val_loss, val_cc))

            #save best model
            if val_cc > max_acc:
                max_acc = val_cc
                no_improve_epoch = 0
                val_cc = round(val_cc, 10)

                torch.save(model.state_dict(),
                           '{}/epoch_{}_acc_{}.pth'.format(model_fold, epoch + 1, val_cc))
                print("performance improve, saved the new model......best acc: {}".format(max_acc))
            else:
                no_improve_epoch += 1
                print("no_improve_epoch: {} best acc {}".format(no_improve_epoch,max_acc))

            if no_improve_epoch > args.patiences:
                print("stop training....")
                break
