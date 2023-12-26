import time

import torch
import argparse
import os

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from SHREC.Confusion import DrawConfusionMatrix
from model.ST_GCN_AltFormer import ST_GCN_AltFormer
from dataset_node import *
from data_process.Hand_Dataset import Hand_Dataset

parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--data_cfg', type=int, default=0)

def init_data_loader(data_cfg):
    train_data, test_data = split_train_test(data_cfg)
    all_data = test_data

    all_data = [sample for sample in all_data if sample["label"] == 1]

    all_dataset = Hand_Dataset(all_data, use_data_aug=False, time_len=180,expand = 0)


    print("test data num: ", len(all_dataset))

    print("batch size:", args.batch_size)

    all_loader = torch.utils.data.DataLoader(
        all_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=8, pin_memory=False)


    return all_loader

def init_model(data_cfg):
    if data_cfg == 0:
        class_num = 14
    elif data_cfg == 1:
        class_num = 28

    output_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    l = [('labeling_mode', 'spatial')]
    graph_args = dict(l)
    address = 'graph.SHRE'
    model = ST_GCN_AltFormer(channel=3, num_class=class_num, window_size=300, num_point=22, attention=True,
                             only_attention=True,
                             tcn_attention=False, all_layers=False, only_temporal_attention=True, attention_3=False,
                             relative=False,
                             double_channel=True,
                             drop_connect=True, concat_original=True, dv=0.25, dk=0.25, Nh=8, dim_block1=10, dim_block2=30,
                             dim_block3=75,
                             data_normalization=True, visualization=False, skip_conn=True, adjacency=False,
                             kernel_temporal=9, bn_flag=True, weight_matrix=2, more_channels=False, n=4, device=output_device,
                             graph=address,
                             graph_args=graph_args
                             )

    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/zjt/Desktop/ST-GCN-AltFormer/weight/0.975.pth'
    state_dict = torch.load(model_path)

    model.load_state_dict(state_dict)

    model.eval()
    return model

def model_forward(sample_batched, model):
    data = sample_batched["skeleton"].float()
    score,x_st,x_ts = model(data)
    return score,x_st,x_ts

def get_acc(score, labels):
    score = score.cpu().data.numpy()
    labels = labels.cpu().data.numpy()
    outputs = np.argmax(score, axis=1)
    return np.sum(outputs == labels) / float(labels.size)



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.figure(figsize=(19.2, 10.8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    test_loader = init_data_loader(args.data_cfg)


    model = init_model(args.data_cfg)

    correct = 0
    correct1 = 0
    correct2 = 0
    total = 0
    true_labels = []  # 存储真实标签
    true_labels_st = []
    true_labels_ts = []
    predicted_labels = []
    predicted_labels_st = []
    predicted_labels_ts = []

    incorrect_samples = []  # 存储判断错误的样本
    incorrect_labels = []  # 存储判断错误的标签
    incorrect_samples_st = []  # 存储判断错误的样本
    incorrect_labels_st = []  # 存储判断错误的标签
    incorrect_samples_ts = []  # 存储判断错误的样本
    incorrect_labels_ts = []  # 存储判断错误的标签
    start_time = time.time()
    correct_label = 7
    with torch.no_grad():
        for sample_batched in test_loader:
            data = sample_batched["skeleton"].float()
            label = sample_batched["label"]
            label = label.type(torch.LongTensor)
            label = label.cuda()
            score,x_st,x_ts = model_forward(sample_batched, model)
            predictions = np.argmax(score.cpu().numpy(), axis=1)
            predictions_st = np.argmax(x_st.cpu().numpy(), axis=1)
            predictions_ts = np.argmax(x_ts.cpu().numpy(), axis=1)
            acc = get_acc(score, label)
            acc1 = get_acc(x_st, label)
            acc2 = get_acc(x_ts, label)
            correct += acc * label.size(0)
            correct1 += acc1 * label.size(0)
            correct2 += acc2 * label.size(0)
            total += label.size(0)
            true_labels.extend(label.cpu().numpy())
            predicted_labels.extend(predictions)
            predicted_labels_st.extend(predictions_st)
            predicted_labels_ts.extend(predictions_ts)

    end_time = time.time()  # End measuring inference time
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")

    if total == 0 :
        print("Test Accuracy: {:.2f}%".format(correct * 100))
        print("st Accuracy: {:.2f}%".format(correct1 * 100))
        print("ts Accuracy: {:.2f}%".format(correct2 * 100))
    else:
        test_accuracy = correct / total
        test_accuracy1 = correct1 / total
        test_accuracy2 = correct2 / total
        print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
        print("st Accuracy: {:.2f}%".format(test_accuracy1 * 100))
        print("ts Accuracy: {:.2f}%".format(test_accuracy2 * 100))

    #输出判断错误的样本及其标签

    print(len(predicted_labels))
    print(predicted_labels)
    incorrect_samples =  [(index, sample) for index, sample in enumerate(predicted_labels) if sample != correct_label]
    print(incorrect_samples)

    # print(len(predicted_labels_st))
    # print(predicted_labels_st)
    incorrect_samples_st = [(index, sample) for index, sample in enumerate(predicted_labels_st) if sample != correct_label]
    print(incorrect_samples_st)

    # print(len(predicted_labels_ts))
    # print(predicted_labels_ts)
    incorrect_samples_ts = [(index, sample) for index, sample in enumerate(predicted_labels_ts) if sample != correct_label]
    print(incorrect_samples_ts)


    # # draw pic
    # cm = confusion_matrix(true_labels, predicted_labels)
    # # # class_names = ['Grab(1)', 'Grab(2)','Tap(1)','Tap(2)','Expand(1)','Expand(2)', 'Pinch(1)', 'Pinch(2)','Rotation CW(1)', 'Rotation CW(2)','Rotation CCW(1)', 'Rotation CCW(2)','Swipe Right(1)','Swipe Right(2)','Swipe Left(1)','Swipe Left(2)',
    # # #                'Swipe Up(1)', 'Swipe Up(2)','Swipe Down(1)', 'Swipe Down(2)','Swipe X(1)','Swipe X(2)', 'Swipe +(1)','Swipe +(2)','Swipe V(1)', 'Swipe V(2)', 'Shake(1)','Shake(2)'] # 替换为你的类别标签
    # class_names = ['Grab', 'Tap', 'Expand', 'Pinch', 'Rotation CW', 'Rotation CCW', 'Swipe Right', 'Swipe Left', 'Swipe Up',
    #  'Swipe Down', 'Swipe X', 'Swipe +', 'Swipe V', 'Shake']
    # plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
    # plt.savefig("/data/zjt/handgesture/pic/ST_confusion_matrix.png")
    # plt.show()