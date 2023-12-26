import csv
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
min_seq = 180
c = np.zeros((46,3))

def get_LMDHG_dataset():

    train_dataset = []
    test_dataset =[]

    for k in range(1,36):
        df = pd.read_csv("/data/zjt/LMDHG/DataFile{}.csv".format(k))
        df = df.drop(labels='Unnamed: 0', axis=1)

        l = len(df)
        label = df.iloc[0][-1]
        data = np.array(df.iloc[0])
        data = np.delete(data, -1)
        dataset = []
        sample = {}
        dataset.append(data)


        for i in range(1, l):

            if df.iloc[i][-1] == label:
                data = np.array(df.iloc[i])
                data = np.delete(data, -1)
                dataset.append(data)
            else:
                a = len(dataset)
                dataset = np.array(dataset).reshape(a, 46, 3)

                while a < min_seq:
                    dataset = np.append(dataset, [c],axis =0)
                    a = a + 1

                sample = {'skeleton': dataset, 'label': label,'index': k}
                train_dataset.append(sample)
                label = df.iloc[i][-1]
                dataset = []
                data = np.array(df.iloc[i])
                data = np.delete(data, -1)
                dataset.append(data)

        a = len(dataset)
        dataset = np.array(dataset).reshape(a, 46, 3)
        while a < min_seq:
            dataset = np.append(dataset, [c],axis =0)
            a = a + 1
        sample = {'skeleton': dataset, 'label': label,'index': k}
        train_dataset.append(sample)


        # with open('/data/zjt/train_dataset.pkl', 'wb') as f:
    #     pickle.dump(train_dataset, f)
    print("train succeed")
    print("train number:",len(train_dataset))



    for j in range(36,51):
        df = pd.read_csv("/data/zjt/LMDHG/DataFile{}.csv".format(j))
        df = df.drop(labels='Unnamed: 0', axis=1)
        l = len(df)
        label = df.iloc[0][-1]
        data = np.array(df.iloc[0])
        data = np.delete(data, -1)
        dataset = []
        sample = {}
        dataset.append(data)

        for i in range(1, l):

            if df.iloc[i][-1] == label:
                data = np.array(df.iloc[i])
                data = np.delete(data, -1)
                dataset.append(data)
            else:
                a = len(dataset)
                dataset = np.array(dataset).reshape(a, 46, 3)
                while a < min_seq:
                    dataset = np.append(dataset, [c],axis =0)
                    a = a + 1
                sample = {'skeleton': dataset, 'label': label,'index': k}
                test_dataset.append(sample)
                label = df.iloc[i][-1]
                dataset = []
                data = np.array(df.iloc[i])
                data = np.delete(data, -1)
                dataset.append(data)

        a = len(dataset)
        dataset = np.array(dataset).reshape(a, 46, 3)
        while a < min_seq:
            dataset = np.append(dataset, [c],axis =0)
            a = a + 1
        sample = {'skeleton': dataset, 'label': label,'index': k}
        test_dataset.append(sample)

    print("test succeed")
    print("test number:", len(test_dataset))
    return train_dataset,test_dataset

if __name__ == '__main__':
    train,test = get_LMDHG_dataset()
    np.save('/data/zjt/LMDHG/npy/train.npy',train)
    np.save('/data/zjt/LMDHG/npy/test.npy',test)
    l = []
    for i in range(0,779):
        l.append(train[i]['skeleton'].shape[0])


    print(l[1])  #734
    print(len(l))    #779
    print(min(l))
    a = l.index(min(l))
    print(a)
    print(train[a]['skeleton'].shape)
    print(train[a]['skeleton'])



