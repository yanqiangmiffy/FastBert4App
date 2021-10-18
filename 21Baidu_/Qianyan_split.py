import json
import numpy as np
np.random.seed(2021)

train_data,test_data = [], []

fs = [
    #   'data/LCQMC/train',
    #   'data/LCQMC/dev',
    #   'data/LCQMC/dev_new',
    #  'data/LCQMC/test',
    # 'data/BQ/train',
    # 'data/BQ/dev',
    # 'data/BQ/dev_new',
    # 'data/BQ/test',
    'data/OPPO/dev',
    #'data/OPPO/dev_new',
    'data/OPPO/train',
]
i=0
for f in fs:
        ff=open(f,encoding='utf-8')
        for l in ff:
                l = l.split('\t')
                train_data.append([i, l[0], l[1], int(l[2])])

print("这儿使用test来做验证集")
fs = [
    'data/test_A.tsv'
]
i=0
for f in fs:
    with open(f) as f:
        for l in f:
                l = l.split('\t')
                test_data.append([i, l[0], l[1]])

def save_data(data,path):
    with open(path,'w') as f:
        f.write(json.dumps(data))


train_data=[d + [idx] for idx,d in enumerate(train_data)]
test_data=[d + [idx] for idx,d in enumerate(test_data)]


np.random.shuffle(train_data)
save_data(train_data,'../data_split/Qianyan_train.json')
np.random.shuffle(test_data)
save_data(test_data,'../data_split/Qianyan_test.json')
print(len(train_data))
print(len(test_data))