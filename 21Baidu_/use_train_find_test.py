# max_len是512不对  所以修改model
from tqdm import tqdm
fs = [
      '../data/LCQMC/train',
      '../data/LCQMC/dev',
     '../data/LCQMC/test',
    '../data/BQ/train',
    '../data/BQ/dev',
    '../data/BQ/test',
    '../data/OPPO/dev',
    '../data/OPPO/train',
]

i=0
train_dict={}
for f in fs:
        ff=open(f,encoding='utf-8')
        for l in ff:
                l = l.split('\t')
                train_dict[l[0]]=l[1]
fs = [
      '../data/test_A.tsv',
]
#test
n=0
test_dict={}
for f in fs:
        ff=open(f,encoding='utf-8')
        for l in tqdm(ff):
                l = l.split('\t')
                test_dict[l[0]]=l[1]

for l0,l1 in test_dict.items():
    if train_dict.get(l0)!=None:
        #if train_dict[l0]==l1:
               n+=1
    if train_dict.get(l1)!=None:
        #if train_dict[l1]==l0:
               n+=1

print(n)

