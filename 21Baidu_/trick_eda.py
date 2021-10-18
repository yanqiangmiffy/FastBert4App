# max_len是512不对  所以修改model
fs = [
      '../data/LCQMC/train',
      '../data/LCQMC/dev',
     '../data/LCQMC/test',
    '../data/BQ/train',
    '../data/BQ/dev',
    '../data/BQ/test',
    '../data/OPPO/dev',
    '../data/OPPO/train',
    '../data/test_A.tsv'
]
max_len=0
i=0
for f in fs:
        ff=open(f,encoding='utf-8')
        for l in ff:
                l = l.split('\t')
                if len(l[0])>max_len:
                    max_len=len(l[0])
                if len(l[1])>max_len:
                    max_len=len(l[1])
print(max_len)
