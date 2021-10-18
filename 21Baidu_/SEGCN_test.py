# 唯一一个实用的测试文件。
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sohu_bert import BertModel, BertConfig
import os
from transformers import BertTokenizer
import sys
from tqdm import tqdm
from torch.nn import functional as F

fold = 1
GPU = "1"
# 用他的方式来跑
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

load_path = '../code/model_saved/student_lookahead_FGM_simcse0804.pth'
out_path ='Qianyan_SEGCN'

model_path = "../model_set/nezha-cn-base"

tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
Config.position_embedding_type = "nezha"
Config.conditional_size = 128
Config.conditional_layers = 2
maxlen = 512
batch_size =128


fs = [
    '../data/test_A.tsv',
]
i=1
test_data=[]
for f in fs:
    with open(f) as f:
        for l in f:
                l = l.split('\t')
                test_data.append([i, l[0], l[1]])


class CustomImageDataset(Dataset):

    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source, target):
        token_id = self.tokenizer(source, target, max_length=self.maxlen, truncation=True, padding='max_length',
                                  return_tensors='pt')
        return {k: token_id[k][0] for k in token_id.keys()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        m=self.data
        c = self.data[idx][0]
        text_source = self.data[idx][1]
        text_target = self.data[idx][2]
        token_id = self.text_to_id(text_source, text_target)
        return text_source,token_id, c


testing_data = CustomImageDataset(test_data, tokenizer, maxlen)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


class CONLINER(nn.Module):
    def __init__(self, Config):
        super(CONLINER, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Config.hidden_size + 128, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax()
        )
        self.standerembed = nn.Embedding(6, 128)
        self.addliner = nn.Embedding(6, 128)
        self.bert = BertModel(Config)

    def forward(self, input_ids, attention_mask, token_type_ids, c):
        conditional = self.standerembed(c)
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, conditional=conditional)
        x2 = x1.last_hidden_state[:, 0]
        x2 = torch.cat([x2, self.addliner(c)], dim=-1)
        logits = self.linear_relu_stack(x2)
        return logits


model = CONLINER(Config)
model = model.to(device)
# print(model)
k = torch.load(load_path)
model.load_state_dict(torch.load(load_path), strict=False)
model.eval()
pred_res_dict = {}
#=====
if __name__ == '__main__':
    real_res=[]
    for yuzhi in (0.5,0.51):
        with open(out_path+str(yuzhi)+'.csv', 'w') as f:
            #f.write('id,label\n')
            with torch.no_grad():
                    for source,data, c in tqdm(test_dataloader):
                        input_ids = data['input_ids'].to(device)
                        attention_mask = data['attention_mask'].to(device)
                        token_type_ids = data['token_type_ids'].to(device)
                        c = c.to(device)
                        pred = model(input_ids, attention_mask, token_type_ids, c)
                        pred_result = []
                        bl_cut=1
                        if bl_cut:
                            for pre_value in pred:
                                if pre_value[1] > yuzhi:
                                    pred_result.append(1)
                                else:
                                    pred_result.append(0)
                        else:
                            pred_result = pred.argmax(1)
                        y_pred = torch.tensor(pred_result).to(device)

                        for y in y_pred:
                            # pred_res_dict[id]=y.item()
                            # f.write('%s,%s\n' % (id, y.item()))
                            f.write('%s\n' % ( y.item()))
    print("Done!")
# ---

