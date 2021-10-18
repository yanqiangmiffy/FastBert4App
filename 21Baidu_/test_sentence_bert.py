'models/yzh930.pth'# 唯一一个实用的测试文件。
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

load_path = 'Qianyan_fgm928.pth3'
out_path ='Qianyan_sentence_bert_fgm101'

model_path = "../model_set/nezha-cn-base"

tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
Config.position_embedding_type = "nezha"
Config.conditional_size =0
Config.conditional_layers =1
maxlen =100
batch_size =1024


fs = [
    'data/test_A.tsv',
]
i=0
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
        # token_id = self.tokenizer(source, target, max_length=self.maxlen, truncation=True, padding='max_length',
        #                           return_tensors='pt')
        token_id=[]
        token_source_id = self.tokenizer(source, truncation=True,max_length=self.maxlen,padding='max_length',
                                  return_tensors='pt')
        token_target_id = self.tokenizer(target, truncation=True,max_length=self.maxlen, padding='max_length',
                                  return_tensors='pt')
        token_id.append({k: token_source_id[k][0] for k in token_source_id.keys()})
        token_id.append({k: token_target_id[k][0] for k in token_target_id.keys()})
        return token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c = self.data[idx][0]
        text_source = self.data[idx][1]
        text_target = self.data[idx][2]
        label=0
        idx=0
        token_id = self.text_to_id(text_source, text_target)
        #return text_source,token_id, c,idx
        return token_id,label, c, idx


testing_data = CustomImageDataset(test_data, tokenizer, maxlen)
test_dataloader = DataLoader(testing_data, batch_size=batch_size)


class CONLINER(nn.Module):
    def __init__(self, Config):
        super(CONLINER, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Config.hidden_size*3, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax()
        )
        self.bert = BertModel(Config)

    def forward(self, source_input_ids, source_attention_mask, source_token_type_ids, target_input_ids,
                target_attention_mask, target_token_type_ids, c):
        source_x1 = self.bert(source_input_ids, attention_mask=source_attention_mask,
                              token_type_ids=source_token_type_ids)
        target_x1 = self.bert(target_input_ids, attention_mask=target_attention_mask,
                              token_type_ids=target_token_type_ids)

        source_mean_pooling = source_x1.last_hidden_state[:, 0]
        target_mean_pooling = target_x1.last_hidden_state[:, 0]

        x2 = torch.cat([source_mean_pooling, target_mean_pooling], dim=-1)
        x2 = torch.cat([x2, torch.abs(source_mean_pooling - target_mean_pooling)], dim=-1)
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
                for line in tqdm(enumerate(test_dataloader)):       
                    batch = line[0]
                    source_data = line[1][0][0]
                    target_data = line[1][0][1]
                    #y = line[1][1]
                    c = line[1][2]
                    source_input_ids = source_data['input_ids'].to(device)
                    source_attention_mask = source_data['attention_mask'].to(device)
                    source_token_type_ids = source_data['token_type_ids'].to(device)

                    target_input_ids = target_data['input_ids'].to(device)
                    target_attention_mask = target_data['attention_mask'].to(device)
                    target_token_type_ids = target_data['token_type_ids'].to(device)
                   #y = y.to(device)
                    c = c.to(device)

                    pred = model(source_input_ids, source_attention_mask, source_token_type_ids, target_input_ids,
                                 target_attention_mask, target_token_type_ids, c)
                    #pred_result = []
                    # bl_cut=1
                    # if bl_cut:
                    #     for pre_value in pred:
                    #         if pre_value[1] > yuzhi:
                    #             pred_result.append(1)
                    #         else:
                    #             pred_result.append(0)
                    # else:
                    pred_result = pred.argmax(1)
                    
                    #y_pred = torch.tensor(pred_result).to(device)
                    for y in pred_result:
                         f.write('%s\n' % ( y.item()))
    print("Done!")

