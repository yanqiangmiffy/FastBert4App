import json
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sohu_bert import BertModel, BertConfig
import os
from torch.nn import init
#from wobert import WoBertTokenizer
from transformers import BertTokenizer

from torch.nn import functional as F
import sys
from tools import FGM
from lookahead import Lookahead


fold = 1
print('making teahcer %s' % fold)

GPU ="1"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

load_path = 'models/yzh930_2.pth'
output_file = 'models/Qin.json'

model_path = "../model_set/nezha-cn-base"
tokenizer = BertTokenizer.from_pretrained(model_path)
Config = BertConfig.from_pretrained(model_path)
Config.position_embedding_type = "nezha"
maxlen =220
Temperature =1
Config.conditional_size =0
Config.conditional_layers =1
batch_size =32
learning_rate =2e-5
epochs =8
use_FGM=True


def load_data(path):
    with open(path) as f:
        for i in f:
            data = json.loads(i)
            return data


train_data = load_data('../data_split/Qianyan_train.json')
test_data=load_data('../data_split/Qianyan_test.json')
#
# train_data = load_data('../data_split/student_train_union.json')
# test_data=load_data('../data_split/student_test_union.json')
class CustomImageDataset(Dataset):

    def __init__(self, data, tokenizer, maxlen, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.transform = transform
        self.target_transform = target_transform

    def text_to_id(self, source, target, c):
        token_id = self.tokenizer(source, target, max_length=self.maxlen, truncation=True, padding='max_length',
                                  return_tensors='pt')
        return {k: token_id[k][0] for k in token_id.keys()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        c =0
        text_source = self.data[idx][1]
        text_target = self.data[idx][2]
        label = self.data[idx][3]
        idx = self.data[idx][4]
        token_id = self.text_to_id(text_source, text_target, c)
        return token_id, label, c, idx


class CONLINER(nn.Module):
    def __init__(self, model_path):
        super(CONLINER, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(Config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Dropout(0.2)
        )
        self.bert = BertModel.from_pretrained(model_path, config=Config)

    def forward(self, input_ids, attention_mask, token_type_ids, c):
        #print(self.bert)
        x1 = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x2 = x1.last_hidden_state[:, 0]
        logits = self.linear_relu_stack(x2)
        return logits


def get_model(model_path):
    model = CONLINER(model_path)
    for i in model.state_dict():
        if 'LayerNorm.bias_dense' in i or 'LayerNorm.weight_dense' in i:
            init.zeros_(model.state_dict()[i])
    model = model.to(device)
    return model


def train(dataloader, model, loss_fn, optimizer, scheduler):
    model.train()
    if use_FGM:
        fgm = FGM(model)
    size = len(dataloader.dataset)
    correct = 0

    for batch, (data, y, c, _) in tqdm(enumerate(dataloader)):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        y = y.to(device)
        c = c.to(device)

        pred = model(input_ids, attention_mask, token_type_ids, c)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        loss = loss_fn(pred, y)
        loss.backward()

        if use_FGM:
            fgm.attack()
            loss_adv = loss_fn(model(input_ids, attention_mask, token_type_ids, c), y)
            loss_adv.backward()
            fgm.restore()

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if batch % 200 == 0:
            loss, current = loss.item(), batch * len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Accuracy: {(100 * correct / size):>0.1f}%")


def test(dataloader, model):
    model.eval()
    TP_a, TN_a, FN_a, FP_a = 0, 0, 0, 0
    #TP_b, TN_b, FN_b, FP_b = 0, 0, 0, 0
    with torch.no_grad():
        for data, y, c, _ in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            y = y.to(device)
            c = c.to(device)
            pred = model(input_ids, attention_mask, token_type_ids, c)
            pred_result = pred.argmax(1)
            TP_a += ((pred_result == 1) & (y == 1) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            TN_a += ((pred_result == 0) & (y == 0) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            FN_a += ((pred_result == 0) & (y == 1) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
            FP_a += ((pred_result == 1) & (y == 0) & ((c == 0) | (c == 2) | (c == 4))).type(torch.float).sum().item()
    p_a = TP_a / (TP_a + FP_a)
    r_a = TP_a / (TP_a + FN_a)

    F1_a = 2 * r_a * p_a / (r_a + p_a)
    F1 = F1_a
    print(f"Test Error: \n ,F1a_score:{(F1_a):>5f} \n")
    return F1

def data_split(data, fold, num_folds,mode):
    """划分训练集和验证集
    """
    if mode == 'train':
        # D = [d for i, d in enumerate(data) if i % num_folds != fold]
        D = [d for i, d in enumerate(data)]
    else:
        # D = [d for i, d in enumerate(data) if i % num_folds == fold]
        D = [d for i, d in enumerate(data)]
    return D


if __name__ == '__main__':
    # train_teacher
    train_ = data_split(train_data[:int(len(train_data)*0.95)], fold, 1, 'train')
    valid_ = data_split(train_data[int(len(train_data)*0.95):], fold, 1, 'valid')
    training_data = CustomImageDataset(train_, tokenizer, maxlen)
    validing_data = CustomImageDataset(valid_, tokenizer, maxlen)
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validing_data, batch_size=batch_size)
    teacher_model = get_model(model_path)

    num_training_steps = epochs * len(training_data) // batch_size
    num_warmup_steps = int(num_training_steps * 0.1)
    print('train num_training_steps:', num_training_steps)
    print('num_warmup_steps', num_warmup_steps)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(teacher_model.parameters(), lr=learning_rate,weight_decay=0.01)
    #optimizer = Lookahead(optimizer, k=5, alpha=0.5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    F1max = 0
    for t in range(epochs):
        print(f"Teacher Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, teacher_model, loss_fn, optimizer, scheduler)
        test_F1 = test(valid_dataloader, teacher_model)
        if test_F1 > F1max:
            F1max = test_F1
            torch.save(teacher_model.state_dict(), load_path+str(t))
            print(f"Higher F1: {(F1max):>5f}%, Saved PyTorch Model State to model.pth")
    print("Teacher training done!")
    # load_best_teacher
    teacher_model.load_state_dict(torch.load(load_path))
    teacher_model.eval()
    # predicting
    all_data = CustomImageDataset(train_data, tokenizer, maxlen)
    all_dataloader = DataLoader(all_data, batch_size=batch_size)