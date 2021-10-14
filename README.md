# FasrBert4App  (torch版本)
当前版本 FastBert 0.1.0  
FasrBert将提供一个快速将Bert和Bert的各种trick落地的方案。复现多种Bert变体方案，支持多种任务，多种预训练模型。
#### 目前支持的预训练模型:
1、NEZHA  
2、roformer  
3、macbert  
4、bert-base
#### 目前支持的(已复现)transformer种类:
1、transformer  
2、transformer-xl  
3、sentence-bert
#### 目前加入的trick方案:
1 PGD  
2 FGM  
3 SWA  
4 Lookahead  
5 梯度惩罚  
6 warm up  
#### 支持的衍生任务
##### 1 多任务学习:  
       1.1  修改layernorm参数和层次  
       1.2  MMOE功能加入  
##### 2 模型蒸馏任务：  
        2.1  logits蒸馏
        2.2  TinyBert蒸馏
#### 目前已经支持的任务:
MNLI  
QQP  
QNLI  
MRPC  
NER  
#### 目前已经支持的调优方案:
1 SimCSE
2 SimCSE-negative 
3 SimCLR
4 SCR
#### 目前更新方案:
1 promote
2 ernie1.0预训练模型加入
3 ernie2.0预训练模型加入
4 Consert模型加入

#### 代码说明


