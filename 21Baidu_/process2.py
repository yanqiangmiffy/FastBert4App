import pandas as pd
import jieba
import re
import string
from tqdm import tqdm
punc = string.punctuation

file_name = ['train', 'dev', 'test']
#path = 'data/BQ/'.format(file_name[0])
path='data/test_A.tsv'
data = pd.read_csv(path, sep='\t', header=None)
wrong=0
new_data = []
for _, item in tqdm(data.iterrows()):
		word1 = set(jieba.cut(item[0]))
		word2 = set(jieba.cut(item[1]))
		common_words = word1 & word2
		for common_word in common_words:
			if common_word not in punc:
				item[0] = re.sub(common_word, '', item[0])
				item[1] = re.sub(common_word, '', item[1])
		new_data.append([
			item[0],
			item[1],
			])

		

new_data = pd.DataFrame(new_data)
new_data.to_csv('data/new_test_A.tsv', sep='\t', header = None, index = None)
#data/test_A.tsv_process

