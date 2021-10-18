import pandas as pd
import jieba
import re

file_name = ['train', 'dev', 'test']
data = pd.read_csv('data/BQ/{}'.format(file_name[0]), sep='\t', header=None)

new_data = []
for _, item in data.iterrows():
	try:
		word1 = set(jieba.cut(item[0]))
		word2 = set(jieba.cut(item[1]))
		common_words = word1 & word2
		for common_word in common_words:
			try:
				item[0] = re.sub(common_word, '', item[0])
				item[1] = re.sub(common_word, '', item[1])
			except:
				print(1)
		new_data.append([
			item[0],
			item[1],
			item[2]])
	except:
		print(1)

new_data = pd.DataFrame(new_data)
#new_data.to_csv('data/BQ/{}_process'.format(file_name[0]), sep='\t', header = None, index = None)
new_data.to_csv('data/BQ/{}_process', sep='\t', header = None, index = None)