import numpy as np
import gensim as gs
import sys
from pprint import pprint

def prepare_files(data_file):
	train = open('A5.train_set.labeled', 'w')
	dev = open('A5.dev_set.labeled', 'w')
	f_len = 0
	with open(data_file) as f:
		for line in f:
			f_len += 1
	i = 0
	with open(data_file) as f:
		for line in f:
			if i < 3600:
				train.write(line)
			else:
				dev.write(line)
			i += 1

def load_labels_and_data(model_file, data_file, smallSentences=False):
	labels = {}
	print "Loading model"
        model = gs.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
	#model = gs.models.Word2Vec.load_word2vec_format(model_file, binary=True)
	print "Model loaded"
	# default these to the most popular sub-categories
	labels['H'] = 0
	labels['M'] = 1

	#
	ret_labels = []
	ref_sentences = []
	can_sentences = []
	with open(data_file) as f:
		for i in xrange(0, len(f)):
			if i % 6 == 0:
				chi_sen = f[i]
			elif i % 6 == 1:
				ref_sen = f[i]
			elif i % 6 == 2:
				can_sen = f[i]
			elif i % 6 == 3:
				score = f[i]
			elif i % 6 == 4:
				label = f[i]
			elif i % 6 == 5:
				lab_vec = np.zeros(2)
				lab_vec[labels[label]] = 1
				ret_labels.append(lab_vec)
				smallSen1 = get_sentence_matrix(ref_sen, model)
				smallSen2 = get_sentence_matrix(can_sen, model)
				ref_sentences.append(smallSen1)
				can_sentences.append(smallSen2)

	return ref_sentences, can_sentences, ret_labels

def get_sentence_matrix(sentence, model):
	try:
		mat = model[sentence[0]]
	except:
		mat = np.zeros(300, dtype=float)
	for i in xrange(1, len(sentence)):
		word = sentence[i]
		try:
			mat = np.column_stack([mat, model[word]])
		except:
			mat = np.column_stack([mat, np.zeros(300, dtype=float)])
        return np.mean(mat, axis=1)


if __name__ == '__main__':
	prepare_files(sys.argv[1])

#if __name__ == '__main__':
#	load_labels_and_data(sys.argv[1], sys.argv[2])
