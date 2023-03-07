import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem((word.lower()))

def bag_of_words(tokenized_sentance, all_words):
    tokenized_sentance = [stem(w) for w in tokenized_sentance]

    bag = np.zetos(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentance:
            bag[idx] = 1.0

    return bag


