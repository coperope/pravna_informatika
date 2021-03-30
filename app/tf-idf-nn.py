import pandas as pd
import pickle
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import re
import nltk
import tensorflow as tf
from tensorflow.keras import layers, metrics
import keras.backend.tensorflow_backend as tfb
from tensorflow import keras
import numpy
from sklearn.metrics import classification_report
import sys, getopt

Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()
# stemmer = PorterStemmer()
stemmer = nltk.stem.SnowballStemmer('english')

# Preprocessing functions

def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\-\s]'
    text = re.sub(regex, '', text)
    return text

def noun_verb_extraction(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word, tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb

def stem_words(words):
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
        tag = nltk.pos_tag([word])[0][1]
        lemmatized_words.append(wordlemmatizer.lemmatize(word, get_wordnet_pos(tag)))
    return lemmatized_words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
def preprocess_sentence(text):
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    text = noun_verb_extraction(text)
    text = [word.lower() for word in text if len(word) > 1 and word not in Stopwords]
    text = stem_words(text)
#     text = lemmatize_words(text)
    return text

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    try:
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)    
    except FileNotFoundError as e:
        return False

POS_WEIGHT = 1.7  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor 
    and a target tensor. POS_WEIGHT is used as a multiplier 
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    loss = tf.compat.v1.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

def main(argv):
  inputfile = ''
  try:
     opts, args = getopt.getopt(argv,"hi:o:",["ifile="])
  except getopt.GetoptError:
     print('test.py -i <inputfile>')
     sys.exit(2)
  for opt, arg in opts:
     if opt == '-h':
        print('test.py -i <inputfile>')
        sys.exit()
     elif opt in ("-i", "--ifile"):
        inputfile = arg
  print('Input file is ', inputfile)

  onehot_encoder = load_obj('helpers/onehot_encoder')

  if inputfile.endswith(".json"): 
    concepts = []
    title = ""
    header = ""
    recitals = ""
    main_body = ""
    attachments = ""

    with open('test_files/' + str(inputfile), encoding='utf-8') as json_file:
        data = json.load(json_file)
        concepts = data["concepts"]
        title = data["title"]
        header = data["header"]
        recitals = data["recitals"]
        main_body = '\n'.join(data["main_body"])
        attachments = data["attachments"]
        combined = "\n".join([title, header, recitals])
        preprocessed = [' '.join(preprocess_sentence(combined))]

        tfidf_vectorizer= load_obj('helpers/tfidf_vectorizer')
        tfidf_vectorizer_vectors=tfidf_vectorizer.transform(preprocessed)

        X_test = tfidf_vectorizer_vectors
        model = keras.models.load_model("nn_model.h5", compile=False)
        model.compile(loss=weighted_binary_crossentropy, optimizer="adam", metrics=[metrics.top_k_categorical_accuracy])

        y_pred = model.predict(X_test.todense().reshape(1, -1))

        result_indexes = numpy.where(y_pred > 0.5)[1]
        dim = y_pred.shape[1]
        one_hot_results = []
        for ind in result_indexes:
            res = numpy.zeros(dim) 
            res[ind] = 1
            one_hot_results.append(res)
        print("-"*80)
        print(" Predicted concepts: ")
        print(onehot_encoder.inverse_transform(one_hot_results))
        print("-"*80)

if __name__ == "__main__":
   main(sys.argv[1:])

