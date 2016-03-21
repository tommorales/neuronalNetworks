
import os
import csv
import itertools
import numpy as np
import nltk

PATH = os.getcwd()
FILENAME = 'reddit-comments-2015-08.csv'
FILENAME = os.path.join(PATH, FILENAME)



vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTECE_START"
sentence_end_token = "SENTENCE_STOP"

def loadFile(FILENAME):
    # Step 1: Read the data and append SENTENCE_START and SENTENCE_ENF token
    print "Reading CSV file ... "
    with open (FILENAME, "rb") as f:
        reader = csv.reader(f, skipinitialspace=True)
        reader.next()
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print "Parsed %d sentences." % (len(sentences))

    return sentences
# ***************************************************************************************************************


def build_matrices(sentences):
    # Step 2: Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # ***************************************************************************************************************


    # Step 3: Count the word frecuencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print "Found %d unique words tokens." % len(word_freq.items())
    # ***************************************************************************************************************


    # Step 4: Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size-1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    print "Using vocabulary size %d." % vocabulary_size
    print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
    # ***************************************************************************************************************

    # Step 5: Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print "\nExample sentence: '%s'" % sentences[0]
    print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]
    # ***************************************************************************************************************

    # Step 6: Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    return X_train, y_train
    # ***************************************************************************************************************

