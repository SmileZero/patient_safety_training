import sys
import time
from urllib.request import *
import simplejson
import pysolr
import config
import itertools 
import pandas as pd
import nltk
import csv
import spacy
from nltk.corpus import stopwords
import re
import string
import numpy as np
import collections
import tensorflow as tf
from gensim.models import KeyedVectors

nlp = spacy.load('en')

stop_words = stopwords.words('english')
fromID = sys.argv[1]
toID = sys.argv[2]
lastID = toID
start_time = time.time()
phrase_length = 27


### Phrase Patterns ################################################################################################
seeds = pd.read_csv('PhrasePatterns1.txt', header=None)
seedPhrases1 = seeds[0]
seedPhrases1 = [i.lower() for i in list(seedPhrases1)]
seeds = pd.read_csv('PhrasePatterns2.txt', header=None)
seedPhrases2 = seeds[0]
seedPhrases2 = [i.lower() for i in list(seedPhrases2)]
model = KeyedVectors.load_word2vec_format('PubMed-and-PMC-w2v.bin', binary=True)
#seeds = pd.read_csv('PhrasePatterns3.txt', header=None)
#seedPhrases3 = seeds[0]
#seedPhrases3 = [i.lower() for i in list(seedPhrases3)]

seedPhrases = []
seedPhrases.extend(seedPhrases1)
seedPhrases.extend(seedPhrases2)
#seedPhrases.extend(seedPhrases3)

seedWords=set(itertools.chain(* [i.lower().split(' ') for i in seedPhrases]))

### No AE Phrases ##################################################################################################
noAESeeds = pd.read_csv('NoEventSeeds.txt', header=None)
noAESeedPhrases = noAESeeds[0]

def checkNoAE(text):
    x = [sent for sent in text if any(seed in str(sent) for seed in noAESeedPhrases)]
    if len(x) == 0:
        return 'Y'
    else:
        return 'N'

### Nouns and Noun phrases #########################################################################################
def get_term(text,stopwords):
    sentence_re = r'(?:(?:[A-Z])(?:.[A-Z])+.?)|(?:\w+(?:-\w+)*)|(?:\$?\d+(?:.\d+)?%?)|(?:...|)(?:[][.,;"\'?():-_`])'
    lemmatizer = nltk.WordNetLemmatizer()
    stemmer = nltk.stem.porter.PorterStemmer()
    grammar = r"""
        NBAR:
            {<NN.*|JJ.*|VBD.*|VBG.*|VBN.*>*<NN.*>}

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  
    """
    chunker = nltk.RegexpParser(grammar)
    text = text.replace("\\", "").strip()
    text = text.replace("'s","").replace("'","")
    postoks = nltk.tag.pos_tag(nltk.word_tokenize(text))
    tree = chunker.parse(postoks)

    def leaves(tree):
        """Finds NP (nounphrase) leaf nodes of a chunk tree."""
        for subtree in tree.subtrees(filter = lambda t: t.label()=='NP'):
            yield subtree.leaves()

    def acceptable_word(word):
        """Checks conditions for acceptable word: length, stopword. We can increase the length if we want to consider large phrase"""
        accepted = bool(2 <= len(word) <= 40 and word.lower() not in stopwords)
        return accepted

    def normalise(word):
        """Normalises words to lowercase and stems and lemmatizes it."""
        word = word.lower()
        #word = re.sub(r'[^\w\s]','',word)
        # word = stemmer.stem_word(word) #if we consider stemmer then results comes with stemmed word, but in this case word will not match with comment
        word = lemmatizer.lemmatize(word)
        return word

# if acceptable_word(w) 
    def get_terms(tree):
        for leaf in leaves(tree):
            term = [normalise(w) for w,t in leaf if acceptable_word(w)]
            yield term

    terms = get_terms(tree)
    return list(terms)

### extract significant sentences ##################################################################################
def extract(text):
    extractedtext = [sent for sent in text if any(seed in str(sent) for seed in seedPhrases1)]
    if len(extractedtext) != 0 :
        return extractedtext
    else:
        extractedtext = [sent for sent in text if any(seed in str(sent) for seed in seedPhrases2)]
        if len(extractedtext) != 0 :
            return extractedtext
        else:
            return text
        

### Extract Noun Phrases ##################################################################################################
def nounPhraseExtractor(extractedtext):
    terms = []
    for i in list(extractedtext):
        terms.extend(get_term(str(i),stop_words))
    phrases = []
    nounPhrases = []
    for i in terms:
        phrases.append(' '.join(i))
    for i in phrases:
        nounPhrases.append(' '.join([word for word in i.split(" ") if word not in seedWords]))
    nounPhrases = list(set([i for i in nounPhrases if i]))
    nounPhrases = [i for i in nounPhrases if not bool(re.search(r'\d', i))]
    return nounPhrases


### tag the data ##################################################################################################
def tagger(nounPhrases,model,pl):
    if len(nounPhrases) == 0:
        return []
    graph = tf.Graph()
    with graph.as_default():
        # Configure the model
        session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
        sess = tf.Session(config=session_conf)

        # Begin the session
        with sess.as_default():
            pretrained = tf.train.import_meta_graph('./current_cnn_model/model.meta')
            pretrained.restore(sess,tf.train.latest_checkpoint('./current_cnn_model/'))
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("predictions").outputs[0]
            clean_plist = []
            for phrase in nounPhrases:
                phrase = re.sub(r'[^\w\s]',' ',phrase)
                final_clean = re.sub(r"\s+", " ", phrase)
                clean_plist.append(final_clean)
            AETags = []
            preds = np.array(predictions.eval(feed_dict={input_x: np.array(get_vectors(model,clean_plist,pl))}))
            for i in range(len(preds)):
                if preds[i] == 1:
                    AETags.append(clean_plist[i])
    return AETags

### Get the data vector ##################################################################################################
# model is the pretrained model. We pass it as parameter so we just need to load the model once
# feeds is a list of twitter feeds we get from data loading part
def get_vectors(model,feeds,pl):
    # high dimensional data which will be assigned to input tensor
    final_input = []
    for sentence in feeds:
        #the matrix of one certain feed
        mid_input = []
        i = 0
        for word in sentence.split(' '):
            i = i + 1
            if word in model.vocab:
                mid_input.append(model.word_vec(word))
            else:
                mid_input.append(model.word_vec('the'))
        #fill the feed to the largest length
        while i < pl:
            mid_input.append(model.word_vec('the'))
            i = i+1
        final_input.append(mid_input)
    return final_input

### Main method ##################################################################################################
def getData(a,b,model):
    taggedAE = pd.DataFrame(columns=["mdr_report_key","ManufacturerName", "BrandName", "GenericName", "Text", "NounPhrases","HasAE","AETags"])
    connection = urlopen(config.solrURL+'maude/select?fl=mdr_report_key,%20device.brand_name,%20device.manufacturer_d_name,%20device.generic_name,%20mdr_text.text&fq=mdr_report_key:['+a+'%20TO%20'+b+']&q=adverse_event_flag:"Y"&rows=100000000&wt=json')
              
    solr = pysolr.Solr(config.solrURL+'adverseEvents/',timeout=1000000)
    response = simplejson.load(connection)
    print (response['response']['numFound'], "documents found.")
    
    for doc in response['response']['docs']:
        brand_name = doc['device.brand_name'][0] if 'device.brand_name' in doc else None
        manufacturer_d_name = doc['device.manufacturer_d_name'][0] if 'device.manufacturer_d_name' in doc else None
        generic_name = doc['device.generic_name'][0] if 'device.generic_name' in doc else None
        text = doc['mdr_text.text'][0] if 'mdr_text.text' in doc else None
        if text:
            text = text.lower()
            text = text.strip()
            sentences = list(nlp(str(text)).sents)
            hasAE = checkNoAE(sentences)
            if hasAE == "Y":
                extractedtext = extract(sentences)
                nounPhrases = nounPhraseExtractor(extractedtext)
                AETags = tagger(nounPhrases,model,phrase_length)
        else:
            text = ""
            hasAE = "N"
            nounPhrases= []
            AETags = []
        AEDictionary = {"id" : doc['mdr_report_key'],
                        "mdr_report_key":doc['mdr_report_key'],
                       "BrandName":brand_name,
                       "ManufacturerName":manufacturer_d_name,
                       "GenericName":generic_name,
                       "Text":text,
                       "NounPhrases":nounPhrases,
                       "HasAE":hasAE,
                       "AETags":AETags
                       }

        taggedAE = taggedAE.append(AEDictionary , ignore_index=True)
    solr.add(taggedAE.T.to_dict().values())
    print("write complete")
    
    
getData(fromID,toID,model)

print("")
print("--- %f Minutes ---" % ((time.time() - start_time)/60))
print("--- %s last record ---" % (lastID))