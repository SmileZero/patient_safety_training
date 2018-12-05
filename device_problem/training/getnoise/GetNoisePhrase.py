import pandas as pd
import re
import string
from gensim.models import KeyedVectors
import numpy as np
import time
import collections
from scipy import spatial


class GetNoisePhrase:
    def __init__(self, file, status_logger):
        self.file = file
        self.status_logger = status_logger
        self.medical_phrase = []
        self.label = []

    def getCleanSeedWords(self, original,noise_list):
        orin = set(original)
        nl = set(noise_list)
        for nw in nl:
            if nw in orin:
                orin.remove(nw)
        return orin

    def phraseSimilarity(self, dictionary,phrase,m,alpha = 0.8):
        '''
        find the similarity between given phrase and dictionary by calculate the average of the maximum similarity
        among every words in phrase against every word in dictionary
        '''
        simil = []
        for w in phrase.split(' '):
            if w in dictionary:
                simil.append(alpha)
            else:
                if w in m.vocab:
                    #print('w in vocabulary:')
                    #print(w)
                    mid = []
                    for a in dictionary:
                        if a in m.vocab:
                            mid.append(m.similarity(w,a))
                        else:
                            mid.append(m.similarity(w,'the'))
                    simil.append(np.max(mid))
                    #print(np.max(mid))
                else:
                    simil.append(0)
        return np.mean(simil)

    def cleanWordList(self, words):
        cleaned = []
        for w in words:
            if len(w) > 1:
                clean_event =w.lower().replace("'s","").replace("'","")
                clean_event = re.sub(r'[^\w\s]',' ',clean_event)
                final_clean = re.sub(r"\s+", " ", clean_event)
                cleaned.append(final_clean)
        return cleaned



    def cleanTags(self, tags_file, original_lists, noise_words,m,threshhold,outname):
        cleaned = []
        total_neg = []
        tags_df = pd.read_csv(tags_file)
        tags = tags_df['Tags']
        seed_words = self.getCleanSeedWords(original_lists, noise_words)
        start_time = time.time()
        last = ''
        for report in tags:
            last = report
            if isinstance(report, float):
                cleaned.append('Empty Tag')
                continue
            each = []
            negative = []
            words = report.split(r';')
            similarity = []
            words = self.cleanWordList(words)
            self.status_logger.write(words)
            for w in words:
                similarity.append(self.phraseSimilarity(seed_words,w,m,alpha=threshhold))
            #print(similarity)
            aver = np.mean(similarity)
            for i in range(len(similarity)):
                if similarity[i] < threshhold:
                    self.medical_phrase.append(words[i])
                    self.label.append(0)
            #cleaned.append(each)
        #tags_df['cleanedTags'] = cleaned
        #tags_df.to_csv(outname)
        self.status_logger.write("--- %f Minutes ---" % ((time.time() - start_time)/60))
        self.status_logger.write(last)  
        return cleaned


    def getNoise(self):
        model = KeyedVectors.load_word2vec_format('/data/PubMed-and-PMC-w2v.bin', binary=True)
        df = pd.read_csv('/data/Processed-V3.csv')
        #print(df.head())
        mdr_df = pd.read_table('/data/meddra.tsv',sep = '\t',header=None)
        #print(mdr_df[3])


        
        self.label = []
        for event in mdr_df[3]:
            phrase_vector = np.zeros(200)
            lowers = event.lower().split('(')[0]
            clean_event =lowers.replace("'s","").replace("'","")
            clean_event = re.sub(r'[^\w\s]',' ',clean_event)
            final_clean = re.sub(r"\s+", " ", clean_event)
            #words = clean_event.split(" ")
            self.medical_phrase.append(clean_event)
            self.label.append(1)

        seed_list = pd.read_csv('/data/seed_words.csv')['col']
        noise_list = ['device','parent','father','pump','product','system','test','examination','sample','water','pool','water','relief','home','manufacturer','information','investigation']
        #noise_list.extend([i for i,j in c.most_common(50)])
        self.cleanTags('/data/Processed-V3.csv', seed_list, noise_list,model,0.8,'test')


        train_df = pd.DataFrame.from_dict({'Phrase':self.medical_phrase,'Label':self.label})
        #train_df.to_csv('medicalData_V2.csv')
        train_df.to_csv(self.file)

