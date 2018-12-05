import sys
import pandas as pd
import re
import nltk
import numpy as np
import tensorflow as tf
import os
import time
import datetime
import string
from tensorflow.contrib import learn
from gensim.models import KeyedVectors
from urllib2 import *
import simplejson
import pysolr
import config


class OnlineTraining:
    def __init__(self, validate, status_logger):
        self.validate = validate
        self.status_logger = status_logger

    def get_vectors(self, model,feeds):
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
            while i < 27:
                mid_input.append(model.word_vec('the'))
                i = i+1
            final_input.append(mid_input)
        return final_input


    def getNoise(self):
        noise = []
        connection = urlopen(config.solrURL+'noise/select?fl=phrase_str&q=*:*&rows=1000000000&wt=json')           
        response = simplejson.load(connection)
        #self.status_logger.write(response['response']['numFound'], "documents found.")
        noise = [doc["phrase_str"][0] for doc in response['response']['docs']]
        solrNoise = pysolr.Solr(config.solrURL+'noise/')
        solrNoise.delete(q='*:*')
        NoiseDF = pd.DataFrame(columns=["Label","Phrase"])
        NoiseDF["Phrase"] = noise
        NoiseDF["Label"] = 0
        return NoiseDF


    def onlineTraining(self, new_x,new_y,model,training,sess,input_x,input_y,pretrained):
        pretrained.save(sess, '/data/last_cnn_model/model')
        label = []
        for i in range(len(new_y)):
            if new_y[i] == 0:
                label.append([1,0])
            else:
                label.append([0,1])
        feed_dict = {input_x: self.get_vectors(model,new_x),input_y: label}
        sess.run([training],feed_dict)
        path = pretrained.save(sess, '/data/current_cnn_model/model')
        #path = pretrained.save(sess, './cnn_model/', global_step=current_step)
        self.status_logger.write("Saved model data to {}\n".format(path))


    def validate(self, prediction, val_set, model):
        x = val_set['Phrase']
        y = np.array(val_set['Label'])
        preds = np.array(predictions.eval(feed_dict={input_x: np.array(self.get_vectors(model,x))}))
        corr = 0
        for i in range(len(y)):
            if y[i] == preds[i]:
                corr += 1
        self.status_logger.write('Accuracy on validation set is:')
        #self.status_logger.write(corr * 1.0/len(y))
        self.status_logger.write('Validation down')


    def updateTrainingset(self, newTrainDF,oldTrainDF):
        return pd.concat([newTrainDF,oldTrainDF])


    def generateValidation(self, trainDf,validationFile):
        new_data = trainDf.sample(frac=0.1)
        new_data.to_csv(validationFile,index = False)
        return new_data


    def start(self):
        start_time = time.time()

        # Initiate the pretrained word vector model
        self.status_logger.write('Loading word vector model...')
        self.model = KeyedVectors.load_word2vec_format('/data/PubMed-and-PMC-w2v.bin', binary=True)

        #newtrain = sys.argv[1]
        validate = self.validate

        self.status_logger.write('Begin Training')
        #newTrain = pd.read_csv(newtrain)
        newTrain = self.getNoise()
        trainingDf = pd.read_csv('/data/TrainingData.csv')
        graph = tf.Graph()
        with graph.as_default():
            # Configure the model
            session_conf = tf.ConfigProto(allow_soft_placement=True,log_device_placement=False)
            sess = tf.Session(config=session_conf)

            # Begin the session
            with sess.as_default():
                pretrained = tf.train.import_meta_graph('/data/current_cnn_model/model.meta')
                pretrained.restore(sess,tf.train.latest_checkpoint('/data/current_cnn_model/'))
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                input_y = graph.get_operation_by_name("input_y").outputs[0]
                training = graph.get_operation_by_name("train").outputs[0]
                predictions = graph.get_operation_by_name("predictions").outputs[0]
                self.onlineTraining(newTrain['Phrase'],newTrain['Label'],self.model,training,sess,input_x,input_y,pretrained)
                self.status_logger.write('Update training set...')
                updated = self.updateTrainingset(newTrain,trainingDf)
                updated.to_csv('/data/TrainingData.csv',index = False)
                self.status_logger.write('Complete update')
                if validate == True:
                    self.status_logger.write('Begin validation...')
                    valiSet = self.generateValidation(updated,'/data/ValidationData.csv')
                    self.validate(predictions, valiSet, self.model)
                    self.status_logger.write('validation complete')
                self.status_logger.write('Batch complete')

        self.status_logger.write("")
        self.status_logger.write("--- %f Minutes ---" % ((time.time() - start_time)/60))