from model import Model
from preprocessor import Preprocessor

from argparse import ArgumentParser
import pandas as pd
import pickle, json

class PipeLine:
    def __init__(self, model_type, preprocessor_type, predictions_path='predictions.json'):
        # initializing members
        self.model_type = model_type
        self.preprocessor_type = preprocessor_type

        self.model = Model(model_type)
        self.preprocessor = Preprocessor(preprocessor_type)

        self.predictions_path = predictions_path 

    def run(self, dataframe, target=None, test=True):
        if test:
            # load preprocessor and model for testing
            __load(self.preprocessor, __preprocessor_file_name + self.preprocessor_type)
            __load(self.model, __model_file_name + self.model_type)
            # save resutlst to prediction.json file ( where you have 2 keys - prediect_probas and threshold )
            predictions = self.model.predict( self.preprocessor.transform( dataframe ) )
            with open(predictions_path, 'w') as pred_file:
                json.dump(pred_file, {'predicted_probas' : predictions, 'threshold' : 0.85}) # TODO: hardcoded threshold value.
        else:
            # call preprocessor and model for traning
            X, Y = dataframe.drop[target], dataframe[target]
            self.preprocessor.fit(X)
            self.model.fit( self.preprocessor.transform(X), y )

            # save preprocessor and model for future testing
            __save(self.preprocessor, __preprocessor_file_name + self.preprocessor_type)
            __save(self.model, __model_file_name + self.model_type)

    __preprocessor_file_name = '.pipline_preprocessor_'
    __model_file_name = '.pipline_model_'

    # intenal helper functions for saveing/loading
    def __save(obj, file_name):
        with open(file_name, 'rb') as binary_object_file:
            pickle.dump(binary_object_file, obj)

    def __load(obj, file_name):
        with open(file_name, 'rb') as binary_object_file:
            pickle.load(binary_object_file, obj)

# precessing arguments passed through the command line
parser = ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--preprocessor')
parser.add_argument('--data_path')
parser.add_argument('--target')
parser.add_argument('--inference', action='store_true')
parser.add_argument('--predictions_path', default='predictions.json')

args = parser.parse_args() 

# create pipline with specified arguments from command line and run it
pipe = PipeLine(args.model, args.preprocessor, args.predictions_path)
pipe.run( data, args.target, args.inference )
