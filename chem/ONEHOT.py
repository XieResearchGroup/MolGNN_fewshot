import os
import numpy as np
import collections
import itertools
import torch 

ONEHOTENCODING_CODEBOOKS = {}

class ONEHOT_ENCODING(object):

    def __init__(self, dataset):

        self.dataset = dataset 

        
        self.FEATURE_NAMES =  [
                "atom_type",
                "degree",
                "formal_charge",
                "hybridization_type",
                "aromatic",
                "chirality_type",
            ]
        self.ONEHOTENCODING = [0, 1, 2, 3, 4, 5]
        
        
    def get_CODEBOOKS(self):
        global ONEHOTENCODING_CODEBOOKS
        if ONEHOTENCODING_CODEBOOKS:
            print(
                "ONEHOTENCODING_CODEBOOKS is available already, do not need to"
                "regenerate ONEHOTENCODING_CODEBOOKS"
            )
            print(ONEHOTENCODING_CODEBOOKS)
            return 
        
        features_all = [data.x.numpy() for data in self.dataset]
        features = np.vstack(features_all)
        node_attributes_cnt = {}
        for j, col in enumerate(zip(*features)):
            node_attributes_cnt[self.FEATURE_NAMES[j]] = collections.Counter(col)

        ONEHOTENCODING_CODEBOOKS.update({
            feature_name: sorted(node_attributes_cnt[feature_name].keys())
            for feature_name in self.FEATURE_NAMES} )
            
        print(f"generating ONEHOTENCODING_CODEBOOKS......")

        
    def get_onehot_features(self,features):
        feature_one_hot = []
        #print(f'input features{features}')
        for row in features.tolist():
            this_row = []
            for j, feature_val_before_onehot in enumerate(row):
                onehot_code = ONEHOTENCODING_CODEBOOKS[self.FEATURE_NAMES[j]]
                onehot_val = [0.0] * len(onehot_code)
                assert feature_val_before_onehot in onehot_code
                onehot_val[onehot_code.index(feature_val_before_onehot)] = 1.0 
                this_row += onehot_val
            feature_one_hot.append(this_row)
        return torch.Tensor(feature_one_hot)


    def __call__(self, data):

        self.get_CODEBOOKS()
        #print(f'before onehot data {data.x.numpy()}')
        onehot_features = self.get_onehot_features(data.x.numpy()) 
        #print(f'after onehot data{onehot_features.size()}')
        data.x = onehot_features
        #print()
        #print ( data )
        return data
    

    def __repr__(self):
        return f'{self.__class__.__name__}'