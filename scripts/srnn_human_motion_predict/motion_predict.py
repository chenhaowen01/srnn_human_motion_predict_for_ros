from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import ptvsd
# ptvsd.enable_attach(address=('0.0.0.0', 5679))
# ptvsd.wait_for_attach()

import sys
import numpy as np
import data_utils
from neuralmodels.loadcheckpoint import loadDRA
import copy
import theano

class Predictor:
    def __init__(self, checkpoint):
        data_utils.load_crf_graph('./crf')
        print('loading checkpoint...')
        self._model = loadDRA(checkpoint)
        print('loaded!')

    def _predict_sequence(self, forecast, forecast_node_feature, length):
        teX = copy.deepcopy(forecast)
        nodeNames = teX.keys()

        teY = {}
        to_return = {}
        T = 0
        nodeFeatures_t_1 = {}
        for nm in nodeNames:
            [T,N,D] = teX[nm].shape
            to_return[nm] = np.zeros((T+length,N,D),dtype=theano.config.floatX)
            to_return[nm][:T,:,:] = teX[nm]
            teY[nm] = []
            nodeName = nm.split(':')[0]
            nodeFeatures_t_1[nodeName] = forecast_node_feature[nm][-1:,:,:]

        for i in range(length):
            nodeFeatures = {}
            for nm in nodeNames:
                nt = nm.split(':')[1]
                nodeName = nm.split(':')[0]
                prediction = self._model.predict_node[nt](to_return[nm][:(T+i),:,:],1e-5)
                nodeFeatures[nodeName] = prediction[-1:,:,:]
                teY[nm].append(nodeFeatures[nodeName][0,:,:])
            for nm in nodeNames:
                nt = nm.split(':')[1]
                nodeName = nm.split(':')[0]
                nodeRNNFeatures = data_utils.get_node_feature(nodeName, nodeFeatures)
                to_return[nm][T+i,:,:] = nodeRNNFeatures[0,:,:]
            nodeFeatures_t_1 = copy.deepcopy(nodeFeatures)
        for nm in nodeNames:
            teY[nm] = np.array(teY[nm])
        del teX
        return teY
    
    def predict(self, groud_truth_sequence, length):
        features, data_mean, data_std, dimensions_to_ignore, new_idx = data_utils.skeleto_to_feature(groud_truth_sequence)
        # import pickle
        # with open('./data_stats.pkl') as f:
        #     data_stats = pickle.load(f)
        # with open('./forecast_nodeFeatures.pkl') as f:
        #     features = pickle.load(f)
        # data_mean = data_stats['mean']
        # data_std = data_stats['std']
        # dimensions_to_ignore = data_stats['ignore_dimensions']

        forecast, forecast_node_feature = data_utils.get_predict_data(features)
        predicted_features = self._predict_sequence(forecast, forecast_node_feature, length)
        return data_utils.feature_to_skeleto(predicted_features, data_mean, data_std, dimensions_to_ignore, new_idx)

# just for test
def main():
    if len(sys.argv) < 3:
        return
    checkpoint_path = sys.argv[1]
    data_set_path = sys.argv[2]

    predictor = Predictor(checkpoint_path)
    groud_truth_sequence = data_utils.readCSVasFloat(data_set_path)
    groud_truth_sequence = groud_truth_sequence[100:200:2, :]
    predicted_sequence = predictor.predict(groud_truth_sequence, 100)
    data_utils.writeFloatAsCVS('groud_truth.txt', groud_truth_sequence)
    data_utils.writeFloatAsCVS('predicted.txt', predicted_sequence)

if __name__ == '__main__':
    main()
