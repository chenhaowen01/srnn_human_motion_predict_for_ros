from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange # pylint: disable=redefined-builtin
import copy

def readCSVasFloat(filename):
  """
  Borrowed from SRNN code. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

  Args
    filename: string. Path to the csv file
  Returns
    returnArray: the read data in a float32 matrix
  """
  returnArray = []
  lines = open(filename).readlines()
  for line in lines:
    line = line.strip().split(',')
    if len(line) > 0:
      returnArray.append(np.array([np.float32(x) for x in line]))

  returnArray = np.array(returnArray)
  return returnArray

def writeFloatAsCVS(filename, sequence):
  with open(filename, 'w') as f:
    for line in sequence:
      line_str = [ str(i) for i in line ]
      f.write(','.join(line_str))
      f.write('\n')

def normalization_stats(completeData):
  """"
  Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

  Args
    completeData: nx99 matrix with data to normalize
  Returns
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    dimensions_to_use: vector with dimensions used by the model
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  # dimensions_to_use    = []

  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  # dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0

  new_idx = []
  count = 0
  for i in range(completeData.shape[1]):
    if i in dimensions_to_ignore:
      new_idx.append(-1)
    else:
      new_idx.append(count)
      count += 1

  return data_mean, data_std, dimensions_to_ignore, np.array(new_idx)

def normalize_data( data, data_mean, data_std):
  """
  Normalize input data by removing unused dimensions, subtracting the mean and
  dividing by the standard deviation

  Args
    data: nx99 matrix with data to normalize
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dim_to_use: vector with dimensions used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    data_out: the passed data matrix, but normalized
  """
  data_out = np.divide( (data - data_mean), data_std )

  return data_out

def unnormalize_data(normalizedData, data_mean, data_std):
  """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

  Args
    normalizedData: nxd matrix with normalized data
    data_mean: vector of mean used to normalize the data
    data_std: vector of standard deviation used to normalize the data
    dimensions_to_ignore: vector with dimensions not used by the model
    actions: list of strings with the encoded actions
    one_hot: whether the data comes with one-hot encoding
  Returns
    origData: data originally used to
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(normalizedData, stdMat) + meanMat
  return origData

def load_crf_graph(crf):
  global nodeNames, nodeList, nodeToEdgeConnections, nodeConnections, nodeFeatureLength, edgeList, edgeFeatures

  lines = open(crf).readlines()
  nodeOrder = []
  nodeNames = {}
  nodeList = {}
  nodeToEdgeConnections = {}
  nodeFeatureLength = {}
  for node_name, node_type in zip(lines[0].strip().split(','),lines[1].strip().split(',')):
    nodeOrder.append(node_name)
    nodeNames[node_name] = node_type
    nodeList[node_type] = 0
    nodeToEdgeConnections[node_type] = {}
    nodeToEdgeConnections[node_type][node_type+'_input'] = [0,0]
    nodeFeatureLength[node_type] = 0
  
  edgeList = []
  edgeFeatures = {}
  nodeConnections = {}
  edgeListComplete = []
  for i in range(2,len(lines)):
    first_nodeName = nodeOrder[i-2]
    first_nodeType = nodeNames[first_nodeName]
    nodeConnections[first_nodeName] = []
    connections = lines[i].strip().split(',')
    for j in range(len(connections)):
      if connections[j] == '1':
        second_nodeName = nodeOrder[j]
        second_nodeType = nodeNames[second_nodeName]
        nodeConnections[first_nodeName].append(second_nodeName)
    
        edgeType_1 = first_nodeType + '_' + second_nodeType
        edgeType_2 = second_nodeType + '_' + first_nodeType
        edgeType = ''
        if edgeType_1 in edgeList:
          edgeType = edgeType_1
          continue
        elif edgeType_2 in edgeList:
          edgeType = edgeType_2
          continue
        else:
          edgeType = edgeType_1
        edgeList.append(edgeType)
        edgeListComplete.append(edgeType)

        if (first_nodeType + '_input') not in edgeListComplete:
          edgeListComplete.append(first_nodeType + '_input')
        if (second_nodeType + '_input') not in edgeListComplete:
          edgeListComplete.append(second_nodeType + '_input')

        edgeFeatures[edgeType] = 0
        nodeToEdgeConnections[first_nodeType][edgeType] = [0,0]
        nodeToEdgeConnections[second_nodeType][edgeType] = [0,0]

node_features_ranges={}
node_features_ranges['torso'] = range(6)
node_features_ranges['torso'].extend(range(36,51))
node_features_ranges['right_arm'] = range(75,99)
node_features_ranges['left_arm'] = range(51,75)
node_features_ranges['right_leg'] = range(6,21)
node_features_ranges['left_leg'] = range(21,36)
drop_right_knee = [9,10,11]

def skeleto_to_feature(skeleto):
  data_mean, data_std, dimensions_to_ignore, new_idx = normalization_stats(skeleto)
  normalized_skeleto = normalize_data(skeleto, data_mean, data_std)
  features = {}
  nodeNames = node_features_ranges.keys()
  for nm in nodeNames:
    filterList = []
    for x in node_features_ranges[nm]:
      if x not in dimensions_to_ignore:
        filterList.append(x)
    features[nm] = normalized_skeleto[:, filterList]
    T, D = features[nm].shape
    features[nm] = features[nm].reshape(T, 1, D)
    
  return features, data_mean, data_std, dimensions_to_ignore, new_idx

def feature_to_skeleto(feature, data_mean, data_std, dimensions_to_ignore, new_idx):
  keys = feature.keys()
  [T, N, D]  = feature[keys[0]].shape
  # print("D: ", D)
  D = len(new_idx) - len(np.where(new_idx < 0)[0])
  single_vec = np.zeros((T, D),dtype=np.float32)
  for k in keys:
    nm = k.split(':')[0]
    idx = new_idx[node_features_ranges[nm]]
    insert_at = np.delete(idx,np.where(idx < 0))
    # print("k: ", k)
    # print("idx: ", idx)
    # print("insert_at: ", insert_at)
    single_vec[:, insert_at] = feature[k][:, 0, :]
  
  T = single_vec.shape[0]
  D = data_mean.shape[0]
  origData = np.zeros((T,D),dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if not len(dimensions_to_use) == single_vec.shape[1]:
    return []
    
  origData[:,dimensions_to_use] = single_vec
  
  return unnormalize_data(origData, data_mean, data_std)

def get_feature(feature, nodeName, edgeType):
  # print(feature.keys())
  if edgeType.split('_')[1] == 'input':
    return feature[nodeName]
  
  features = []
  nodesConnectedTo = nodeConnections[nodeName]
  for nm in nodesConnectedTo:
    et1 = nodeNames[nm] + '_' + nodeNames[nodeName]
    et2 = nodeNames[nodeName] + '_' + nodeNames[nm]
    
    f1 = 0
    f2 = 0

    x = 0
    y = 0
    if nm == 'torso':
      x = 0
    if nodeName == 'torso':
      y = 0

    if et1 == et2 and et1 == edgeType:
      f1 = feature[nodeName][:,:,y:] 
      f2 = feature[nm][:,:,x:]
    elif et1 == edgeType:
      f1 = feature[nm][:,:,x:] 
      f2 = feature[nodeName][:,:,y:]
    elif et2 == edgeType:
      f1 = feature[nodeName][:,:,y:] 
      f2 = feature[nm][:,:,x:]
    else:
      continue

    if len(features) == 0:
      features = np.concatenate((f1,f2),axis=2)
    else:
      features += np.concatenate((f1,f2),axis=2)

  return features

def get_node_feature(nodeName, nodeFeatures):
  edge_features = {}
  nodeType = nodeNames[nodeName]
  edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()

  for edgeType in edgeTypesConnectedTo:
    edge_features[edgeType] = get_feature(nodeFeatures, nodeName, edgeType)

  edgeType = nodeType + '_input'
  nodeRNNFeatures = copy.deepcopy(edge_features[edgeType])

  for edgeType in edgeList:
    if edgeType not in edgeTypesConnectedTo:
      continue
    nodeRNNFeatures = np.concatenate((nodeRNNFeatures,edge_features[edgeType]),axis=2)

  return nodeRNNFeatures

def get_predict_data(feature):
  forecast = {}
  forecast_node_feature = {}
  for nodeName in nodeNames.keys():
    forecast_edge_features = {}

    nodeType = nodeNames[nodeName]
    edgeTypesConnectedTo = nodeToEdgeConnections[nodeType].keys()

    for edgeType in edgeTypesConnectedTo:
      forecast_edge_features[edgeType] = get_feature(feature, nodeName, edgeType)
    
    edgeType = nodeType + '_input'
    forecast_nodeRNNFeatures = copy.deepcopy(forecast_edge_features[edgeType])

    for edgeType in edgeList:
      if edgeType not in edgeTypesConnectedTo:
        continue
      forecast_nodeRNNFeatures = np.concatenate((forecast_nodeRNNFeatures,forecast_edge_features[edgeType]),axis=2)
    
    idx = nodeName + ':' + nodeType
    forecast[idx] = forecast_nodeRNNFeatures
    forecast_node_feature[idx] = feature[nodeName]
  return forecast, forecast_node_feature

# just for test
if __name__ == '__main__':
  load_crf_graph('./crf')
  print(nodeNames, nodeList, nodeToEdgeConnections, nodeConnections, nodeFeatureLength, edgeList, edgeFeatures)
  seq = readCSVasFloat('/home/chw/srnn/h3.6m/dataset/S1/walking_1.txt')
  seq = seq[100:200:2, :]
  features, data_mean, data_std, dimensions_to_ignore, new_idx = skeleto_to_feature(seq)
  print(features)
  get_predict_data(features)
  # print(features, data_mean, data_std, dimensions_to_ignore, new_idx)
  seq2 = feature_to_skeleto(features, data_mean, data_std, dimensions_to_ignore, new_idx)