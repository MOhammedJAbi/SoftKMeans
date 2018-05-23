from __future__ import print_function

import sys
import os
import time
import timeit
import numpy as np
import theano
import theano.tensor as T
import lasagne
from theano.sandbox.rng_mrg import MRG_RandomStreams

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.utils import linear_assignment_
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import softmax

try:
    import cPickle as pickle
except:
    import pickle
import h5py
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mean_squared_error
try:
    from six.moves import xrange
except:
    pass

import scipy
from numpy.matlib import repmat
from scipy.spatial.distance import cdist
from scipy import sparse

theano.config.floatX = 'float32'


def normV(z):
    # Compute the norm ||z||^2
    # Input: Theano 2D tensor
    # Output: Theano 1D tensor
    return theano.tensor.sum(z*z,1)
        
def weightedquaredloss(a,b):
    # Compute sum over dimension 1 of element-wise multiply
    # Input: Theano 2D tensor
    # Output: Theano 1D tensor
    return theano.tensor.sum(a*b,1)
        
def normMatrix(a,b):
    # Compute distance between lines of two matrix , i.e. the (i,k)th element of
    # the output is ||a_i-b_k||^2 where x_j is the j th line of matrix x
    # Input:
    #   - a: matrix of dimension (N*N')
    #   - b: matrix of dimension (K*N')
    # Output:
    #	- matrix of dimension (N*K)

    bdim=b.shape
    adim=a.shape
    al = np.tile(a,bdim[0]).reshape(adim[0]*bdim[0], adim[1])
    bl = np.tile(b.flatten(),(adim[0],1)).reshape(adim[0]*bdim[0], adim[1])
    
    return np.linalg.norm(al-bl, axis=1).reshape(adim[0], bdim[0])**2

def normMatrixT(a,b):
    # Compute distance between lines of two matrix , i.e. the (i,k)th element of
    # the output is ||a_i-b_k||^2 where x_j is the j th line of matrix x
    # Input:
    #   - a: Theano 2D tensor of dimension (N*N')
    #   - b: Theano 2D tensor of dimension (K*N')
    # Output:
    #	- Theano 2D tensor of dimension (N*K)

    bdim = theano.tensor.shape(b)
    adim = theano.tensor.shape(a)
    al = theano.tensor.tile(a,bdim[0])
    al2 = theano.tensor.reshape(al,(adim[0]*bdim[0], adim[1]))
    bl = theano.tensor.tile(b.flatten(),(adim[0],1))
    bl2 = theano.tensor.reshape(bl,(adim[0]*bdim[0], adim[1]))
    
    return theano.tensor.reshape(normV(al2-bl2),(adim[0], bdim[0]))*theano.tensor.reshape(normV(al2-bl2),(adim[0], bdim[0]))

def normMatrixT2(a,b):
    # Second appraoch to compute the distance between lines of two matrix , i.e. the (i,k)th element of
    # the output is ||a_i-b_k||^2 where x_j is the j th line of matrix x
    # Input:
    #   - a: Theano 2D tensor of dimension (N*N')
    #   - b: Theano 2D tensor of dimension (K*N')
    # Output:
    #	- Theano 2D tensor of dimension (N*K)
    norma = theano.tensor.sum(a*a,1)
    normaT = theano.tensor.reshape(norma,(-1,1))
    normb = theano.tensor.sum(b*b,1).T
    ab = theano.tensor.dot(a,b.T)
    c = normaT + normb - 2 * ab
    return c

def ThetaT(q,z):
    # Compute the soft mean of 2D tensor z. 
    # Input: 
    #   - q: Theano 2D tensor of dimension (N*K), it contains the probabilities
    #   - z: Theano 2D tensor of dimension (N*N'), it contains the features
    # Output: Theano 2D tensor of dimension (K*N')

    normalization = theano.tensor.sum(q, axis=0)
    normalizationT = theano.tensor.reshape(normalization,(-1,1))
    ap = theano.tensor.dot(q.T,z)
    
    return ap/normalizationT

def gacPathCondEntropy(IminuszW, cluster_i, cluster_j):
    # Compute conditional complexity from the subpart of the weighted adjacency matrix
    # Inputs:
    #   - IminuszW: the matrix (I - z*P)
    #	- cluster_i: index vector of cluster i
    #	- cluster_j: index vector of cluster j
    # Output:
    #	- L_ij - the sum of conditional complexities of cluster i and j after merging.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    num_i = np.size(cluster_i)
    num_j = np.size(cluster_j)

    # detecting cross elements (this check costs much and is unnecessary)

    ijGroupIndex = np.append(cluster_i, cluster_j)

    y_ij = np.zeros((num_i + num_j, 2))  # [y_i, y_j]
    y_ij[:num_i, 0] = 1
    y_ij[num_i:, 1] = 1
    idx = np.ix_(ijGroupIndex, ijGroupIndex)
    L_ij = scipy.linalg.inv(IminuszW[idx]).dot(y_ij)
    L_ij = sum(L_ij[:num_i, 0]) / (num_i * num_i) + sum(L_ij[num_i:, 1]) / (num_j * num_j)

    return L_ij


def gacPathEntropy(subIminuszW):
    # Compute structural complexity from the subpart of the weighted adjacency matrix
    # Input:
    #   - subIminuszW: the subpart of (I - z*P)
    # Output:
    #	- clusterComp - strucutral complexity of a cluster.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    N = subIminuszW.shape[0]
    clusterComp = scipy.linalg.inv(subIminuszW).dot(np.ones((N, 1)))
    clusterComp = sum(clusterComp) / (N * N)

    return clusterComp


def gacMerging(graphW, initClusters, groupNumber, strDescr, z):
    # Cluster merging for Graph Agglomerative Clustering
    # Implements an agglomerative clustering algorithm based on maiximum graph
    #   strcutural affinity of two groups
    # Inputs:
    #	- graphW: asymmetric weighted adjacency matrix
    #   - initClusters: a cell array of clustered vertices
    #   - groupNumber: the final number of clusters
    #   - strDescr: structural descriptor, 'zeta' or 'path'
    #   - z: (I - z*P), default: 0.01
    # Outputs:
    #   - clusterLabels: 1 x m list whose i-th entry is the group assignment of
    #                   the i-th data vector w_i. Groups are indexed
    #                   sequentially, starting from 1.
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    numSample = graphW.shape[0]
    IminuszW = np.eye(numSample) - z * graphW
    myInf = 1e10

    # initialization
    VERBOSE = True

    numClusters = len(initClusters)
    if numClusters <= groupNumber:
        print('GAC: too few initial clusters. Do not need merging!');

    # compute the structural complexity of each initial cluster
    clusterComp = np.zeros((numClusters, 1))
    for i in xrange(numClusters):
        clusterComp[i] = gacPathEntropy(IminuszW[np.ix_(initClusters[i], initClusters[i])])

    # compute initial(negative) affinity table(upper trianglar matrix), very slow
    if VERBOSE:
        print('   Computing initial table.')

    affinityTab = np.full(shape=(numClusters, numClusters), fill_value=np.inf)
    for j in xrange(numClusters):
        for i in xrange(j):
            affinityTab[i, j] = -1 * gacPathCondEntropy(IminuszW, initClusters[i], initClusters[j])

    affinityTab = (clusterComp + clusterComp.T) + affinityTab

    if VERBOSE:
        print('   Starting merging process')

    curGroupNum = numClusters
    while True:
        if np.mod(curGroupNum, 20) == 0 & VERBOSE:
            print('   Group count: ', str(curGroupNum))

        # Find two clusters with the best affinity
        minAff = np.min(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex1 = np.argmin(affinityTab[:curGroupNum, :curGroupNum], axis=0)
        minIndex2 = np.argmin(minAff)
        minIndex1 = minIndex1[minIndex2]
        if minIndex2 < minIndex1:
            minIndex1, minIndex2 = minIndex2, minIndex1

        # merge the two clusters

        new_cluster = np.unique(np.append(initClusters[minIndex1], initClusters[minIndex2]))

        # move the second cluster to be merged to the end of the cluster array
        # note that we only need to copy the end cluster's information to
        # the second cluster 's position
        if minIndex2 != curGroupNum:
            initClusters[minIndex2] = initClusters[-1]
            clusterComp[minIndex2] = clusterComp[curGroupNum - 1]
            # affinityTab is an upper triangular matrix
            affinityTab[: minIndex2, minIndex2] = affinityTab[:minIndex2, curGroupNum - 1]
            affinityTab[minIndex2, minIndex2 + 1: curGroupNum - 1] = affinityTab[minIndex2 + 1:curGroupNum - 1,
                                                                     curGroupNum - 1]

        # update the first cluster and remove the second cluster
        initClusters[minIndex1] = new_cluster
        initClusters.pop()
        clusterComp[minIndex1] = gacPathEntropy(IminuszW[np.ix_(new_cluster, new_cluster)])
        clusterComp[curGroupNum - 1] = myInf
        affinityTab[:, curGroupNum - 1] = myInf
        affinityTab[curGroupNum - 1, :] = myInf
        curGroupNum = curGroupNum - 1
        if curGroupNum <= groupNumber:
            break

        # update the affinity table for the merged cluster
        for groupIndex1 in xrange(minIndex1):
            affinityTab[groupIndex1, minIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        for groupIndex1 in xrange(minIndex1 + 1, curGroupNum):
            affinityTab[minIndex1, groupIndex1] = -1 * gacPathCondEntropy(IminuszW, initClusters[groupIndex1],
                                                                          new_cluster)
        affinityTab[:minIndex1, minIndex1] = clusterComp[:minIndex1].reshape(-1) + clusterComp[minIndex1] + affinityTab[
                                                                                                            :minIndex1,
                                                                                                            minIndex1]
        affinityTab[minIndex1, minIndex1 + 1: curGroupNum] = clusterComp[minIndex1 + 1: curGroupNum].T + clusterComp[
            minIndex1] + affinityTab[minIndex1, minIndex1 + 1:curGroupNum]

    # generate sample labels
    clusterLabels = np.ones((numSample, 1))
    for i in xrange(len(initClusters)):
        clusterLabels[initClusters[i]] = i
    if VERBOSE:
        print('   Final group count: ', str(curGroupNum))

    return clusterLabels


def gacNNMerge(distance_matrix, NNIndex):
    # merge each vertex with its nearest neighbor
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
    #

    # NN indices
    sampleNum = distance_matrix.shape[0]
    clusterLabels = np.zeros((sampleNum, 1))
    counter = 1
    for i in xrange(sampleNum):
        idx = NNIndex[i, :2]
        assignedCluster = clusterLabels[idx]
        assignedCluster = np.unique(assignedCluster[np.where(assignedCluster > 0)])
        if len(assignedCluster) == 0:
            clusterLabels[idx] = counter
            counter = counter + 1
        elif len(assignedCluster) == 1:
            clusterLabels[idx] = assignedCluster
        else:
            clusterLabels[idx] = assignedCluster[0]
            for j in xrange(1, len(assignedCluster)):
                clusterLabels[np.where(clusterLabels == assignedCluster[j])] = assignedCluster[0]

    uniqueLabels = np.unique(clusterLabels)
    clusterNumber = len(uniqueLabels)

    initialClusters = []
    for i in xrange(clusterNumber):
        initialClusters.append(np.where(clusterLabels[:].flatten() == uniqueLabels[i])[0])

    return initialClusters


def gacBuildDigraph(distance_matrix, K, a):
    # Build directed graph
    # Input:
    #   - distance_matrix: pairwise distances, d_{i -> j}
    #   - K: the number of nearest neighbors for KNN graph
    #   - a: for covariance estimation
    #       sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
    #   - graphW: asymmetric weighted adjacency matrix,
    #               w_{ij} = exp(- d_{ij}^2 / sig2), if j \in N_i^K
    #	- NNIndex: (2K) nearest neighbors, N x (2K+1) matrix
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011

    # NN indices
    N = distance_matrix.shape[0]
    # find 2*K NNs in the sense of given distances
    sortedDist = np.sort(distance_matrix, axis=1)
    NNIndex = np.argsort(distance_matrix, axis=1)
    NNIndex = NNIndex[:, :K + 1]

    # estimate derivation
    sig2 = np.mean(np.mean(sortedDist[:, 1:max(K + 1, 4)])) * a
    #########
    tmpNNDist = np.min(sortedDist[:, 1:], axis=1)
    while any(np.exp(- tmpNNDist / sig2) < 1e-5):  # check sig2 and magnify it if it is too small
        sig2 = 2 * sig2

    #########
    print('  sigma = ', str(np.sqrt(sig2)))

    # build graph
    ND = sortedDist[:, 1:K + 1]
    NI = NNIndex[:, 1:K + 2]
    XI = repmat(np.arange(0, N).reshape(-1, 1), 1, K)
    sig2 = np.double(sig2)
    ND = np.double(ND)
    graphW = sparse.csc_matrix((np.exp(-ND[:] * (1 / sig2)).flatten(), (XI[:].flatten(), NI[:].flatten())),
                               shape=(N, N)).todense()
    graphW += np.eye(N)

    return graphW, NNIndex


def gacCluster(distance_matrix, groupNumber, strDescr, K, a, z):
    # Graph Agglomerative Clustering toolbox
    # Input:
    #   - distance_matrix: pairwise distances, d_{i -> j}
    #   - groupNumber: the final number of clusters
    #   - strDescr: structural descriptor. The choice can be
    #                 - 'zeta':  zeta function based descriptor
    #                 - 'path':  path integral based descriptor
    #   - K: the number of nearest neighbors for KNN graph, default: 20
    #   - p: merging (p+1)-links in l-links algorithm, default: 1
    #   - a: for covariance estimation, default: 1
    #       sigma^2 = (\sum_{i=1}^n \sum_{j \in N_i^K} d_{ij}^2) * a
    #   - z: (I - z*P), default: 0.01
    # Output:
    #   - clusteredLabels: clustering results
    # by Wei Zhang (wzhang009 at gmail.com), June, 8, 2011
    #
    # Please cite the following papers, if you find the code is helpful
    #
    # W. Zhang, D. Zhao, and X. Wang.
    # Agglomerative clustering via maximum incremental path integral.
    # Pattern Recognition, 46 (11): 3056-3065, 2013.
    #
    # W. Zhang, X. Wang, D. Zhao, and X. Tang.
    # Graph Degree Linkage: Agglomerative Clustering on a Directed Graph.
    # in Proceedings of European Conference on Computer Vision (ECCV), 2012.

    print('--------------- Graph Structural Agglomerative Clustering ---------------------');

    # initialization

    print('---------- Building graph and forming initial clusters with l-links ---------');
    [graphW, NNIndex] = gacBuildDigraph(distance_matrix, K, a);
    # from adjacency matrix to probability transition matrix
    graphW = np.array((1. / np.sum(graphW, axis=1))) * np.array(graphW)  # row sum is 1
    initialClusters = gacNNMerge(distance_matrix, NNIndex)

    print('-------------------------- Zeta merging --------------------------');
    clusteredLabels = gacMerging(graphW, initialClusters, groupNumber, strDescr, z);

    return clusteredLabels


def predict_ac_mpi(feat, nClass, nSamples, nfeatures):
    K = 20
    a = 1
    z = 0.01

    distance_matrix = cdist(feat, feat) ** 2
    # path intergral
    label_pre = gacCluster(distance_matrix, nClass, 'path', K, a, z)

    return label_pre[:, 0]


def bestMap(L1, L2):
    if L1.__len__() != L2.__len__():
        print('size(L1) must == size(L2)')

    Label1 = np.unique(L1)
    nClass1 = Label1.__len__()
    Label2 = np.unique(L2)
    nClass2 = Label2.__len__()

    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[i][j] = np.nonzero((L1 == Label1[i]) * (L2 == Label2[j]))[0].__len__()

    c = linear_assignment_.linear_assignment(-G.T)[:, 1]
    newL2 = np.zeros(L2.__len__())
    for i in range(nClass2):
        for j in np.nonzero(L2 == Label2[i])[0]:
            if len(Label1) > c[i]:
                newL2[j] = Label1[c[i]]

    return accuracy_score(L1, newL2)


def dataset_settings(dataset):
    if (dataset == 'MNIST-full') | (dataset == 'MNIST-test'):
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [0, 2]
        test_batch_size = 100
    elif dataset == 'USPS':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [0, 2]
        test_batch_size = 100
    elif dataset == 'FRGC':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [2, 2]
        test_batch_size = 1231
    elif dataset == 'CMU-PIE':
        kernel_sizes = [4, 5]
        strides = [2, 2]
        paddings = [2, 2]
        test_batch_size = 8
    elif dataset == 'YTF':
        kernel_sizes = [5, 4]
        strides = [2, 2]
        paddings = [2, 0]
        test_batch_size = 100

    return kernel_sizes, strides, paddings, test_batch_size


def create_result_dirs(output_path, file_name):
    if not os.path.exists(output_path):
        print('creating log folder')
        os.makedirs(output_path)
        try:
            os.makedirs(os.path.join(output_path, '../params'))
        except:
            pass
        func_file_name = os.path.basename(__file__)
        if func_file_name.split('.')[1] == 'pyc':
            func_file_name = func_file_name[:-1]
        functions_full_path = os.path.join(output_path, func_file_name)
        cmd = 'cp ' + func_file_name + ' "' + functions_full_path + '"'
        os.popen(cmd)
        run_file_full_path = os.path.join(output_path, file_name)
        cmd = 'cp ' + file_name + ' "' + run_file_full_path + '"'
        os.popen(cmd)


class Logger(object):
    def __init__(self, output_path):
        self.terminal = sys.stdout
        self.log = open(output_path + "log.txt", "w+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def kmeans(encoder_val_clean, y, nClusters, y_pred_prev=None, weight_initilization='k-means++', seed=42, n_init=40,
           max_iter=300):
    # weight_initilization = { 'kmeans-pca', 'kmean++', 'random', None }

    if weight_initilization == 'kmeans-pca':

        start_time = timeit.default_timer()
        pca = PCA(n_components=nClusters).fit(encoder_val_clean)
        kmeans_model = KMeans(init=pca.components_, n_clusters=nClusters, n_init=1, max_iter=300, random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        centroids = kmeans_model.cluster_centers_.T

        end_time = timeit.default_timer()

    elif weight_initilization == 'k-means++':

        start_time = timeit.default_timer()
        kmeans_model = KMeans(init='k-means++', n_clusters=nClusters, n_init=n_init, max_iter=max_iter, n_jobs=15,
                              random_state=seed)
        y_pred = kmeans_model.fit_predict(encoder_val_clean)

        D = 1.0 / euclidean_distances(encoder_val_clean, kmeans_model.cluster_centers_, squared=True)
        D **= 2.0 / (2 - 1)
        D /= np.sum(D, axis=1)[:, np.newaxis]

        centroids = kmeans_model.cluster_centers_.T

        end_time = timeit.default_timer()

    print('k-means: \t nmi =', normalized_mutual_info_score(y, y_pred), '\t arc =', adjusted_rand_score(y, y_pred),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
          'K-means objective = {:.1f} '.format(kmeans_model.inertia_), '\t runtime =', end_time - start_time)

    if y_pred_prev is not None:
        print('Different Assignments: ', sum(y_pred == y_pred_prev), '\tbestMap: ', bestMap(y_pred, y_pred_prev),
              '\tdatapoints-bestMap*datapoints: ',
              encoder_val_clean.shape[0] - bestMap(y_pred, y_pred_prev) * encoder_val_clean.shape[0])

    return centroids, kmeans_model.inertia_, y_pred


def load_dataset(dataset_path,dataset="None"):
    if dataset=="MNIST-test":
        hf = h5py.File(dataset_path + '/data_test.h5', 'r')
        X = np.asarray(hf.get('data'), dtype='float32')
        X_train = (X - np.float32(127.5)) / np.float32(127.5)
        y_train = np.asarray(hf.get('label'), dtype='int32')
    elif dataset=="MNIST-full":
        hf = h5py.File(dataset_path + '/data_test.h5', 'r')
        X = np.asarray(hf.get('data'), dtype='float32')
        X_test = (X - np.float32(127.5)) / np.float32(127.5)
        y_test = np.asarray(hf.get('label'), dtype='int32')
        hf_t = h5py.File(dataset_path + '/data_training.h5', 'r')
        X_t = np.asarray(hf_t.get('data'), dtype='float32')
        X_training = (X_t - np.float32(127.5)) / np.float32(127.5)
        y_training = np.asarray(hf_t.get('label'), dtype='int32')
        X_train=np.concatenate((X_training, X_test), axis=0)
        y_train=np.concatenate((y_training, y_test), axis=0)
    else:
        hf = h5py.File(dataset_path + '/data.h5', 'r')
        X = np.asarray(hf.get('data'), dtype='float32')
        X_train = (X - np.float32(127.5)) / np.float32(127.5)
        y_train = np.asarray(hf.get('labels'), dtype='int32')
    return X_train, y_train


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt], excerpt


def build_eml(input_var=None, n_out=None, W_initial=None):
    l_in = input_var

    if W_initial is None:
        l_out = lasagne.layers.DenseLayer(
            l_in, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Uniform(std=0.5, mean=0.5), b=lasagne.init.Constant(1))

    else:
        l_out = lasagne.layers.DenseLayer(
            l_in, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=W_initial, b=lasagne.init.Constant(0))

    return l_out


def build_depict(input_var=None, n_in=[None, None, None], feature_map_sizes=[50, 50],
                 dropouts=[0.1, 0.1, 0.1], kernel_sizes=[5, 5], strides=[2, 2],
                 paddings=[2, 2], hlayer_loss_param=0.1):
    # ENCODER
    l_e0 = lasagne.layers.DropoutLayer(
        lasagne.layers.InputLayer(shape=(None, n_in[0], n_in[1], n_in[2]), input_var=input_var), p=dropouts[0])

    l_e1 = lasagne.layers.DropoutLayer(
        (lasagne.layers.Conv2DLayer(l_e0, num_filters=feature_map_sizes[0], stride=(strides[0], strides[0]),
                                    filter_size=(kernel_sizes[0], kernel_sizes[0]), pad=paddings[0],
                                    nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01),
                                    W=lasagne.init.GlorotUniform())),
        p=dropouts[1])

    l_e2 = lasagne.layers.DropoutLayer(
        (lasagne.layers.Conv2DLayer(l_e1, num_filters=feature_map_sizes[1], stride=(strides[1], strides[1]),
                                    filter_size=(kernel_sizes[1], kernel_sizes[1]), pad=paddings[1],
                                    nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01),
                                    W=lasagne.init.GlorotUniform())),
        p=dropouts[2])

    l_e2_flat = lasagne.layers.flatten(l_e2)

    l_e3 = lasagne.layers.DenseLayer(l_e2_flat, num_units=feature_map_sizes[2],
                                     nonlinearity=lasagne.nonlinearities.tanh)

    # DECODER
    l_d2_flat = lasagne.layers.DenseLayer(l_e3, num_units=l_e2_flat.output_shape[1],
                                          nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01))

    l_d2 = lasagne.layers.reshape(l_d2_flat,
                                  shape=[-1, l_e2.output_shape[1], l_e2.output_shape[2], l_e2.output_shape[3]])

    l_d1 = lasagne.layers.Deconv2DLayer(l_d2, num_filters=feature_map_sizes[0], stride=(strides[1], strides[1]),
                                        filter_size=(kernel_sizes[1], kernel_sizes[1]), crop=paddings[1],
                                        nonlinearity=lasagne.nonlinearities.LeakyRectify(leakiness=0.01))

    l_d0 = lasagne.layers.Deconv2DLayer(l_d1, num_filters=n_in[0], stride=(strides[0], strides[0]),
                                        filter_size=(kernel_sizes[0], kernel_sizes[0]), crop=paddings[0],
                                        nonlinearity=lasagne.nonlinearities.tanh)

    # Loss
    tar0 = input_var
    tar1 = lasagne.layers.get_output(l_e1, deterministic=True)
    tar2 = lasagne.layers.get_output(l_e2, deterministic=True)
    tar3 = lasagne.layers.get_output(l_e3, deterministic=False)
    rec2 = lasagne.layers.get_output(l_d2)
    rec1 = lasagne.layers.get_output(l_d1)
    rec0 = lasagne.layers.get_output(l_d0)
    rec2_clean = lasagne.layers.get_output(l_d2, deterministic=True)
    rec1_clean = lasagne.layers.get_output(l_d1, deterministic=True)
    rec0_clean = lasagne.layers.get_output(l_d0, deterministic=True)

    loss0 = lasagne.objectives.squared_error(rec0, tar0)
    loss1 = lasagne.objectives.squared_error(rec1, tar1) * hlayer_loss_param
    loss2 = lasagne.objectives.squared_error(rec2, tar2) * hlayer_loss_param
    loss3 = lasagne.objectives.squared_error(np.zeros(feature_map_sizes[2]), tar3)
    
    loss0_clean = lasagne.objectives.squared_error(rec0_clean, tar0)
    loss1_clean = lasagne.objectives.squared_error(rec1_clean, tar1) * hlayer_loss_param
    loss2_clean = lasagne.objectives.squared_error(rec2_clean, tar2) * hlayer_loss_param

    loss_recons = loss0.mean() + loss1.mean() + loss2.mean()
    loss_recons_clean = loss0_clean.mean() + loss1_clean.mean() + loss2_clean.mean()

    return l_e3, l_d0, loss_recons, loss_recons_clean, loss3


def train_depict_ae(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, num_clusters, output_path,
                    batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, verbose=1, seed=42,
                    continue_training=False):
    learning_rate_shared = theano.shared(lasagne.utils.floatX(learning_rate))
    params = lasagne.layers.get_all_params(decoder, trainable=True)
    updates = lasagne.updates.adam(loss_recons, params, learning_rate=learning_rate_shared)
    train_fn = theano.function([input_var], loss_recons, updates=updates)
    val_fn = theano.function([input_var], loss_recons_clean)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.10, random_state=42)
    best_val = np.inf
    last_update = 0

    # Load if pretrained weights are available.
    if os.path.isfile(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle')) & continue_training:
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                  "rb") as input_file:
            best_params = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values(decoder, best_params)
    else:
        # TRAIN MODEL
        if verbose > 1:
            encoder_clean = lasagne.layers.get_output(encoder, deterministic=True)
            encoder_clean_function = theano.function([input_var], encoder_clean)

        for epoch in range(num_epochs + 1):
            train_err = 0
            num_batches = 0

            # Training
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                inputs, targets, idx = batch
                train_err += train_fn(inputs)
                num_batches += 1

            validation_error = np.float32(val_fn(X_val))

            print("Epoch {} of {}".format(epoch + 1, num_epochs),
                  "\t  training loss:{:.6f}".format(train_err / num_batches),
                  "\t  validation loss:{:.6f}".format(validation_error))
            # if epoch % 10 == 0:
            last_update += 1
            if validation_error < best_val:
                last_update = 0
                print("new best error: ", validation_error)
                best_val = validation_error
                best_params_values = lasagne.layers.get_all_param_values(decoder)
                with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                          "wb") as output_file:
                    pickle.dump(best_params_values, output_file)
            if last_update > 100:
                break

            if (verbose > 1) & (epoch % 50 == 0):
                # Extract MdA features
                minibatch_flag = 1
                for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                    inputs, targets, idx = batch
                    minibatch_x = encoder_clean_function(inputs)
                    if minibatch_flag:
                        encoder_val_clean = minibatch_x
                        minibatch_flag = 0
                    else:
                        encoder_val_clean = np.concatenate((encoder_val_clean, minibatch_x), axis=0)

                kmeans(encoder_val_clean, y, num_clusters, seed=seed)

        last_params_values = lasagne.layers.get_all_param_values(decoder)
        with open(os.path.join(output_path, '../params/params_' + dataset + '_last.pickle'), "wb") as output_file:
            pickle.dump(params, output_file)
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_last.pickle'),
                  "wb") as output_file:
            pickle.dump(last_params_values, output_file)
        lasagne.layers.set_all_param_values(decoder, best_params_values)


def clustering(dataset, X, y, input_var, encoder, decoder, num_clusters, output_path, test_batch_size=100, seed=42,
               continue_training=False):
    encoder_clean = lasagne.layers.get_output(encoder, deterministic=True)
    encoder_clean_function = theano.function([input_var], encoder_clean)
    
    if os.path.isfile(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle')):
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                  "rb") as input_file:
            best_params = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values(decoder, best_params)
    else:
        sys.exit("Decoder Initialization weigths are not available")

    # Extract MdA features
    minibatch_flag = 1
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        inputs, targets, idx = batch
        minibatch_x = encoder_clean_function(inputs)
        if minibatch_flag:
            encoder_val_clean = minibatch_x
            minibatch_flag = 0
        else:
            encoder_val_clean = np.concatenate((encoder_val_clean, minibatch_x), axis=0)

    # Check kmeans results
    kmeans(encoder_val_clean, y, num_clusters, seed=seed)
    initial_time = timeit.default_timer()
    if (dataset == 'MNIST-full') | (dataset == 'YTF'):
        # K-means on MdA Features
        centroids, inertia, y_pred = kmeans(encoder_val_clean, y, num_clusters, seed=seed)
        y_pred = (np.array(y_pred)).reshape(np.array(y_pred).shape[0], )
        y_pred = y_pred - 1
        with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(y_pred, output_file)
        with open(os.path.join(output_path, '../params/centroids' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(centroids, output_file)        
        
    else:
        # AC-PIC on MdA Features
        if os.path.isfile(os.path.join(output_path, '../params/pred' + dataset + '.pickle')) & continue_training:
            with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'), "rb") as input_file:
                y_pred = pickle.load(input_file, encoding='latin1')
        else:
            try:
                import matlab.engine
                eng = matlab.engine.start_matlab()
                eng.addpath(eng.genpath('matlab'))
                targets_init = eng.predict_ac_mpi(
                    matlab.double(
                        encoder_val_clean.reshape(encoder_val_clean.shape[0] * encoder_val_clean.shape[1]).tolist()),
                    num_clusters, encoder_val_clean.shape[0], encoder_val_clean.shape[1])
                y_pred = (np.array(targets_init)).reshape(np.array(targets_init).shape[0], )
                eng.quit()
                y_pred = y_pred - 1
            except:
                y_pred = predict_ac_mpi(encoder_val_clean, num_clusters, encoder_val_clean.shape[0],
                                        encoder_val_clean.shape[1])
        with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(y_pred, output_file)
        
        final_time = timeit.default_timer()
        print('AC-PIC: \t nmi =  ', normalized_mutual_info_score(y, y_pred),
              '\t arc = ', adjusted_rand_score(y, y_pred),
              '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
              '\t time taken = {:.4f}'.format(final_time - initial_time))
        centroids_acpic = np.zeros(shape=(num_clusters, encoder_val_clean.shape[1]))
        for i in range(num_clusters):
            centroids_acpic[i] = encoder_val_clean[y_pred == i].mean(axis=0)

        centroids = centroids_acpic.T
        with open(os.path.join(output_path, '../params/centroids' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(centroids, output_file)

    return np.int32(y_pred), np.float32(centroids)

    

def train_soft_k_means(dataset, X, y, input_var, decoder, encoder, loss_recons, loss_recons_clean, loss3,  num_clusters, output_path,
                 batch_size=100, test_batch_size=100, num_epochs=1000, learning_rate=1e-4, prediction_status='soft',
                 rec_mult=1, clus_mult=1, reg_lambda=1, init_flag=1, continue_training=False):
    ######################
    #   ADD RLC TO MdA   #
    ######################
    
    initial_time = timeit.default_timer()
        
    # import y_pred and centroids for initialization
        
    if os.path.isfile(os.path.join(output_path, '../params/pred' + dataset + '.pickle')):
            with open(os.path.join(output_path, '../params/pred' + dataset + '.pickle'),
                      "rb") as input_file:
                y_pred = pickle.load(input_file, encoding='latin1')
    else:
        sys.exit("Initialization: y_pred are not available")
        
    if os.path.isfile(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle')):
        with open(os.path.join(output_path, '../params/params_' + dataset + '_values_best.pickle'),
                  "rb") as input_file:
            best_params = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values(decoder, best_params)
    else:
        sys.exit("Initialization: best_params are not available")
    
    rec_lambda = theano.shared(lasagne.utils.floatX(rec_mult))
    clus_lambda = theano.shared(lasagne.utils.floatX(clus_mult))
    pred_normalizition_flag = 1
    num_batches = X.shape[0] // batch_size

    if prediction_status == 'soft':
        target_var = T.matrix('minibatch_out')
        target_var_init = T.matrix('minibatch_out_init')
        theta_init = T.matrix('centroids_init')
        theta_var = T.matrix('centroids_init')
        q_var = T.matrix('Q')
        z_var = T.matrix('Z')
    elif prediction_status == 'hard':
        target_var = T.ivector('minibatch_out')
        target_val = T.vector()
    
    
    reg_lambda_T = theano.shared(lasagne.utils.floatX(reg_lambda))
    num_clusters_T = theano.shared(lasagne.utils.floatX(num_clusters))
    encoder_clean = lasagne.layers.get_output(encoder, deterministic=True)
    encoder_noisy = lasagne.layers.get_output(encoder, deterministic=False)
    encoder_clean_function = theano.function([input_var], encoder_clean)
    loss3 = - normV(encoder_noisy)
    
    norm_opt_init = normMatrixT2(z_var, theta_init)
    Function_Q_init = theano.function([z_var, theta_init], norm_opt_init)
    # For training
    
    prediction_t = normMatrixT2(encoder_noisy,theta_var)
    loss_clus_soft = weightedquaredloss(prediction_t,target_var).mean()/ (reg_lambda_T*num_clusters_T)
    loss_soft = loss_recons + loss3.mean()/ (reg_lambda_T*num_clusters_T) + loss_clus_soft
    params_soft = lasagne.layers.get_all_params(decoder, trainable=True)
    updates_soft = lasagne.updates.adam(loss_soft, params_soft, learning_rate=learning_rate)
    train_soft = theano.function([input_var, theta_var, target_var],
                               [loss_soft, loss_recons], updates=updates_soft)
    
    function_lost = theano.function([input_var, theta_var, target_var],
                               [loss_soft, loss_recons])
    
    thetha_opt = ThetaT(q_var,z_var)
    Function_theta = theano.function([q_var, z_var],thetha_opt)
    
    norm_opt = normMatrixT2(z_var, ThetaT(q_var,z_var))
    Function_Q = theano.function([z_var, q_var], norm_opt)
    
    # For initialization
    
    prediction_t_init = normMatrixT2(encoder_noisy,theta_init)
    loss_clus_soft_init = weightedquaredloss(prediction_t_init,target_var_init).mean()/ (reg_lambda_T*num_clusters_T)
    loss_soft_init = loss_recons + loss3.mean()/ (reg_lambda_T*num_clusters_T) + loss_clus_soft_init
    params_soft_init = lasagne.layers.get_all_params(decoder, trainable=True)
    updates_soft_init = lasagne.updates.adam(loss_soft_init, params_soft_init, learning_rate=learning_rate)
    train_soft_init = theano.function([input_var, theta_init, target_var_init],
                               [loss_soft_init, loss_recons], updates=updates_soft_init)
    
    
    
    final_time = timeit.default_timer()
    

    print("\n...Soft_K_means initialization")
    
    if init_flag:
        if os.path.isfile(os.path.join(output_path, '../params/weights' + dataset + '.pickle')) & continue_training:
            with open(os.path.join(output_path, '../params/weights' + dataset + '.pickle'),
                      "rb") as input_file:
                weights = pickle.load(input_file, encoding='latin1')
                lasagne.layers.set_all_param_values([decoder], weights)
        else:
            y_pred = y_pred.astype(int)
            X_train, X_val, y_train, y_val, y_pred_train, y_pred_val = train_test_split(
                X, y, y_pred , stratify=y, test_size=0.10, random_state=42)
            last_update = 0
            # Initilization
            ind_y = np.copy(y_pred)
            y_targ = np.zeros((ind_y.shape[0], num_clusters))
            y_targ[np.arange(ind_y.shape[0]),ind_y]=1
            
            ind_y_train = np.copy(y_pred_train)
            y_targ_train = np.zeros((ind_y_train.shape[0], num_clusters))
            y_targ_train[np.arange(ind_y_train.shape[0]),ind_y_train]=1
            # y_targ_train = np.copy(y_pred_train)
            y_targ_val = np.copy(y_pred_val)
            
            minibatch_flag = 1
            for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                inputs, targets, idx = batch
                minibatch_x = encoder_clean_function(inputs)
                if minibatch_flag:
                    features_L = minibatch_x
                    minibatch_flag = 0
                else:
                    features_L = np.concatenate((features_L, minibatch_x), axis=0)
                # Theta K
            centroids = Function_theta(np.ndarray.astype(y_targ, 'float32'),features_L)
            
            train_err, val_err = 0, 0
            lossre_train, lossre_val = 0, 0
            num_batches_train = 0
            for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                minibatch_inputs, targets, idx = batch
                minibatch_error, lossrec = function_lost(minibatch_inputs,np.ndarray.astype(centroids, 'float32'), np.ndarray.astype(y_targ[idx], 'float32')) #np.asarray(y_targ_train[idx], dtype=theano.config.floatX)) #np.int32(y_targ_train[idx]),allow_input_downcast=True)
                train_err += minibatch_error
                lossre_train += lossrec
                num_batches_train += 1
            print('Kmeans or AC-PIC over auto-encoder:','\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                      '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                      '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
                      '\t loss= {:.10f}'.format(train_err / num_batches_train),
                      '\t loss_reconstruction= {:.10f}'.format(lossre_train / num_batches_train))
            
            # Z L
            features_L_val = encoder_clean_function(X_val)
            # Q ik
            y_val_prob = - Function_Q_init(features_L_val,np.ndarray.astype(centroids, 'float32'))/(reg_lambda*num_clusters)
            y_val_prob =  softmax(y_val_prob)
            # a ik
            log_bais_val = np.sqrt(np.mean(np.ndarray.astype(y_val_prob, 'float32'), axis=0))
            # b ik
            bais_val = np.log(log_bais_val)#
            # p ik
            y_val_prob = np.dot(features_L_val,np.ndarray.astype(centroids, 'float32').T)+bais_val
            y_val_prob = softmax(y_val_prob)
            y_val_pred = np.argmax(y_val_prob, axis=1)
            
            val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)
            
            best_val = val_nmi
            
            print('initial val nmi: ', val_nmi)
            
                
            best_params_values = lasagne.layers.get_all_param_values([decoder])
            for epoch in range(num_epochs):
                train_err, val_err = 0, 0
                lossre_train, lossre_val = 0, 0
                num_batches_train = 0
                for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    minibatch_inputs, targets, idx = batch
                    minibatch_error, lossrec = train_soft_init(minibatch_inputs,np.ndarray.astype(centroids, 'float32'), np.ndarray.astype(y_targ_train[idx], 'float32')) #np.asarray(y_targ_train[idx], dtype=theano.config.floatX)) #np.int32(y_targ_train[idx]),allow_input_downcast=True)
                    train_err += minibatch_error
                    lossre_train += lossrec
                    num_batches_train += 1
                    
                # Z L
                features_L_val = encoder_clean_function(X_val)
                # Z L
                minibatch_flag = 1
                for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                    inputs, targets, idx = batch
                    minibatch_x = encoder_clean_function(inputs)
                    if minibatch_flag:
                        features_L = minibatch_x
                        minibatch_flag = 0
                    else:
                        features_L = np.concatenate((features_L, minibatch_x), axis=0)
                # Theta K
                centroids = Function_theta(np.ndarray.astype(y_targ, 'float32'),features_L)
                # Q ik
                y_val_prob = - Function_Q_init(features_L_val,np.ndarray.astype(centroids, 'float32'))/(reg_lambda*num_clusters)
                y_val_prob = softmax(y_val_prob)
                # a ik
                log_bais_val = np.sqrt(np.mean(np.ndarray.astype(y_val_prob, 'float32'), axis=0))
                # b ik
                bais_val = np.log(log_bais_val)#
                # p ik
                y_val_prob = np.dot(features_L_val,np.ndarray.astype(centroids, 'float32').T)+bais_val
                y_val_prob = softmax(y_val_prob)
                y_val_pred = np.argmax(y_val_prob, axis=1)

                # Q ik
                y_prob = - Function_Q_init(features_L,np.ndarray.astype(centroids, 'float32'))/(reg_lambda*num_clusters)
                y_prob = softmax(y_prob)
                # a ik
                log_bais = np.sqrt(np.mean(np.ndarray.astype(y_prob, 'float32'), axis=0))
                # b ik
                bais = np.log(log_bais)#
                # p ik
                y_prob = np.dot(features_L,np.ndarray.astype(centroids, 'float32').T)+bais
                y_prob = softmax(y_prob)
                y_pred = np.argmax(y_prob, axis=1)
                val_nmi = normalized_mutual_info_score(y_targ_val, y_val_pred)

                print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                      '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                      '\t acc = {:.4f} '.format(bestMap(y, y_pred)),
                      '\t loss= {:.10f}'.format(train_err / num_batches_train),
                      '\t loss_reconstruction= {:.10f}'.format(lossre_train / num_batches_train),
                      '\t val nmi = {:.4f}  '.format(val_nmi))
                last_update += 1
                if val_nmi > best_val:
                    last_update = 0
                    print("new best val nmi: ", val_nmi)
                    best_val = val_nmi
                    best_params_values = lasagne.layers.get_all_param_values([decoder])
                    # if (losspre_val / num_batches_val) < 0.2:
                    #     break

                if last_update > 5:
                    break

            lasagne.layers.set_all_param_values([decoder], best_params_values)
            with open(os.path.join(output_path, '../params/weights' + dataset + '.pickle'), "wb") as output_file:
                pickle.dump(lasagne.layers.get_all_param_values([decoder]), output_file)
    
    
                
    # Epoch 0
    print("\n...Start Soft_K_means training")
    y_prob_prev = np.zeros((X.shape[0], num_clusters))
    lasagne.layers.set_all_param_values([decoder], best_params_values)
    # Z L
    minibatch_flag = 1
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        inputs, targets, idx = batch
        minibatch_x = encoder_clean_function(inputs)
        if minibatch_flag:
            features_L = minibatch_x
            minibatch_flag = 0
        else:
            features_L = np.concatenate((features_L, minibatch_x), axis=0)
            
    # Theta K
    centroids = Function_theta(np.ndarray.astype(y_targ, 'float32'),features_L)
    # Q ik
    y_prob = - Function_Q_init(features_L,np.ndarray.astype(centroids, 'float32'))/(reg_lambda*num_clusters)
    y_prob = softmax(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    y_prob_prev = np.copy(y_prob)

    print('epoch: 0', '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))
    if os.path.isfile(os.path.join(output_path, '../params/rlc' + dataset + '.pickle')) & continue_training:
        with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'),
                  "rb") as input_file:
            weights = pickle.load(input_file, encoding='latin1')
            lasagne.layers.set_all_param_values([decoder], weights)
    else:
        for epoch in range(num_epochs):

            # In each epoch, we do a full pass over the training data:
            train_err = 0
            lossre = 0
            losspre = 0

            for batch in iterate_minibatches(X, y, batch_size, shuffle=True):
                minibatch_inputs, targets, idx = batch

                # M_step
                if prediction_status == 'hard':
                    minibatch_err, lossrec, losspred = train_fn(minibatch_inputs,
                                                                np.ndarray.astype(y_pred[idx], 'int32'),
                                                                np.ndarray.astype(y_prob_max[idx],
                                                                                  'float32'))
                elif prediction_status == 'soft':
                    minibatch_err, lossrec = train_soft(minibatch_inputs, centroids,
                                                                np.ndarray.astype(y_prob[idx], 'float32'))

                train_err += minibatch_err
                lossre += lossrec
                losspre += 1
                
            # Z L
            minibatch_flag = 1
            for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
                inputs, targets, idx = batch
                minibatch_x = encoder_clean_function(inputs)
                if minibatch_flag:
                    features_L = minibatch_x
                    minibatch_flag = 0
                else:
                    features_L = np.concatenate((features_L, minibatch_x), axis=0)
             
            # Theta K
            centroids = Function_theta(np.ndarray.astype(y_prob, 'float32'),features_L)
            # print (np.sum(np.isnan(centroids)==True))
            
            # Q ik
            y_prob = - Function_Q_init(features_L,centroids)/(reg_lambda*num_clusters)
            y_prob = softmax(y_prob)
            y_pred = np.argmax(y_prob, axis=1)
                
            
            if mean_squared_error(y_prob, y_prob_prev) < 1e-5:
                break
            
            y_prob_prev = np.copy(y_prob)

            print('epoch:', epoch + 1, '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
                  '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
                  '\t acc = {:.4f} '.format(bestMap(y, y_pred)), '\t loss= {:.10f}'.format(train_err / num_batches),
                  '\t loss_recons= {:.10f}'.format(lossre / num_batches))
    
    with open(os.path.join(output_path, '../params/rlc' + dataset + '.pickle'), "wb") as output_file:
        pickle.dump(lasagne.layers.get_all_param_values([decoder]), output_file)
        
    with open(os.path.join(output_path, '../params/qik_final' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(y_prob, output_file)
    # Final
    # Z L
    minibatch_flag = 1
    for batch in iterate_minibatches(X, y, test_batch_size, shuffle=False):
        inputs, targets, idx = batch
        minibatch_x = encoder_clean_function(inputs)
        if minibatch_flag:
            features_L = minibatch_x
            minibatch_flag = 0
        else:
            features_L = np.concatenate((features_L, minibatch_x), axis=0)
    
    # Theta K
    thetaK = Function_theta(np.ndarray.astype(y_prob, 'float32'),features_L)/(reg_lambda*num_clusters)
    
    # a ik
    
    log_bais = np.sqrt(np.mean(np.ndarray.astype(y_prob, 'float32'), axis=0))
                
    # b ik
    
    bais = np.log(log_bais)
    
    # p ik
    y_pred = np.zeros(X.shape[0])
    y_prob = np.dot(features_L,thetaK.T)+bais
    y_prob = softmax(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    print('final: ', '\t Lambda = {:.4f}  '.format(reg_lambda),
          '\t nmi = {:.4f}  '.format(normalized_mutual_info_score(y, y_pred)),
          '\t arc = {:.4f} '.format(adjusted_rand_score(y, y_pred)),
          '\t acc = {:.4f} '.format(bestMap(y, y_pred)))
    
    with open(os.path.join(output_path, '../params/centroids_final' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(thetaK, output_file)
            
    with open(os.path.join(output_path, '../params/pik_final' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(y_prob, output_file)
    
    with open(os.path.join(output_path, '../params/bais_final' + dataset + '.pickle'), "wb") as output_file:
            pickle.dump(bais, output_file)

