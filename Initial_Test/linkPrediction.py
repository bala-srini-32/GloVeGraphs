###packages
import numpy as np
import networkx as nx
import gensim
from sklearn import svm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist
import pickle
from collections import defaultdict
import random
import pickle
import os
import sys

def combine_embedding(method,n_out,n_in):
    if(method == 1):
        #print "Implementing Simple average"
        return (n_out+n_in)/2.0    
    
    elif(method == 2):
        #print "Implementing Hadamard"
        #print n_in,n_out
        return np.multiply(n_in,n_out)
    else:
        print "Invalid Method. Enter 1 or 2"
        return


def evaluate_perf(data,clf,labels):
    label_pred = clf.predict(data)
    #print label_pred.shape
    diff = np.abs(np.subtract(label_pred,labels))
    return np.sum(diff)*1.0/len(labels)


#Argument validation
op = 2
if len(sys.argv) < 4:
	print "Parameters needed: training data path, testing data path and operator"
	try:
		op = int(sys.argv[3])
	except ValueError:
		print "Operator param needs to be integer... defaulting to Hadamard"

# Import training set
training_set = pickle.load(open(sys.argv[1], "rb"))
testing_set = pickle.load(open(sys.argv[2],"rb"))

# Create graph from training set
G = nx.Graph()
for edge in training_set.keys():
    nodes = edge.split('-')
    if(training_set[edge]==1):
        if(G.has_edge(nodes[0],nodes[1])):
            G[nodes[0]][nodes[1]]['weight'] += 1
        else:
            G.add_edge(nodes[0],nodes[1],weight = 1)
    else:
        G.add_node(nodes[0])
        G.add_node(nodes[1])

for node in G.nodes():
    if G.degree(node) == 0:
        G.add_edge(node,node,weight = 1)

# Add only the nodes from test set to graph if not already present in generated graph
node_list_conn = G.nodes()
for edge in testing_set.keys():
    nodes = edge.split('-')
    for node in nodes:
        if node in node_list_conn:
            continue
        else:
            G.add_node(node)

# Build new edgelist node2vec can utilize for generating embeddings
nx.write_edgelist(G,'graph/train_n2v.txt')


#Run embedding
CmdStr = "python main.py --p 1 --q 0.5 --iter 200 --input graph/train_n2v.txt \
    --output emb/emb_train_n2v.emb --dimensions 64"
os.system(CmdStr)

node_list_conn_int = sorted(map(lambda x : int(x),node_list_conn))
node_list_conn = map(lambda x : str(x),node_list_conn_int)


## Read embeddings file from and generate features
model = gensim.models.KeyedVectors.load_word2vec_format('emb/emb_train_n2v.emb')
embeddings = {}
err_count = 0
missing_node = []
for node in node_list_conn:
    try:
        embeddings[node] = model.word_vec(node)
    except:
        err_count += 1
        missing_node.append(node)
        continue


# Training feature vectors
feature = []
label = []
for edge in training_set.keys():
    nodes = edge.split('-')
    try:
    	feature.append(combine_embedding(op,embeddings[nodes[0]],embeddings[nodes[1]]))
    except KeyError:
	continue
    label.append(training_set[edge])
feature_np = np.asarray(feature)
label_np = np.asarray(label)


#x,residuals,rank,s = np.linalg.lstsq(feature_np,label_np)
clf = svm.SVC()
clf.fit(feature_np,label_np)
train_error = evaluate_perf(feature_np,clf,label_np)


#Get test data and evaluate performance
feature_test = []
label_test = []
for edge in testing_set.keys():
    nodes = edge.split('-')
    try:
    	feature_test.append(combine_embedding(op,embeddings[nodes[0]],embeddings[nodes[1]]))
    except KeyError:
	continue
    label_test.append(testing_set[edge])
test_error = evaluate_perf(feature_test,clf,label_test)
print test_error
