import os
import sys
from subprocess import call
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def evaluate(filename):
	#read ground truth
	communities = {}
	doc = open(filename, "r")
	for line in doc:
		try:
			a  = line.split()
			communities[float(a[0])] = float(a[1])
		except ValueError:
			continue
	doc.close()
	noof_comm = len(set(communities.values()))
	
	#generate embeddings
	graphfname = filename.replace("community", "network")
	print "Generating embeddings for " + graphfname + "..."
	call(["cp", graphfname, "tmpi"])
	command = "./../../../snap-master/snap-master/examples/node2vec/node2vec"
	call([command, "-i:tmpi", "-o:tmp", "-v", "-p:1", "-e:10"]) 

	#read and cluster embeddings
	print "Clustering embeddings of " + graphfname + "..."
	vectors = []
	doc = open("tmp","r")
	for line in doc:
		a = line.split()
		tmp = []
		for l in a:
			tmp.append(float(l))
		vectors.append(tmp)
	doc.close()
	del vectors[0] #remove summary line
	ordered = sorted(vectors, key=lambda x: x[0])
	for o in ordered:
		del o[0] #remove node id
	km = KMeans(n_clusters=noof_comm).fit(ordered)

	#evaluating
	comm_labels = []
	for k in sorted(communities.keys()):
		comm_labels.append(communities[k])
	return normalized_mutual_info_score(comm_labels, km.labels_)


def main():
	if len(sys.argv) < 2:
		print "Please provide directory of graphs... Exiting..."
		return
	op = open("results.txt", "a")
	dirname = sys.argv[1]
	filenames = os.listdir(dirname)
	nmi = []
	for filename in filenames:
		if filename.startswith("community"):
			nmi.append(evaluate(dirname + "/" + filename))
	score = sum(nmi)/len(nmi)
	op.write(dirname + " " + str(score) + "\n")
	op.close()

main()
