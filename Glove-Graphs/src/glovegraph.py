from argparse import ArgumentParser
import codecs
from collections import Counter
import itertools
from functools import partial
import logging
from math import log
import os.path
import cPickle as pickle
from random import shuffle

import msgpack
import numpy as np
from scipy import sparse


class NodeEmbedding():

	def __init__(self, xmax,size,alpha,learning_rate,dimensions,output_path,iterations):
		self.x_max = xmax
		self.size = size
		self.alpha = alpha
		self.learning_rate = learning_rate
		self.dimensions = dimensions
		self.output_path = output_path
		self.iterations = iterations

	def run_iter(self,data):
	    """
	    Run a single iteration of GloVe training using the given
	    cooccurrence data and the previously computed weight vectors /
	    biases and accompanying gradient histories.

	    `data` is a pre-fetched data / weights list where each element is of
	    the form

	        (v_main, v_context,
	         b_main, b_context,
	         gradsq_W_main, gradsq_W_context,
	         gradsq_b_main, gradsq_b_context,
	         cooccurrence)

	    as produced by the `train_glove` function. Each element in this
	    tuple is an `ndarray` view into the data structure which contains
	    it.

	    See the `train_glove` function for information on the shapes of `W`,
	    `biases`, `gradient_squared`, `gradient_squared_biases` and how they
	    should be initialized.

	    The parameters `x_max`, `alpha` define our weighting function when
	    computing the cost for two word pairs; see the GloVe paper for more
	    details.

	    Returns the cost associated with the given weight assignments and
	    updates the weights by online AdaGrad in place.
	    """

	    global_cost = 0

	    # We want to iterate over data randomly so as not to unintentionally
	    # bias the word vector contents
	    shuffle(data)
	    #i = 0
	    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
	         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

	        weight = (cooccurrence / self.x_max) ** self.alpha if cooccurrence < self.x_max else 1

	        # Compute inner component of cost function, which is used in
	        # both overall cost calculation and in gradient calculation
	        #
	        #   $$ J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
	        #print i,b_main,b_context
	        #i+=1
	        cost_inner = (v_main.dot(v_context)
	                      + b_main[0] + b_context[0]
	                      - log(cooccurrence))

	        # Compute cost
	        #
	        #   $$ J = f(X_{ij}) (J')^2 $$
	        cost = weight * (cost_inner ** 2)

	        # Add weighted cost to the global cost tracker
	        global_cost += 0.5 * cost

	        # Compute gradients for word vector terms.
	        #
	        # NB: `main_word` is only a view into `W` (not a copy), so our
	        # modifications here will affect the global weight matrix;
	        # likewise for context_word, biases, etc.
	        grad_main = weight * cost_inner * v_context
	        grad_context = weight * cost_inner * v_main

	        # Compute gradients for bias terms
	        grad_bias_main = weight * cost_inner
	        grad_bias_context = weight * cost_inner

	        # Now perform adaptive updates
	        v_main -= (self.learning_rate * grad_main / np.sqrt(gradsq_W_main))
	        v_context -= (self.learning_rate * grad_context / np.sqrt(gradsq_W_context))

	        b_main -= (self.learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
	        b_context -= (self.learning_rate * grad_bias_context / np.sqrt(
	                gradsq_b_context))

	        # Update squared gradient sums
	        gradsq_W_main += np.square(grad_main)
	        gradsq_W_context += np.square(grad_context)
	        gradsq_b_main += grad_bias_main ** 2
	        gradsq_b_context += grad_bias_context ** 2

	    return global_cost


	def train_glove(self,cooccurrences):
		"""
		Train GloVe vectors on the given generator `cooccurrences`, where
		each element is of the form

		    (word_i_id, word_j_id, x_ij)

		where `x_ij` is a cooccurrence value $X_{ij}$ as presented in the
		matrix defined by `build_cooccur` and the Pennington et al. (2014)
		paper itself.


		Returns the computed word vector matrix `W`.
		"""

		vocab_size = self.size
		vector_size = self.dimensions

		# Word vector matrix. This matrix is (2V) * d, where N is the size
		# of the corpus vocabulary and d is the dimensionality of the word
		# vectors. All elements are initialized randomly in the range (-0.5,
		# 0.5]. We build two word vectors for each word: one for the word as
		# the main (center) word and one for the word as a context word.
		#
		# It is up to the client to decide what to do with the resulting two
		# vectors. Pennington et al. (2014) suggest adding or averaging the
		# two for each word, or discarding the context vectors.
		W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / float(vector_size + 1)

		# Bias terms, each associated with a single vector. An array of size
		# $2V$, initialized randomly in the range (-0.5, 0.5].
		biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

		# Training is done via adaptive gradient descent (AdaGrad). To make
		# this work we need to store the sum of squares of all previous
		# gradients.
		#
		# Like `W`, this matrix is (2V) * d.
		#
		# Initialize all squared gradient sums to 1 so that our initial
		# adaptive learning rate is simply the global learning rate.
		gradient_squared = np.ones((vocab_size * 2, vector_size),
		                           dtype=np.float64)

		# Sum of squared gradients for the bias terms.
		gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

		# Build a reusable list from the given cooccurrence generator,
		# pre-fetching all necessary data.
		#
		# NB: These are all views into the actual data matrices, so updates
		# to them will pass on to the real data structures
		#
		# (We even extract the single-element biases as slices so that we
		# can use them as views)
		data = [(W[i_main], W[i_context + vocab_size],
		         biases[i_main : i_main + 1],
		         biases[i_context + vocab_size : i_context + vocab_size + 1],
		         gradient_squared[i_main], gradient_squared[i_context + vocab_size],
		         gradient_squared_biases[i_main : i_main + 1],
		         gradient_squared_biases[i_context + vocab_size
		                                 : i_context + vocab_size + 1],
		         cooccurrence)
		        for i_main, i_context, cooccurrence in cooccurrences]

		for i in range(self.iterations):
		    print "\tBeginning iteration ", i
		    cost = self.run_iter(data)
		    print "\t\tDone - cost : ",cost
		    if i%10 == 0:
		    	save_model(W,self.output_path)

		save_model(W,self.output_path)
		return W



def save_model(W, path):
    with open(path, 'wb') as vector_f:
        pickle.dump(W, vector_f, protocol=2)
