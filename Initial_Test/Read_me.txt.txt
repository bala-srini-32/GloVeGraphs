Instructions on Running Glove and node2vec

1) Node2vec

Run main.py with the parameters required by the argparser. Example command:

>python main.py --p 1 --q 0.5 --iter 1000 --input Fortunato_Synthetic/1000_nodes_gamma_2_beta_1_network.dat --output emb/node2vec_Fortunato.emb --dimensions 64

You may choose to update the walk length and number of random walks as well for experiments!


2) Glove

Run main.py with same parameters as node2vec. This generates a co-occurance file which is needed as input for glove.py

Run glove.py from command line with parameters described in requirements of argparser in code

Example command: 

python glove.py --cooccur-path Co_occurance_matrices/Fortunato-cooccur --size 32 --vector-path emb/Fortunato_glove_1.emb --iter 1000 --save-often --size 1000

The hyperparameter size corresponds to the number of nodes in the graph. Please update according to number of nodes in the graph you want to run. 


Do let me know if there are questions!

Sudarshan