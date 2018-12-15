Step 1. Start with the data used for the ANN problem (Node Flooding) from Homework #1.  This data is provided in the
	homework submission packet.

Step 2. Run the Python code provided in the code file for Homework #2.  This will create an output .csv file to be used
	within ABAGAIL for the purpose of leveraging randomized optimization algorithms to determine ANN weights.

Step 3. Move the output .csv file over to the appropriate directory in ABAGAIL.  Rename the file as "flood.csv".

Step 4. Update the "NeurNetTest.java" code provided in the code file appropriately so as to obtain the desired number of
	hidden layers and iterations.  Save the updated "NeurNetTest.java" file.

Step 5. Compile the updated "NeurNetTest.jave" file using Apache Ant.  Navigate the command prompt to the ABAGAIL directory
	containing the "build.xml" file and type "ant" at the command prompt and hit return.

Step 6. Run the "NeurNet.java" code by typing "java -cp ABAGAIL.jar opt.test.NeurNetTest" at the command prompt
	and hitting return.  Output provides predictive accuracies for each of RHC, SA, and GA algorithms.

Step 7. Repeat steps 4-6 as many times as desired using different hidden layers and iterations.  Capture results.

Step 8. For the Four Peaks problem:
	a. Update the "FourPeaksTest1.java" code provided in the code file appropriately so as to obtain the
		desired number of bit sizes (N) and iterations.  Save the updated "FourPeaksTest1.java" file.
	b. Compile the updated "FourPeaksTest1.jave" file using Apache Ant.  Navigate the command prompt to the
		ABAGAIL directory containing the "build.xml" file and type "ant" at the command prompt and hit return.
	c. Run the "FourPeaksTest1.java" code by typing "java -cp ABAGAIL.jar opt.test.FourPeaksTest1" at the
		command prompt and hitting return.  Output provides model performance for each of RHC, SA, GA, and MIMIC
		algorithms.  Looking to maximize model performance.
	d. Repeat steps a-c as many times as desired using different bit sizes (N) and iterations.  Capture results.

Step 9. For the Traveling Salesman problem:
	a. Update the "TravelingSalesmanTest1.java" code provided in the code file appropriately so as to obtain the
		desired number of cities (N) and iterations.  Save the updated "TravelingSalesmanTest1.java" file.
	b. Compile the updated "TravelingSalesmanTest1.jave" file using Apache Ant.  Navigate the command prompt to the
		ABAGAIL directory containing the "build.xml" file and type "ant" at the command prompt and hit return.
	c. Run the "TravelingSalesmanTest1.java" code by typing "java -cp ABAGAIL.jar opt.test.TravelingSalesmanTest1"
		at the command prompt and hitting return.  Output provides model performance for each of RHC, SA, GA,
		and MIMIC algorithms.  Looking to maximize model performance.
	d. Repeat steps a-c as many times as desired using different number of cities (N) and iterations.  Capture results.

Step 10. For the N Queens problem:
	a. Update the "NQueensTest1.java" code provided in the code file appropriately so as to obtain the desired
		number of Queens/board sizes (N) and iterations.  Save the updated "NQueensTest1.java" file.
	b. Compile the updated "NQueensTest1.jave" file using Apache Ant.  Navigate the command prompt to the ABAGAIL
		directory containing the "build.xml" file and type "ant" at the command prompt and hit return.
	c. Run the "NQueensTest1.java" code by typing "java -cp ABAGAIL.jar opt.test.NQueensTest1" at the command prompt
		and hitting return.  Output provides model performance for each of RHC, SA, GA, and MIMIC algorithms.
		Looking to maximize model performance.
	d. Repeat steps a-c as many times as desired using different number of Queens/board sizes (N) and iterations.
		Capture results.

The remaining material found below existed originally with the ABAGAIL repository that was cloned.

ABAGAIL
=======

[![Build Status](https://travis-ci.org/pushkar/ABAGAIL.svg?branch=master)](https://travis-ci.org/pushkar/ABAGAIL)

The library contains a number of interconnected Java packages that implement machine learning and artificial intelligence algorithms. These are artificial intelligence algorithms implemented for the kind of people that like to implement algorithms themselves.

Usage
------

* See [FAQ](https://github.com/pushkar/ABAGAIL/blob/master/faq.md)
* See [Wiki](https://github.com/pushkar/ABAGAIL/wiki)

Issues
-------

See [Issues page](https://github.com/pushkar/ABAGAIL/issues?state=open).

Contributing
------------

1. Fork it.
2. Create a branch (`git checkout -b my_branch`)
3. Commit your changes (`git commit -am "Awesome feature"`)
4. Push to the branch (`git push origin my_branch`)
5. Open a [Pull Request][1]
6. Enjoy a refreshing Diet Coke and wait 

Features
========

### Hidden Markov Models

* Baum-Welch reestimation algorithm, scaled forward-backward algorithm, Viterbi algorithm
* Support for Input-Output Hidden Markov Models
* Write your own output or transition probability distribution or use the provided distributions, including neural network based conditional probability distributions
* Neural Networks

### Feed-forward backpropagation neural networks of arbitrary topology
* Configurable error functions with sum of squares, weighted sum of squares
* Multiple activation functions with logistic sigmoid, linear, tanh, and soft max
* Choose your weight update rule with standard update rule, standard update rule with momentum, Quickprop, RPROP
* Online and batch training
* Support Vector Machines

### Fast training with the sequential minimal optimization algorithm
* Support for linear, polynomial, tanh, radial basis function kernels
* Decision Trees

### Information gain or GINI index split criteria
* Binary or all attribute value splitting
* Chi-square signifigance test pruning with configurable confidence levels
* Boosted decision stumps with AdaBoost
* K Nearest Neighbors

### Fast kd-tree implementation for instance based algorithms of all kinds
* KNN Classifier with weighted or non-weighted classification, customizable distance function
* Linear Algebra Algorithms

### Basic matrix and vector math, a variety of matrix decompositions based on the standard algorithms
* Solve square systems, upper triangular systems, lower triangular systems, least squares
* Singular Value Decomposition, QR Decomposition, LU Decomposition, Schur Decomposition, Symmetric Eigenvalue Decomposition, Cholesky Factorization
* Make your own matrix decomposition with the easy to use Householder Reflection and Givens Rotation classes
* Optimization Algorithms

### Randomized hill climbing, simulated annealing, genetic algorithms, and discrete dependency tree MIMIC
* Make your own crossover functions, mutation functions, neighbor functions, probability distributions, or use the provided ones.
* Optimize the weights of neural networks and solve travelling salesman problems
* Graph Algorithms

### Kruskals MST and DFS
* Clustering Algorithms

### EM with gaussian mixtures, K-means
* Data Preprocessing

### PCA, ICA, LDA, Randomized Projections
* Convert from continuous to discrete, discrete to binary
* Reinforcement Learning

### Value and policy iteration for Markov decision processes

[1]: https://help.github.com/articles/using-pull-requests
