from __future__ import print_function
import tensorflow as tf
import os
import logging
from lifelines.utils import concordance_index
import numpy
import matplotlib.pyplot as plt
import pdb


class Parameters(object):
	# __slots__ = ["n_in","learning_rate","hidden_layers_sizes","lr_decay","momentum",
	#           "L2_reg","L1_reg","activation","dropout","batch_norm","standardize",
	#           "n_epochs", "batch_norm_epsilon", "modelPath", "patience",
	#           "improvement_threshold","patience_increase","summaryPlots"]

	def __init__(self):
		self.n_in = None
		self.learning_rate = 0.00001
		self.hidden_layers_sizes = [10,10]
		self.lr_decay = 0.0
		self.momentum = 0.9
		self.L2_reg = 0.001
		self.L1_reg = 0.0
		self.activation = tf.nn.relu
		self.dropout = None
		self.batch_norm = False
		self.standardize = False
		self.batch_norm_epsilon = 0.00001 ## numerical stability

		##training params
		self.n_epochs = 500 ## no batches, only epochs since loss requires complete data to calculate
		self.modelPath = "out/learned.model" ## path to save the model, so that it can be restored later for use
		self.patience = 1000
		self.improvement_threshold = 0.99999
		self.patience_increase = 2

		##
		self.summaryPlots = None
		# self.summaryPlots = "out/summaryPlots"

class DeepSurvTF(object):
	def __init__(self, params):
		self.params = params
		x = tf.placeholder(dtype = tf.float32, shape = [None, self.params.n_in])
		e = tf.placeholder(dtype = tf.float32)
		
		assert (self.params.hidden_layers_sizes is not None \
			and type(self.params.hidden_layers_sizes) == list), \
			"invalid hidden layers type"
		assert self.params.n_in

		weightsList = [] ## for regularisation

		## to see training and validation performance
		self.trainingStats = {}

		out = x
		in_size = self.params.n_in


		for i in self.params.hidden_layers_sizes:
			weights = tf.Variable(tf.truncated_normal((in_size, i)),dtype = tf.float32)
			weightsList.append(weights)

			out = tf.matmul(out, weights)

			if self.params.batch_norm: ##TODO : check if ewma needs to be there for non CNN type layers
				batch_mean1, batch_var1 = tf.nn.moments(out,[0])
				out_hat = (out - batch_mean1) / tf.sqrt(batch_var1 + self.params.batch_norm_epsilon)
				scale = tf.Variable(tf.ones(i))
				beta = tf.Variable(tf.zeros(i))
				out = scale * out_hat + beta
			else:
				bias = tf.Variable(tf.zeros(i), dtype = tf.float32)
				out = out + bias

			out = self.params.activation(out)
			if self.params.dropout is not None:
				out = tf.nn.dropout(out, keep_prob = 1-self.params.dropout)

			in_size = i

		##final output linear layer with single output
		weights = tf.Variable(tf.truncated_normal((in_size, 1)),dtype = tf.float32)
		bias = tf.Variable(tf.zeros(1), dtype = tf.float32)
		out = tf.matmul(out, weights) + bias

		##flattening
		out = tf.reshape(out, [-1])

		##loss
		##assuming the inputs are sorted reverse time
		hazard_ratio = tf.exp(out)
		log_risk = tf.log(tf.cumsum(hazard_ratio))
		uncensored_likelihood = out - log_risk
		censored_likelihood = uncensored_likelihood * e
		loss = -tf.reduce_sum(censored_likelihood)

		##regularisation is only on weights, not on biases
		##ideally do only 1 of l1+l2  or drop out
		if self.params.L1_reg> 0:
			for kk in weightsList:
				loss += self.params.L1_reg * tf.reduce_sum(tf.abs(kk))

		if self.params.L2_reg> 0:
			for kk in weightsList:
				loss += self.params.L2_reg * tf.nn.l2_loss(kk)
		
		##optimiser
		##momentum with decay
		global_step = tf.Variable(0, trainable=False)
		learning_rate = tf.train.inverse_time_decay(
			learning_rate = self.params.learning_rate, 
			global_step = global_step,
			decay_steps = 1, 
			decay_rate = self.params.lr_decay, 
		)
		grad_step = (
    		tf.train.MomentumOptimizer(learning_rate, momentum = self.params.momentum, use_nesterov =True)
    		.minimize(loss, global_step=global_step)
		)

		##Adam optimiser
		# grad_step = tf.train.AdgradOptimizer(learning_rate = self.params.learning_rate)\
					# .minimize(loss)

		##gradient descent
		# grad_step = tf.train.GradientDescentOptimizer(learning_rate = self.params.learning_rate)\
					# .minimize(loss)

		##input handles
		self.x = x
		self.e = e

		##metrics to retrieve later
		self.risk = out
		self.grad_step = grad_step
		self.loss = loss
			
	def train(self, trainingData, validationData = None, validation_freq = 10): 
		#tdata required to sort data only
		## sort data
		xdata, edata, tdata = trainingData['x'], trainingData['e'], trainingData['t']
		sort_idx = numpy.argsort(tdata)[::-1]
		xdata = xdata[sort_idx]
		edata = edata[sort_idx].astype(numpy.float32)
		tdata = tdata[sort_idx]

		if validationData:
			xdata_valid, edata_valid, tdata_valid = validationData['x'], validationData['e'], validationData['t']
			sort_idx = numpy.argsort(tdata_valid)[::-1]
			xdata_valid = xdata_valid[sort_idx]
			edata_valid = edata_valid[sort_idx].astype(numpy.float32)
			tdata_valid = tdata_valid[sort_idx]

		##TODO : cache
		if self.params.standardize:
			mean, var = xdata.mean(axis=0), xdata.std(axis =0)
			xdata = (xdata - mean) / var
			##same mean and var as train
			xdata_valid = (xdata_valid - mean) / var

		assert self.params.modelPath
		assert xdata.shape[1] == self.params.n_in, "invalid number of covariates"
		assert (edata.ndim == 1)  and (tdata.ndim == 1) ##sanity check

		train_losses, train_ci, train_index = [], [], []
		validation_losses, validation_ci, validation_index = [], [], []

		best_validation_loss = numpy.inf
		best_params_idx = -1

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer()) ##init graph with given initializers
			##start training
			for epoch in range(self.params.n_epochs):
				loss, risk, _ = sess.run(
					[self.loss, self.risk, self.grad_step], 
					feed_dict = {
						self.x : xdata,
						self.e : edata
					})
				
				train_losses.append(loss)
				train_ci.append(concordance_index(tdata, -numpy.exp(risk.ravel()), edata))
				train_index.append(epoch)

				##frequently check metrics on validation data
				if validationData and (epoch % validation_freq == 0):
					vloss, vrisk = sess.run(
						[self.loss, self.risk], 
						feed_dict = {
							self.x : xdata_valid,
							self.e : edata_valid
						})
					
					validation_losses.append(vloss)
					validation_ci.append(concordance_index(tdata_valid, -numpy.exp(vrisk.ravel()), edata_valid))                 
					validation_index.append(epoch)

					# improve patience if loss improves enough
					if vloss < best_validation_loss * self.params.improvement_threshold:
						self.params.patience = max(self.params.patience, epoch * self.params.patience_increase)

						best_params_idx = epoch
						best_validation_loss = vloss

				if self.params.patience <= epoch:
					break

			print("Training done")
			print("Best epoch", best_params_idx)
			print("Best loss", best_validation_loss)

			##save model
			saver = tf.train.Saver()
			saver.save(sess, self.params.modelPath)

			self.trainingStats["training"] = {
				"loss" : train_losses,
				"ci" : train_ci,
				"epochs" : train_index,
				"type" : "training"
			}

			if validationData:
				self.trainingStats["validation"] = {
					"loss" : validation_losses,
					"ci" : validation_ci,
					"epochs" : validation_index,
					"type" : "validation"
				}

			return self.trainingStats

	def plotSummary(self):
		validationData = 1 if "validation" in self.trainingStats else 0
		#########################################
		##plot losses
		fig, [ax1, ax2] = plt.subplots(figsize = (15,6), nrows=1, ncols=2 )  # create figure & 1 axis
		##losses of train and validation
		ax1.plot(self.trainingStats["training"]["epochs"], self.trainingStats["training"]["loss"], "ro")
		l1, = ax1.plot(self.trainingStats["training"]["epochs"], self.trainingStats["training"]["loss"], "r")
		if validationData:
			ax1.plot(self.trainingStats["validation"]["epochs"], self.trainingStats["validation"]["loss"], "bo")
			l2, = ax1.plot(self.trainingStats["validation"]["epochs"], self.trainingStats["validation"]["loss"], "b")
		ax1.set_xlabel("Epochs")
		ax1.set_ylabel("Loss")
		ax1.grid()

		##ci of train and validation
		ax2.plot(self.trainingStats["training"]["epochs"], self.trainingStats["training"]["ci"], "ro")
		ax2.plot(self.trainingStats["training"]["epochs"], self.trainingStats["training"]["ci"], "r")
		if validationData:
			ax2.plot(self.trainingStats["validation"]["epochs"], self.trainingStats["validation"]["ci"], "bo")
			ax2.plot(self.trainingStats["validation"]["epochs"], self.trainingStats["validation"]["ci"], "b")
		ax2.set_xlabel("Epochs")
		ax2.set_ylabel("CI")
		ax2.grid()

		if validationData:
			fig.legend((l1, l2), ('Training', 'Validation'), 'upper left')

		if self.params.summaryPlots:
			fig.savefig(self.params.summaryPlots)   # save the figure to file
			plt.close(fig)
		else:
			plt.show()

	def predict(self, testXdata):
		assert os.path.exists(self.params.modelPath)
		with tf.Session() as sess:
			saver = tf.train.Saver()
			saver.restore(sess, self.params.modelPath)
			print("model loaded")

			risk = sess.run([risk], feed_dict = {self.x : testXdata})

		assert risk.shape[1] == 1
		return risk.ravel()

	def get_concordance_index(self, xdata, edata, tdata):
		risk = self.predict(xdata)
		partial_hazards = -numpy.exp(risk)
		return concordance_index(tdata, partial_hazards, edata)

	def recommend_treatment(self, x, trt_i, trt_j, trt_idx = -1):
		# Copy x to prevent overwritting data
		x_trt = numpy.copy(x)

		# Calculate risk of observations treatment i
		x_trt[:,trt_idx] = trt_i
		h_i = self.predict(x_trt)
		# Risk of observations in treatment j
		x_trt[:,trt_idx] = trt_j;
		h_j = self.predict(x_trt)

		rec_ij = h_i - h_j
		return rec_ij

	#TODO : from deepsurv: plot risk surface, different optimisers (not necessary)