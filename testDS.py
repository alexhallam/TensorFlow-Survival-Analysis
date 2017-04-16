'''
R unoriginal deepsurve package
'''

import deepsurv
from deepsurv import DeepSurv
import numpy

def generate_data(treatment_group = False):
	numpy.random.seed(123)
	sd = deepsurv.datasets.SimulatedData(5, num_features = 9,
		treatment_group = treatment_group)
	train_data = sd.generate_data(5000)
	valid_data = sd.generate_data(2000)
	test_data = sd.generate_data(2000)
	return train_data, valid_data, test_data

hyperparams = {
			'n_in': 10,
			'learning_rate': 1e-5,
			'hidden_layers_sizes': [10,10]
		}

train, valid, test = generate_data(treatment_group=True)
network = deepsurv.DeepSurv(**hyperparams)
network.restored_update_params = False
log = network.train(train, valid, n_epochs=500)