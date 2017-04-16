'''
Run the tf version and check output folder
'''

import deepsurv_tf
import numpy
import deepsurv

def generate_data(treatment_group = False):
	numpy.random.seed(123)
	sd = deepsurv.datasets.SimulatedData(5, num_features = 9,
		treatment_group = treatment_group)
	train_data = sd.generate_data(5000)
	valid_data = sd.generate_data(2000)
	test_data = sd.generate_data(2000)
	return train_data, valid_data, test_data

train, valid, test = generate_data(treatment_group=True)

params = deepsurv_tf.Parameters()
params.n_in = train['x'].shape[1]
params.n_epochs = 100

ds_tf = deepsurv_tf.DeepSurvTF(params)
ds_tf.train(train, valid)
ds_tf.plotSummary()