import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pickle
import time
from os import makedirs

from .data_utilities import (
	create_array_permuter,
    load_data_disk_direct_to_gpu,
	save_to_pickle,
	slice_data,
    split_training_testing_data)
from .dino import instantiate_nn
from .embed_data import embed_data_in_encoder_decoder_subspaces
from .metrics import (
				create_grad_mean_h1_seminorm_loss,
				grad_mean_l2_norm_loss,
				compute_l2_loss_metrics,
				create_compute_h1_loss_metrics)


def train_nn_regressor(*,
	untrained_regressor,
	training_data,
	testing_data,
	permute_key,
	training_config_dict,
	training_results_dict):
	#DOCUMENT ME, remainder not allowed
	start_time = time.time()
	####################################################################################
	# Create variable aliases for readability
	####################################################################################
	nn = untrained_regressor
	n_epochs = training_config_dict['optax_epochs']
	batch_size = training_config_dict['batch_size']
	loss_norm_weights = training_config_dict['loss_weights']
	# shuffle_every_epoch = training_config_dict['shuffle_every_epoch']

	####################################################################################
	# Check if batch_size evenly divides the testing and training data
	####################################################################################
	n_train, dM = training_data[0].shape
	n_test = testing_data[0].shape[0]
	n_train_batches, remainder = divmod(n_train, batch_size)
	n_test_batches, remainder_test = divmod(n_test, batch_size)
	if remainder != 0 :
		raise(f"Warning, the number of training data ({n_train}) does not evenly devide into n_batches={n_train_batches}. Adjust `batch_size` to evenly divide {n_train}.")
	if remainder_test != 0 :
		raise(f"Warning, the number of testing data ({n_test}) does not evenly devide into n_batches={n_test_batches}. Adjust `batch_size` to evenly divide {n_test}.")
	permute_training_arrays = create_array_permuter(n_train)
	permute_testing_arrays = create_array_permuter(n_test)
	####################################################################################
	# Setup the optimization step (H1 Mean Error or MSE)
	####################################################################################
	if loss_norm_weights[1] == 0.0:
		grad_loss = grad_mean_l2_norm_loss
		compute_loss_metrics = compute_l2_loss_metrics
	else:
		#FUTURE TODO: #do this outside of here so that it doesnt rejit? if we have this in an outside loop
		grad_loss = create_grad_mean_h1_seminorm_loss(dM)
		compute_loss_metrics = create_compute_h1_loss_metrics(dM) 
	
	@eqx.filter_jit
	def take_step( optimizer_state, nn, X, Y, dYdX ):
		#DOCUMENT ME
		updates, optimizer_state = optimizer.update(
			grad_loss(nn, X, Y, dYdX),
			optimizer_state)
		return optimizer_state, eqx.apply_updates(nn, updates)
		
	####################################################################################
	# Setup, instantiate and initialize the optax optimizer 						   #
	####################################################################################
	if True:
		# LR scheduling does not seem to help in this case, but this is how you do it. 
		num_train_steps = n_epochs*n_train_batches
		print('Using piecewise lr schedule')
		lr_schedule = optax.piecewise_constant_schedule(
			init_value =training_config_dict['step_size'],
			boundaries_and_scales =
				{int(num_train_steps*0.95):0.1,
				int(num_train_steps*0.975):0.1})								
	else:
		lr_schedule = training_config_dict['step_size']
	
	optimizer_name = training_config_dict.get('optax_optimizer', 'adam')
	optimizer = optax.__getattribute__(optimizer_name)(learning_rate=lr_schedule)
	# Initialize optimizer with eqx state (its pytree of weights)
	optimizer_state = optimizer.init(eqx.filter(nn, eqx.is_inexact_array))

	####################################################################################
	# Setup data structures for storing training results
	####################################################################################
	# results_dict['train_loss': [],
	# 			'train_accuracy_l2': [],
	# 			'train_accuracy_h1': [],
	# 			'test_loss': [],
	# 			'test_accuracy_l2': [],
	# 			'test_accuracy_h1': [],
	# 			'epoch_time': [],
	# 			'epoch': []]

	####################################################################################
	# Train the neural network 
	###################################################################################	
	for epoch in jnp.arange(1, n_epochs+1):
		permute_key, permuted_training_data = permute_training_arrays(*training_data, permute_key)
		X, Y, dYdX, _, _ = permuted_training_data 

		start_time = time.time()
		end_idx = 0
		for _ in range(n_train_batches):
			end_idx, X_batch, Y_batch, dYdX_batch = \
				slice_data(X, Y, dYdX,  batch_size, end_idx)
			optimizer_state, nn  = \
				take_step(optimizer_state, nn, X_batch, Y_batch, dYdX_batch) #loss,
			# could combin Y and dYdX (flattened) into Y_dYdX_data and take the L2 norm that way (but then you would need to scale up one of them by the dimension), this would probably speed things up
		epoch_time = time.time() - start_time
		# print(epoch) #, "loss", loss)#, loss_fn(nn, X_batch, Y_batch, dYdX_batch))

		print('The last epoch took', epoch_time, 's')

		# metric_t0 = time.time()

		# Post-process and compute metrics after each epoch
		# training_data_metrics = \
		# 	compute_loss_metrics(nn, permuted_training_data, n_train_batches)

		# permute_key, permuted_test_data = permute_testing_arrays(testing_data, key=permute_key)
		# compute_loss_metrics(nn, permuted_test_data, n_test_batches)
		
		#testing_data_metrics
		# metrics_train = compute_metrics(train_state,params,training_data)
		# metrics_test = compute_metrics(train_state,params,testing_data)
		# metric_time = time.time() - metric_t0

		# metrics_history['train_loss'].append_idx(np.array(training_data_metrics['loss']))
		# metrics_history['train_accuracy_l2'].append_idx(np.array(training_data_metrics['acc_l2']))
		# metrics_history['train_accuracy_h1'].append_idx(np.array(training_data_metrics['acc_h1']))
		# metrics_history['test_loss'].append_idx(np.array(test_data_metrics['loss']))
		# metrics_history['test_accuracy_l2'].append_idx(np.array(test_data_metrics['acc_l2']))
		# metrics_history['test_accuracy_h1'].append_idx(np.array(test_data_metrics['acc_h1']))
		# metrics_history['epoch_time'].append_idx(epoch_time)
		# metrics_history['epoch'].append_idx(epoch)


		# print(f"train epoch: {epoch}, "
		# 			f"loss: {metrics_history['train_loss'][-1]}, "
		# 			f"accuracy l2 : {metrics_history['train_accuracy_l2'][-1] * 100}, "
		# 			f"accuracy h1 : {metrics_history['train_accuracy_h1'][-1] * 100}")
		# print(f"test epoch: {epoch}, "
		# 			f"loss: {metrics_history['test_loss'][-1]}, "
		# 			f"accuracy l2: {metrics_history['test_accuracy_l2'][-1] * 100}, "
		# 			f"accuracy h1: {metrics_history['test_accuracy_h1'][-1] * 100}")
		# print('Max test accuracy L2 = ',100*np.max(np.array(metrics_history['test_accuracy_l2'])))
		# print('Max test accuracy H1 (semi-norm) = ',100*np.max(np.array(metrics_history['test_accuracy_h1'])))
		# print('The metrics took', metric_time, 's')
	print("Total time", time.time() - start_time)
def train_dino_in_embedding_space(model_key, embedded_training_config_dict):
	#################################################################################
	# Create variable aliases for readability										#
	#################################################################################
	config_dict = embedded_training_config_dict

	#################################################################################
	# Load training/testing data from disk, directly onto GPU as jax arrays  		#
	#################################################################################
	config_dict['data']['N'] = 5000 #TEMPORARY HACK until i know what to do here
	data_config_dict = config_dict['data']
	data = load_data_disk_direct_to_gpu(data_config_dict) # Involves Disk I/O
	
	#################################################################################
	# Embed training data in subspace using encoder/decoder bases/cobases           #
	#################################################################################
	encodec_dict = config_dict['encoder_decoder']

	#if the loaded data is not 'reduced' and we want to encode/decode (i.e. use the active subspace)
	if (encodec_dict['encode'] or encodec_dict['decode']) and \
	   data_config_dict.get('reduced_data_filenames') is None:
        # Involves Disk I/O
		data = embed_data_in_encoder_decoder_subspaces(
			data,
			encodec_dict)
		# config_dict['encoder_decoder']['last_layer_bias'] = np.mean(training_data['q_data'],axis = 0)
			
	#################################################################################
	# Split the data into training/testing data 	                                #
	#################################################################################
	training_data, testing_data = split_training_testing_data(data, config_dict['data'])
	
	#################################################################################
	# Set up the neural network and train it										#
	#################################################################################
	training_results_dict = {}
	nn_config_dict = config_dict['nn']
	nn_config_dict['input_size']  = training_data[0].shape[1]
	nn_config_dict['output_size'] = training_data[1].shape[1]
	untrained_regressor, permute_key = \
		instantiate_nn(key=model_key, nn_config_dict = nn_config_dict)
	trained_regressor = \
		train_nn_regressor( 
			untrained_regressor=untrained_regressor,
			training_data=training_data,
			testing_data =testing_data,
			permute_key=permute_key,
			training_config_dict = config_dict['training'],
			training_results_dict = training_results_dict)

	#################################################################################
	# Save training metrics results to disk							                #
	#################################################################################
	network_serialization_config_dict = config_dict['network_serialization']
	save_name = network_serialization_config_dict['network_name']
	# logger = {'reduced':training_logger} #,'full': final_logger}
	
	# save_to_pickle(logging_dir, save_name, training_results_dict)
	logging_dir = 'logging/'
	# Involves Disk I/O
	makedirs(logging_dir, exist_ok = True)
	with open(logging_dir+save_name +'.pkl', 'wb+') as file:
		pickle.dump(training_results_dict, file, pickle.HIGHEST_PROTOCOL)

	#################################################################################
	# Save neural network parameters to disk (serialize the equinox pytrees)        #
	#################################################################################
    # Involves Disk I/O
	if network_serialization_config_dict['save_weights']:
		eqx.tree_serialise_leaves(
			f"{network_serialization_config_dict['weights_dir']}{save_name}.eqx",
			trained_regressor)
		
	#################################################################################
	# Save config file for reproducibility                                          #
	#################################################################################	
	cli_dir = 'cli/'
	# save_to_pickle(cli_dir, save_name, config_dict)

	# Involves Disk I/O
	makedirs(cli_dir, exist_ok = True)
	#does the name tell you everything? we won't vary other parameters, yea?
	with open(cli_dir+save_name +'.pkl', 'wb+') as file:
		pickle.dump(config_dict, file, pickle.HIGHEST_PROTOCOL)
