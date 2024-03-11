import jax
import jax.numpy as jnp
import equinox as eqx
import pickle

from .data_utilities import (
    embed_data_in_encoder_decoder_subspaces,
    load_data_from_disk,
	permute_arrays,
	slice_data,
    split_training_testing_data)

from .metrics import (
				grad_mean_h1_norm_loss_fn,
				grad_mean_l2_norm_loss_fn
				mean_h1_seminorm_l2_errors_and_norms=) 

def train_nn_regressor(*,
	untrained_regressor,
	training_data,
	testing_data,
	permute_key,
	training_config_dict,
	logger)
	#DOCUMENT ME, remainder not allowed

	####################################################################################
	# Create variable aliases for readability
	####################################################################################
	nn = untrained_regressor
	n_epochs = training_config_dict['optax_epochs']
	batch_size = training_config_dict['batch_size']
	loss_norm_weights = training_config_dict['loss_weights']
	shuffle_every_epoch = training_config_dict['shuffle_every_epoch']
	####################################################################################
	# Setup the optimization step (H1 Mean Error or MSE)
	####################################################################################
	if loss_norm_weights[1] == 0.0:
		grad_loss_fn = grad_mean_l2_norm_loss_fn
	else:
		grad_loss_fn = grad_mean_h1_norm_loss_fn

	@eqx.filter_jit
	def take_step( optimizer_state, nn, X, Y, dYdX ):
		#DOCUMENT ME
		updates, optimizer_state = optimizer.update(
			grad_loss_fn(nn, 
						X, Y, dYdX),
			optimizer_state)
		return optimizer_state, eqx.apply_updates(nn, updates)
		
	####################################################################################
	# Check if batch_size evenly divides the testing and training data
	####################################################################################
	n_train = training_data[0].shape[0]
	n_test = testing_data[0].shape[0]
	n_train_batches, remainder = divmod(n_train, batch_size)
	n_test_batches, remainder_test = divmod(n_test, batch_size)

	if remainder != 0 :
		raise(f"Warning, the number of training data ({n_train}) does not evenly devide into n_batches={n_train_batches}. Adjust `batch_size` to evenly divide {n_train}.")
	if remainder_test != 0 :
		raise(f"Warning, the number of testing data ({n_test}) does not evenly devide into n_batches={n_test_batches}. Adjust `batch_size` to evenly divide {n_test}.")

	####################################################################################
	# Setup, instantiate and initialize the optax optimizer 						   #
	####################################################################################
	if False:
		# LR scheduling does not seem to help in this case, but this is how you do it. 
		train_steps_per_epoch = int(training_config_dict['train_data_size']/ batch_size )
		num_train_steps = n_epochs*train_steps_per_epoch

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
	r_logger
	metrics_history = {'train_loss': [],
						'train_accuracy_l2': [],
						'train_accuracy_h1': [],
						'test_loss': [],
						'test_accuracy_l2': [],
						'test_accuracy_h1': [],
						'epoch_time': [],
						'epoch': []}

	####################################################################################
	# Train the neural network 
	####################################################################################
	for epoch in np.arange(1, n_epochs+1):
		if shuffle_every_epoch:
			permute_key, permuted_training_data = \
				permute_arrays(training_data, batch_size, key=permute_key) #, Y_norms_data, dYdX_norms_data
		else:
			permuted_training_data = training_data
		X, Y, dYdX, _, _ = permuted_training_data
		# start_time = time.time()
		end_idx = 0
		for _ in range(n_train_batches):
			end_idx, X_batch, Y_batch, dYdX_batch = \
				slice_data(X, Y, dYdX,  batch_size, end_idx)
			optimizer_state, nn  = \
				take_step(optimizer_state, nn, X_batch, Y_batch, dYdX_batch) #loss,
			# could combin Y and dYdX (flattened) into Y_dYdX_data and take the L2 norm that way (but then you would need to scale up one of them but the dimension), this woudl probably speed things up
		# epoch_time = time.time() - start_time

		
		# print(epoch) #, "loss", loss)#, loss_fn(nn, X_batch, Y_batch, dYdX_batch))

		# print('The last epoch took', epoch_time, 's')

		# metric_t0 = time.time()

		# Post-process and compute metrics after each epoch
		training_data_metrics = \
			compute_loss_metrics(nn, *permuted_training_data, n_train_batches)

		permute_key, permuted_test_data = permute_arrays(testing_data, key=permute_key)
		compute_loss_metrics(nn, *permuted_test_data, n_test_batches)
		
		#testing_data_metrics
		# metrics_train = compute_metrics(train_state,params,training_data)
		# metrics_test = compute_metrics(train_state,params,testing_data)
		metric_time = time.time() - metric_t0

		metrics_history['train_loss'].append_idx(np.array(training_data_metrics['loss']))
		metrics_history['train_accuracy_l2'].append_idx(np.array(training_data_metrics['acc_l2']))
		metrics_history['train_accuracy_h1'].append_idx(np.array(training_data_metrics['acc_h1']))
		metrics_history['test_loss'].append_idx(np.array(test_data_metrics['loss']))
		metrics_history['test_accuracy_l2'].append_idx(np.array(test_data_metrics['acc_l2']))
		metrics_history['test_accuracy_h1'].append_idx(np.array(test_data_metrics['acc_h1']))
		metrics_history['epoch_time'].append_idx(epoch_time)
		metrics_history['epoch'].append_idx(epoch)


		print(f"train epoch: {epoch}, "
					f"loss: {metrics_history['train_loss'][-1]}, "
					f"accuracy l2 : {metrics_history['train_accuracy_l2'][-1] * 100}, "
					f"accuracy h1 : {metrics_history['train_accuracy_h1'][-1] * 100}")
		print(f"test epoch: {epoch}, "
					f"loss: {metrics_history['test_loss'][-1]}, "
					f"accuracy l2: {metrics_history['test_accuracy_l2'][-1] * 100}, "
					f"accuracy h1: {metrics_history['test_accuracy_h1'][-1] * 100}")
		print('Max test accuracy L2 = ',100*np.max(np.array(metrics_history['test_accuracy_l2'])))
		print('Max test accuracy H1 (semi-norm) = ',100*np.max(np.array(metrics_history['test_accuracy_h1'])))
		print('The metrics took', metric_time, 's')

def train_dino_in_embedding_space(embedded_training_config_dict)
	config_dict = embedded_training_config_dict

	#################################################################################
	# Load training/testing data from disk, directly onto GPU as jax arrays  		#
	#################################################################################
	data = load_data_from_disk(config_dict['data']) # Involves Disk I/O

	#################################################################################
	# Embed training data in subspace using encoder/decoder bases/cobases           #
	#################################################################################
	encoder_decoder_dict = config_dict['encoder_decoder']
	if encoder_decoder_dict['encode'] or encoder_decoder_dict['decode']:
        # Involves Disk I/O
		data = embed_data_in_encoder_decoder_subspaces(
			data,
			encoder_decoder_dict)
		# config_dict['encoder_decoder']['last_layer_bias'] = np.mean(training_data['q_data'],axis = 0)
			
	#################################################################################
	# Split the data into training/testing data 	                                #
	#################################################################################
	training_data, testing_data = split_training_testing_data(data, config_dict['data'])
	
	#################################################################################
	# Set up the neural network and train it										#
	#################################################################################
	training_results = {}
	nn_config_dict = config_dict['nn']
	nn_config_dict['input_size']  = training_data.X.shape[1]
	nn_config_dict['output_size'] = training_data.Y.shape[1]
	untrained_regressor, permute_key = \
		instantiate_nn(nn_config_dict = nn_config_dict)
	trained_regressor = \
		train_nn_regressor( 
			untrained_regressor=untrained_regressor,
			training_data =training_data,
			testing_data =testing_data,
			permute_key=permute_key,
			training_config_dict = config_dict['training'],
			results_dict = training_results)

	#################################################################################
	# Save training metrics results to disk							                #
	#################################################################################
	# logger = {'reduced':training_logger} #,'full': final_logger}
	logging_dir = 'logging/'
    # Involves Disk I/O
	os.makedirs(logging_dir, exist_ok = True)
	with open(logging_dir+config_dict['network_serialization']['network_name'] +'.pkl', 'wb+') as f:
        pickle.dump(training_results, f, pickle.HIGHEST_PROTOCOL)

	#################################################################################
	# Save neural network parameters to disk (serialize the equinox pytrees)        #
	#################################################################################
    # Involves Disk I/O
	eqx.tree_serialise_leaves(
		f"{config_dict['network_serialization']['network_name'] }.eqx",
		trained_regressor)
		
	#################################################################################
	# Save config file for reproducibility                                          #
	#################################################################################	
    cli_dir = 'cli/'
    # Involves Disk I/O
	os.makedirs(cli_dir, exist_ok = True)
	#does the name tell you everything? we won't vary other parameters, yea?
	with open(cli_dir+config_dict['network_serialization']['network_name'] +'.pkl', 'wb+') as f:
        pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)