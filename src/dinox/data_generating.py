import time
from operator import itemgetter
from pathlib import Path
from typing import Iterable, List, Literal, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from bayesflux.encoding import encode_input_output_Jacobian_data
from bayesflux.generation import (
    GaussianInputOuputAndDerivativesSampler,
    generate_full_Jacobian_data,
    generate_output_data,
    generate_reduced_training_data,
)
from bayesflux.subspace_detection import (
    information_theoretic_dimension_reduction,
    moment_based_dimension_reduction,
)
from numpy.typing import ArrayLike

from .data_loading import device_get_pytree, dump_pytree_to_disk, load_pytree_from_disk


def __find_eigs_truncation_idxs(
    lambda_seq: List[float], mu_seq: List[float], percentage: float, threshold: float = 1.0
) -> Tuple[int, int]:
    """
    Find indices at which to truncate ordered eigenvalue sequences to capture a given
    percentage of the tail sum of each eigenvalue sequences below a specified threshold.

    Parameters:
        lambda_seq (List[float]):
            A list of eigenvalues (sorted in descending order).

        mu_seq (List[float]):
            Another list of eigenvalues (sorted in descending order).

        percentage (float):
            % of the sum of eigenvalues below the threshold we wish to capture.

        threshold (float, optional):
            The eigenvalue threshold defining the "tail" region (default is 1).
            Choose np.inf for all eigenvalues.

    Returns:
        Tuple[int, int]:
            A tuple containing the indices (r_lambda, r_mu) indicating how many
            eigenvalues below the threshold in each sequence should be included
            to reach at least the specified percentage of the tail sum.

    Notes:
        - The input sequences must be sorted in descending order.
        - The returned indices are counts of eigenvalues below the threshold included
          from the start of the tail region.
    """
    # Filter eigenvalues below the threshold
    lambda_tail = [val for val in lambda_seq if val < threshold]
    mu_tail = [val for val in mu_seq if val < threshold]

    # Calculate the total sum of eigenvalues in the tails
    total_tail_sum = sum(lambda_tail) + sum(mu_tail)
    target_sum = percentage * total_tail_sum

    # Initialize indices and running sum
    i, j = 0, 0
    curr_sum = 0.0

    # Accumulate eigenvalues from tails until the target sum is reached
    while curr_sum < target_sum:
        # Next available eigenvalue from lambda_tail or negative infinity if exhausted
        next_lambda = lambda_tail[i] if i < len(lambda_tail) else -float("inf")
        next_mu = mu_tail[j] if j < len(mu_tail) else -float("inf")

        # Select the larger available eigenvalue to accumulate more quickly
        if next_lambda >= next_mu:
            curr_sum += next_lambda
            i += 1
        else:
            curr_sum += next_mu
            j += 1
    return i, j


def generate_moment_based_encoder_decoders(
    key,
    input_covariance_matrix=None,
    L2_inner_product_matrix=None,
    output_samples: ArrayLike = None,
    input_output_dim_percentage_pairs: Iterable[float] = None,
    input_output_dim_pairs: Iterable[Tuple[int, int]] = None,
    input_dims: Iterable[int] = None,
    output_dims: Iterable[int] = None,  # only one of input_dims / output_dims should be provided
):
    if input_output_dim_percentage_pairs:
        max_input_dimension = input_covariance_matrix.shape[0]
        max_output_dimension = output_samples.shape[1]
    elif input_output_dim_pairs:
        max_input_dimension = max(input_output_dim_pairs, key=itemgetter(0))[0]
        max_output_dimension = max(input_output_dim_pairs, key=itemgetter(1))[1]
    elif input_dims:
        max_input_dimension = max(input_dims)
        max_output_dimension = None
        input_output_dim_pairs = [(input_dim, None) for input_dim in input_dims]
    elif output_dims:
        max_input_dimension = None
        max_output_dimension = max(output_dims)
        input_output_dim_pairs = [(None, output_dim) for output_dim in output_dims]
    else:
        raise Exception("No reduction dimensions provided!")
    # Compute the input/output encodec given maximum truncation dimensions
    full_encodecs = moment_based_dimension_reduction(
        key=key,
        output_samples=jax.device_put(output_samples) if output_samples is not None else output_samples,
        input_covariance_matrix=input_covariance_matrix,
        L2_inner_product_matrix=L2_inner_product_matrix,
        max_input_dimension=max_input_dimension,
        max_output_dimension=max_output_dimension,
    )
    if input_output_dim_percentage_pairs:
        input_output_dim_pairs = [
            __find_eigs_truncation_idxs(
                full_encodecs["input"]["eigenvalues"], full_encodecs["output"]["eigenvalues"], percentage
            )
            for percentage in input_output_dim_percentage_pairs
        ]
    encodecs = dict()
    for input_dim_r, output_dim_r in input_output_dim_pairs:
        encodecs[(input_dim_r, output_dim_r)] = dict()
        print("reduced dimensions", input_dim_r, output_dim_r)
        for k1, dim_r in zip(("input", "output"), (input_dim_r, output_dim_r)):
            encodecs[(input_dim_r, output_dim_r)][k1] = dict()
            if dim_r is not None:
                for k2 in ("eigenvalues", "encoder", "decoder"):
                    # print("shape before subsetting", encodecs[(input_dim_r, output_dim_r)][k1][k2].shape)
                    encodecs[(input_dim_r, output_dim_r)][k1][k2] = full_encodecs[k1][k2].copy()[..., :dim_r]
                    print(k1, k2, "reduced dimension", dim_r)
                    print(encodecs[(input_dim_r, output_dim_r)][k1][k2].shape, "\n\n")
    return encodecs  # needs to be saved to model's name


def generate_derivative_informed_encoder_decoders_from_full_jacobians(
    key,
    prior_precision,
    noise_variance,
    full_jacobian_data: ArrayLike,
    input_output_dim_percentage_pairs: Iterable[float] = None,
    input_output_dim_pairs: Iterable[Tuple[int, int]] = None,
    input_dims: Iterable[int] = None,
    output_dims: Iterable[int] = None,
    # prior,
):
    if input_output_dim_percentage_pairs:
        max_input_dimension = full_jacobian_data.shape[1]
        max_output_dimension = full_jacobian_data.shape[2]
    elif input_output_dim_pairs:
        max_input_dimension = max(input_output_dim_pairs, key=itemgetter(0))[0]
        max_output_dimension = max(input_output_dim_pairs, key=itemgetter(1))[1]
    elif input_dims:
        max_input_dimension = max(input_dims)
        max_output_dimension = None
        input_output_dim_pairs = [(input_dim, None) for input_dim in input_dims]
    elif output_dims:
        max_input_dimension = None
        max_output_dimension = max(output_dims)
        input_output_dim_pairs = [(None, output_dim) for output_dim in output_dims]
    else:
        raise Exception("No reduction dimensions provided!")
    print("max dimensions", max_input_dimension, max_output_dimension)
    # Compute the input/output encodec given maximum truncation dimensions
    full_encodecs = information_theoretic_dimension_reduction(
        key=key,
        J_samples=jax.device_put(full_jacobian_data),
        noise_variance=jax.device_put(noise_variance),
        prior_precision=jax.device_put(prior_precision),
        prior_covariance=jnp.linalg.inv(prior_precision),
        max_input_dimension=max_input_dimension,
        max_output_dimension=max_output_dimension,
    )
    if input_output_dim_percentage_pairs:
        input_output_dim_pairs = [
            __find_eigs_truncation_idxs(
                full_encodecs["input"]["eigenvalues"], full_encodecs["output"]["eigenvalues"], percentage
            )
            for percentage in input_output_dim_percentage_pairs
        ]
        # hp_eigs_input, hp_input_encoder, hp_input_decoder = hippylib_active_subspace(full_jacobian_data, noise_variance, prior, 2145, 10)

        # eigs_input = encoder_decoders_dict['input']['eigenvalues']
        # eigs_output = encoder_decoders_dict['output']['eigenvalues']
        # plt.semilogy(eigs_input[jnp.argsort(eigs_input)[::-1]])
        # plt.semilogy(hp_eigs_input[jnp.argsort(hp_eigs_input)[::-1]])
        # # plt.semilogy(eigs_input[jnp.argsort(eigs_input)[::-1]],label="estimated")

        # # plt.semilogy(hp_eigs_input[jnp.argsort(hp_eigs_input)[::-1]],label="true_hp")
        # plt.title("Eigenvalues of C E[J^T Gamma^{-1}_N J]")
        # # plt.legend()
        # plt.show()

        # plt.semilogy(eigs_output[jnp.argsort(eigs_output)[::-1]])
        # # plt.semilogy(eigs_input[jnp.argsort(eigs_input)[::-1]],label="estimated")

        # # plt.semilogy(hp_eigs_input[jnp.argsort(hp_eigs_input)[::-1]],label="true_hp")
        # plt.title("Eigenvalues of Gamma^{-1}_N E[J^T C_N J]")
        # # plt.legend()
        # plt.show()

        # # print(f"Use only {input_reduced_dim} eigenfunctions, percent of spectrum", 100*sum(eigs_input)/sum(true_eigs_input))
    encodecs = dict()
    for input_dim_r, output_dim_r in input_output_dim_pairs:
        encodecs[(input_dim_r, output_dim_r)] = dict()
        # print(f"Use only {input_reduced_dim} eigenfunctions, percent of TRUE input spectrum", 100*sum(hp_eigs_input[0:input_reduced_dim])/sum(hp_eigs_input))
        # # print(f"Use only {input_reduced_dim} eigenfunctions, percent of TRUE input+output spectrum", 100*sum(hp_eigs_input[0:input_reduced_dim])/(sum(true_eigs_output)+sum(hp_eigs_input)))

        # print("Trace of expected J^TGamma^-1 J", sum(hp_eigs_input))
        # print("Trace of intput tail sum", sum(hp_eigs_input[input_reduced_dim:]))

        # # print(f"Use only {output_reduced_dim} eigenfunctions, percent of spectrum", 100*sum(eigs_output)/sum(true_eigs_output)) #1,2 5, 50
        # print(f"Use only {output_reduced_dim} eigenfunctions, percent of TRUE output spectrum", 100*sum(eigs_output[0:output_reduced_dim])/sum(eigs_output)) #1,2 5, 50
        # print(f"Use only {input_reduced_dim, output_reduced_dim} eigenfunctions, percent of TRUE intput+output spectrum", (100*sum(eigs_output[0:output_reduced_dim])+ 100*sum(hp_eigs_input[0:input_reduced_dim]))/(sum(eigs_output)+sum(hp_eigs_input))) #1,2 5, 50

        # print("Trace of expected JCJ^T", sum(eigs_output))
        # print("Trace of output tail sum", sum(eigs_output[output_reduced_dim:]),"\n\n")
        print("reduced dimensions", input_dim_r, output_dim_r)
        for k1, dim_r in zip(("input", "output"), (input_dim_r, output_dim_r)):
            encodecs[(input_dim_r, output_dim_r)][k1] = dict()
            if dim_r is not None:
                for k2 in ("eigenvalues", "encoder", "decoder"):
                    # print("shape before subsetting", encodecs[(input_dim_r, output_dim_r)][k1][k2].shape)
                    encodecs[(input_dim_r, output_dim_r)][k1][k2] = full_encodecs[k1][k2].copy()[..., :dim_r]
                    print(k1, k2, "reduced dimension", dim_r)
                    print(encodecs[(input_dim_r, output_dim_r)][k1][k2].shape, "\n\n")
    return encodecs  # needs to be saved to model's name


def generate_latent_data_for_dino_training(
    model_obj: GaussianInputOuputAndDerivativesSampler,  # noise_variance?
    N_trains: Iterable[int],
    key: jax.Array,
    dimension_reduction_type: Literal["derivative_informed", "moment_based"],
    input_output_dim_percentage_pairs: Iterable[float] = None,
    input_output_dim_pairs: Iterable[Tuple[int, int]] = None,
    input_dims: Iterable[int] = None,
    output_dims: Iterable[int] = None,
    N_encodec_computation: int = 1000,
    N_test: int = 25_000,
    N_val: int = 2500,
    save_path: Path = None,
    reduce_input_before: bool = False,
):
    N_train_max = max(N_trains)
    N_reduced_samples = N_train_max + N_test + N_val  # + N_encodec_computation
    if dimension_reduction_type == "derivative_informed":
        print("Generating data for derivative informed subspace detection")
        subspace_detect_data = generate_full_Jacobian_data(
            sampler_wrapper=model_obj, N_samples=N_encodec_computation
        )  # model_obj PRNG key??
        # print("loaded_results",susbpace_detect_data)
        # input_output_reduced_dim_percentage_pairs  = [0.999, 0.99,0.95,0.9 ] #(2145, 10), (2145,25)
        print("Computing derivative informed encodecs")
        encodec_dict = generate_derivative_informed_encoder_decoders_from_full_jacobians(
            key=key,
            prior_precision=model_obj.precision,
            noise_variance=2e-3,
            full_jacobian_data=subspace_detect_data["Jacobians"],  # max_input_dimension=500, max_output_dimension=100)
            input_output_dim_percentage_pairs=input_output_dim_percentage_pairs,
            input_output_dim_pairs=input_output_dim_pairs,
            input_dims=input_dims,
            output_dims=output_dims,
        )
    elif dimension_reduction_type == "moment_based":
        print("Generating data for moment-based subspace detection")
        if not input_dims:
            subspace_detect_data = generate_output_data(
                sampler_wrapper=model_obj, N_samples=N_encodec_computation
            )  # model_obj PRNG key??
        encodec_dict = generate_moment_based_encoder_decoders(
            key=key,
            input_covariance_matrix=np.linalg.inv(model_obj.precision) if output_dims is None else None,
            L2_inner_product_matrix=model_obj.L2_inner_product_matrix if output_dims is None else None,
            output_samples=subspace_detect_data["outputs"] if input_dims is None else None,
            input_output_dim_percentage_pairs=input_output_dim_percentage_pairs,
            input_output_dim_pairs=input_output_dim_pairs,
            input_dims=input_dims,
            output_dims=output_dims,
        )

    if save_path:
        if not (input_dims and dimension_reduction_type == "moment_based"):
            print("Dumping this data to disk")
            Path(save_path, dimension_reduction_type).mkdir(parents=True, exist_ok=True)
            path = Path(save_path, dimension_reduction_type, "subpace_detect_data.hkl")
            dump_pytree_to_disk(subspace_detect_data, path)
            print(f"Checking saved correctly to {path}")
            susbpace_detect_data = load_pytree_from_disk(path)

        print("Dumping this data to disk")
        path = Path(save_path, dimension_reduction_type, "encodec_dict.hkl")  # include type of encodec
        dump_pytree_to_disk(encodec_dict, path)
        print(f"Checking saved correctly to {path}")
        encodec_dict = load_pytree_from_disk(path)

    if not (input_dims and dimension_reduction_type == "moment_based"):
        print("Encoding subspace detection training samples")
        reduced_subspace_detect_data_dicts = dict()
        for reduced_dims_tuple_i, encodec_dict_i in encodec_dict.items():
            print("...input/output reduced dimensions:", reduced_dims_tuple_i)
            reduced_subspace_detect_data_dicts[reduced_dims_tuple_i] = encode_input_output_Jacobian_data(
                inputs=subspace_detect_data["inputs"],
                outputs=subspace_detect_data["outputs"],
                jacobians=subspace_detect_data.get("Jacobians"),  # If Jacobians are not present, it still works
                input_encoder=encodec_dict_i["input"].get("encoder"),
                output_encoder=encodec_dict_i["output"].get("encoder"),
                input_decoder=encodec_dict_i["input"].get("decoder"),
            )
            print(
                "\tEncoding times (input, output, Jacobian): ",
                reduced_subspace_detect_data_dicts[reduced_dims_tuple_i].get(
                    "input_encoding_time", "No input dimension reduction"
                ),
                reduced_subspace_detect_data_dicts[reduced_dims_tuple_i].get(
                    "output_encoding_time", "No output dimension reduction"
                ),
                reduced_subspace_detect_data_dicts[reduced_dims_tuple_i].get(
                    "Jacobian_encoding_time", "No Jacobians to encode"
                ),
            )
            reduced_subspace_detect_data_dicts[reduced_dims_tuple_i]["encodec"] = encodec_dict_i

        if save_path:
            print("\tDumping this data to disk")
            path = Path(save_path, dimension_reduction_type, "subspace_detection_data_reduced_and_encoders.hkl")
            dump_pytree_to_disk(
                reduced_subspace_detect_data_dicts, path
            )  # name should reflect apparoach to dim reduction
            print(f"\tChecking saved correctly to {path}")
            reduced_subspace_detect_data_dicts = load_pytree_from_disk(path)

    reduced_all_data = dict()
    print("Getting encoders from gpu, placing back on cpu")
    encodec_dict_on_cpu = device_get_pytree(encodec_dict)
    print("Generating remaining training data in encodec latent space")
    for reduced_dims_tuple_i, encodec_dict_i in encodec_dict_on_cpu.items():
        print("...input/output reduced dimensions:", reduced_dims_tuple_i)
        reduced_all_data[reduced_dims_tuple_i] = generate_reduced_training_data(
            sampler_wrapper=model_obj,  # set the hippylib PRNG key!!!
            N_samples=N_reduced_samples,
            input_encoder=encodec_dict_i["input"].get("encoder"),
            output_encoder=encodec_dict_i["output"].get("encoder"),
            input_decoder=encodec_dict_i["input"].get("decoder"),
            reduce_input_before=reduce_input_before,
        )
        reduced_all_data[reduced_dims_tuple_i]["encodec"] = encodec_dict_i

        if dimension_reduction_type == "derivative_informed":
            print("\tjacobian encoding time", reduced_all_data[reduced_dims_tuple_i]["Jacobian_encoding_time"])
            if encodec_dict_i["input"].get("decoder"):
                print("\tjacobian decoding time", reduced_all_data[reduced_dims_tuple_i]["Jacobian_decoding_time"])

    if save_path:
        print("\tDumping latent train, test, val data to disk; returning test_train_val dictionary.")
    reduced_train_test_val = dict()
    val_start_idx = N_train_max + N_test
    for (
        reduced_dims_tuple_i,
        reduced_data_i,
    ) in reduced_all_data.items():
        reduced_train_test_val[reduced_dims_tuple_i] = {
            "train": dict(),
            "test": dict(),
            "val": dict(),
        }

        for data_key in ["encoded_Jacobians", "encoded_inputs", "encoded_outputs"]:
            data_values = reduced_data_i.get(data_key)
            if data_values is not None:
                reduced_train_test_val[reduced_dims_tuple_i]["train"][data_key] = data_values[:N_train_max]
                reduced_train_test_val[reduced_dims_tuple_i]["test"][data_key] = data_values[N_train_max:val_start_idx]
                reduced_train_test_val[reduced_dims_tuple_i]["val"][data_key] = data_values[
                    val_start_idx : val_start_idx + N_val
                ]

                if save_path:
                    Path(save_path, dimension_reduction_type, "training").mkdir(parents=True, exist_ok=True)
                    Path(save_path, dimension_reduction_type, "testing").mkdir(parents=True, exist_ok=True)
                    Path(save_path, dimension_reduction_type, "validation").mkdir(parents=True, exist_ok=True)

                    # saves different sets of training data for easy loading from file
                    for N_train in N_trains:
                        jnp.save(
                            Path(
                                save_path,
                                dimension_reduction_type,
                                "training",
                                f"{reduced_dims_tuple_i}_{data_key}_{N_train}.npy",
                            ),
                            data_values[0:N_train],
                        )
                    jnp.save(
                        Path(save_path, dimension_reduction_type, "testing", f"{reduced_dims_tuple_i}_{data_key}.npy"),
                        data_values[N_train_max:val_start_idx],
                    )
                    jnp.save(
                        Path(
                            save_path, dimension_reduction_type, "validation", f"{reduced_dims_tuple_i}_{data_key}.npy"
                        ),
                        data_values[val_start_idx : val_start_idx + N_val],
                    )

    return reduced_train_test_val  # keeps all N_train_max data in the training data set
