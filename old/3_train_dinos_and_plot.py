# THIS and all data generation/model problem codes should go in a common "problems" subrepo of DinoSciML i think. and people can refactor their codes
# once we come to a consensus of the easiest format for problems.
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"

import argparse
from datetime import datetime

today = datetime.now()  # current date and time

parser = argparse.ArgumentParser()
parser.add_argument(
    "-save_base_folder",
    "--save_base_folder",
    type=str,
    default="/storage/joshua/",
    help="Folder to save everything in",
)
parser.add_argument(
    "-problem_name",
    "--problem_name",
    type=str,
    default="hyperelasticity",  # "nonlinear_diffusion_reaction",#"hyperelasticity",
    help="Problem name",
)

parser.add_argument(
    "-loss",
    "--loss",
    type=str,
    default="h1",  #'l2,meanVar, meanEnt, hybrid, h1, fisher
    help="loss function",
)

todays_date = today.strftime("%b%d")
print(todays_date)
args = parser.parse_args()
loss = args.loss

PDEproblems_dir = f"{args.save_base_folder}{args.problem_name}/amortizedProblem/"
n_trains = [125, 500, 2000, 8_000, 32_000]  # 500, 2_000,
# n_trains.reverse()
batch_sizes = [25] + [50] * 4  # + [10]*10
# batch_sizes.reverse()
n_epochs = [600] + [300] * 4
# n_epochs.reverse()
step_sizes = [1e-4, 2e-4, 5e-4, 1e-3]  #
step_sizes.reverse()
import time

# n_trains.reverse()
for loss in ["hybrid"]:  # 6 losses ,'hybrid','fisher' 'l2', 'h1','hybrid',
    # hybrid and fisher make sure they use the normalized data
    for batch_size, n_train, n_epoch in zip(batch_sizes, n_trains, n_epochs):  # 5 training sizes
        if n_train == 2000:
            print(f"batchsize: {batch_size}")
            # if loss is not 'h1' or n_train not in (32_000, 8_000):
            for step_size in step_sizes:  # 5 stepsizes
                print(f"step_size: {step_size}")
                for j in range(5):  # 5 seeds, for spaghetti plots
                    print(f"random seed: {j}")
                    start = time.time()
                    os.system(
                        f"python -m dinox -nnSavePrefix {todays_date} -problemDir '{PDEproblems_dir}' -saveEmbeddedData -bipNum 0 -nTrain {n_train} -numBatchesCPUToGPU 1 -batchSize {batch_size} -loss {loss} -stepSize {step_size} -nEpochs {n_epoch} -runSeed {j}"  # =-bipNum 0 -nTrain {n_train} -batchSize {batch_size} -loss {loss} -stepSize 1e-3  -nEpochs 1500" #-batchedEncoding, 2e-4 for all L2 training, 1500 for [125..8_000]  500 for [10_000,...20_000] epochs for L2 for NDR, 3000 epochs for L2 hyperelasiticty for up to 5k, 1e-4, 5000 epochs for [8k, 10k], 5e-5, 5000 epochs for [16k, 20k],        1e-3, 1500 epochs for all H1 training
                    )
                    print("total setup/teardown time: ", time.time() - start)

            # load pkl file

            # plot for batch_size/stepsize all 10 (violin? or just a bunch of lines)
            # plot

    # load results and save to file for todays_date for plotting
    #

    # show error every 100 epochs
    # change the weight on the l2 term by 0.001
    # add more data (more log likelihoods)
