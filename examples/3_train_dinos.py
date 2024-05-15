# THIS and all data generation/model problem codes should go in a common "problems" subrepo of DinoSciML i think. and people can refactor their codes
# once we come to a consensus of the easiest format for problems.
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    default= "hyperelasticity", #"nonlinear_diffusion_reaction",#"hyperelasticity",
    help="Problem name",
)

parser.add_argument(
    "-loss",
    "--loss",
    type=str,
    default= "l2", #'l2
    help="loss function",
)

todays_date = today.strftime("%b%d")
print(todays_date)
args = parser.parse_args()
loss = args.loss

PDEproblems_dir = f"{args.save_base_folder}{args.problem_name}/problem/"
n_trains = [16_000, 20_000]  #125, 250, 500, 1_000, 2_000, 4_000, 5_000, 8_000, 10_000, 
batch_sizes = [25]*11 # + [10]*10

import time
# n_trains.reverse()
for BIPproblem_sub_dir in os.listdir(PDEproblems_dir):
    for batch_size, n_train in zip(batch_sizes, n_trains):
        print(f"batchsize: {batch_size}")
        start = time.time()
        os.system(
            f"python -m dinox -nnSavePrefix {todays_date} -problemDir '{PDEproblems_dir}{BIPproblem_sub_dir}/' -nTrain {n_train} -batchSize {batch_size} -loss {loss} -stepSize 5e-5 -nEpochs 5000  -batchedEncoding" #2e-4 for all L2 training, 1500 for [125..8_000]  500 for [10_000,...20_000] epochs for L2 for NDR, 3000 epochs for L2 hyperelasiticty for up to 5k, 1e-4, 5000 epochs for [8k, 10k], 5e-5, 5000 epochs for [16k, 20k],        1e-3, 1500 epochs for all H1 training
         )
        print("total setup/teardown time: ", time.time()-start)
 