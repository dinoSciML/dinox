#THIS and all data generation/model problem codes should go in a common "problems" subrepo of DinoSciML i think. and people can refactor their codes
#once we come to a consensus of the easiest format for problems.
import os
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from datetime import datetime

today = datetime.now() # current date and time

parser = argparse.ArgumentParser()
parser.add_argument(
    "-save_base_folder", "--save_base_folder", type=str, default="/storage/joshua/", help="Folder to save everything in"
)
parser.add_argument(
    "-problem_name", "--problem_name", type=str, default="nonlinear_diffusion_reaction", help="Problem name"
)

todays_date = today.strftime("%b_%d")
print(todays_date)
args = parser.parse_args()
PDEproblems_dir = f"{args.save_base_folder}{args.problem_name}/problem/"
n_trains = [125, 250, 500, 1_000, 2_000, 4_000, 5_000, 8_000, 10_000, 16_000, 20_000]
n_trains.reverse()
for BIPproblem_sub_dir in os.listdir(PDEproblems_dir):
    for n_train in n_trains:
        os.system(f"python -m dinox -nn_save_name {todays_date} -problem_dir '{PDEproblems_dir}{BIPproblem_sub_dir}/' -train_data_size {n_train}")
        