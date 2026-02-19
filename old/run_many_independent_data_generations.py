import subprocess


def run_many():
    for i in range(50, 85):  # 50 to 85: 125, 500, 2k, 8k, 32k, 128k
        command = (
            f"export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; export NUMEXPR_NUM_THREADS=1; nohup python 1_generate_data_gradient_loaded_data_problems.py "
            f"-process {i} "
            f"2>&1 &"
        )
        # Start independent runs using subprocess, nohup & so  you can terminate your shell
        subprocess.Popen(command, shell=True)


if __name__ == "__main__":
    run_many()
