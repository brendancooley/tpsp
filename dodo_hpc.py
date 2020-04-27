# transfer to adroit:
# scp ~/Github/tpsp/dodo_hpc.py bcooley@adroit.princeton.edu:~/tpsp/dodo.py
# module load anaconda3
# conda activate python37
    # conda install -c conda-forge doit
    # conda install -c anaconda numpy
    # conda install -c conda-forge autograd
    # conda install -c anaconda statsmodels
    # conda install -c conda-forge ipyopt
    # conda install -c intel mkl


# ssh bcooley@adroit.princeton.edu
# cd tpsp
# doit hpc_setup
# transfer data using other dodo.py, transfer_hpc
# sbatch *.slurm


hpc_base_dir = "~/home/bcooley/tpsp/"
hpc_data_dir = "data/"
hpc_code_dir = "code/"
hpc_source_dir = "source/"
hpc_results_dir = "results/"

sizes = ["mini/", "mid/", "large/", "mid_RUS/"]

def task_hpc_setup():
    yield {
        'name': "setting up hpc...",
        'actions':["mkdir -p " + hpc_data_dir,
        "mkdir -p " + hpc_code_dir,
        "mkdir -p " + hpc_source_dir]
        }
    for i in sizes:
        yield {
            'name': "data dir " + i,
            'actions':["mkdir -p " + hpc_data_dir + i]
        }

def task_results():
    # first: conda activate python37
    yield {
        'name': "results",
        'params':[{'name':'size',
		      'long':'size',
		      'type':str,
		      'default':'mini/'}],
        'actions':["python " + hpc_code_dir + "bootstrap.py hpc %(size)s"],
        'verbosity': 2,
    }
