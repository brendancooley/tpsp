# transfer to adroit:
# scp ~/Github/tpsp/dodo_hpc.py bcooley@adroit.princeton.edu:~/tpsp/dodo.py
# module load anaconda3
# conda activate python37
# conda install -c conda-forge doit
# conda install -c anaconda numpy
# conda install -c conda-forge autograd
# conda install -c anaconda statsmodels
# conda install -c conda-forge ipyopt


hpc_base_dir = "~/home/bcooley/tpsp/"
hpc_data_dir = "data/"
hpc_code_dir = "code/"
hpc_source_dir = "source/"
hpc_results_dir = "results/"

hpc_estimates_dir = hpc_results_dir + "estimates/"
hpc_counterfactuals_dir = hpc_results_dir + "counterfactuals/"

def task_hpc_setup():
    yield {
        'name': "setting up hpc...",
        'actions':["mkdir -p " + hpc_data_dir + "; \
        mkdir -p " + hpc_code_dir + "; \
        mkdir -p " + hpc_source_dir + "; \
        mkdir -p " + hpc_results_dir + "; \
        mkdir -p " + hpc_estimates_dir + "; \
        mkdir -p " + hpc_counterfactuals_dir
        ]
    }

def task_results():
    # first: conda activate python37
    yield {
        'name': "collecting results...",
        'actions':["python " + hpc_code_dir + "results.py hpc"],
        'verbosity': 2,
    }
