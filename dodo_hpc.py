# transfer to adroit:
# scp ~/Github/tpsp/dodo_hpc.py bcooley@adroit.princeton.edu:~/tpsp/dodo.py
# module load anaconda3
# conda activate python37

hpc_base_dir = "~/home/bcooley/tpsp/"
hpc_data_dir = "data/"
hpc_code_dir = "code/"
hpc_source_dir = "source/"
hpc_results_dir = "results/"

def task_hpc_setup():
    yield {
        'name': "setting up hpc...",
        'actions':["mkdir -p " + hpc_data_dir + "; \
        mkdir -p " + hpc_code_dir + "; \
        mkdir -p " + hpc_source_dir + "; \
        mkdir -p " + hpc_results_dir
        ]
    }
