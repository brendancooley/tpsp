import os
import sys

source_path = "~/Dropbox (Princeton)/14_Software/"
source_path_esc = "~/Dropbox\ \(Princeton\)/14_Software/"

helpersPath = os.path.expanduser(source_path + "python/")
sys.path.insert(1, helpersPath)

import helpers

conda_env = "python37"

github = "~/GitHub/tpsp"
website_docs = "~/Dropbox\ \(Princeton\)/5_CV/website/static/docs"
website_docs_github = "~/Github/brendancooley.github.io/docs"
templatePath = "~/Dropbox\ \(Princeton\)/8_Templates/"
verticatorPath = "~/Dropbox\ \(Princeton\)/8_Templates/plugin/verticator"
pluginDest = "index_files/reveal.js-3.8.0/plugin"
revealPath = "~/Dropbox\ \(Princeton\)/8_Templates/reveal.js-3.8.0"
# pluginDest = "index_files/reveal.js-4.0.2/plugin"
# revealPath = "~/Dropbox\ \(Princeton\)/8_Templates/reveal.js-4.0.2"

csv_dir_base = "~/Dropbox\ \(Princeton\)/1_Papers/tpsp/01_files/"
results_dir_base = csv_dir_base + "results/"
data_dir_base = csv_dir_base + "data/"

# sizes = ["mini/", "mid/", "large/", "mid_RUS/"]
sizes = ["mid/"]

code_dir = "01_code/"

hpc_base_dir = "~/tpsp/"
hpc_data_dir = hpc_base_dir + "data/"
hpc_source_dir = hpc_base_dir + "source/"
hpc_code_dir = hpc_base_dir + "code/"
hpc_results_dir = hpc_base_dir + "results/"

fig_files = ["f_ccodes.R", "f_cfct_X.R", "f_cfct_tau.R", "f_cfct_welfare.R", "f_estimates.R", "f_fit.R", "f_milex.R", "f_pr_peace.R", "f_rcv.R", "f_tau_epbt.R", "f_tau_rf.R", ]

def task_source():
    yield {
        'name': "migrating templates...",
        'actions': ["mkdir -p templates",
                    "cp -a " + templatePath + "cooley-paper-template.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-plain.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-latex-beamer.tex " + "templates/",
                    "mkdir -p bib/",
                    "cp -a " + templatePath + "python.bib " + "bib/",
                    "mkdir -p source/",
                    "cp -a " + source_path_esc + " source/"]
    }

def task_results():
    # first: conda activate python3
    # e.g. doit results:results --size mini/
    yield {
        'name': "results",
        'params':[{'name':'size',
		      'long':'size',
		      'type':str,
		      'default':'mid/'}],
        'actions':["cd " + code_dir + "; python " + "01_bootstrap.py local %(size)s 1 100"],
        'verbosity': 2,
    }

def task_paper():
	"""Build paper"""
	if os.path.isfile("references.RData") is False:
		yield {
			'name': "collecting references...",
			'actions':["R --slave -e \"set.seed(100);knitr::knit('tpsp.rmd')\""]
        }
	yield {
    	'name': "writing paper...",
    	'actions':["R --slave -e \"set.seed(100);knitr::knit('tpsp.rmd')\"",
                   "pandoc --template=templates/cooley-paper-template.latex --filter pandoc-citeproc -o tpsp.pdf tpsp.md"],
                   'verbosity': 2,
	}

def task_post_to_web():
	"""

	"""
	yield {
		'name': "posting...",
		'actions': ["cp -a tpsp.pdf " + website_docs,
					"cp -a tpsp.pdf " + website_docs_github]
	}

def task_prep_slides():
	"""

	"""
	yield {
		'name': "moving slide files",
		'actions': ["mkdir -p css",
					"cp -a " + templatePath + "slides/ " + "css/"]
	}

def task_slides():
	"""

	"""
	yield {
		'name': 'draft slides',
		'actions': ["R --slave -e \"rmarkdown::render('tpsp_slides.Rmd', output_file='index.html')\"",
            "perl -pi -w -e 's{reveal.js-3.3.0.1}{reveal.js-3.8.0}g' index.html",
            "cp -r " + revealPath + " index_files/",
            "cp -a " + verticatorPath + " " + pluginDest],
		'verbosity': 2,
	}

def task_setup_dirs():
    for i in sizes:
        yield {
            'name': "setting up results directory " + i,
            'actions':["mkdir -p " + results_dir_base + i]
        }

def task_transfer_dodo_hpc():
    yield {
        'name': "transfering dodo to hpc...",
        'actions':["scp -r dodo_hpc.py " + "bcooley@adroit.princeton.edu:" + hpc_base_dir + "dodo.py"],
        'verbosity':2
    }

def task_transfer_slurm_hpc():
    yield {
        'name': "transfering slurms to hpc...",
        'actions':["scp -r slurm/* " + "bcooley@adroit.princeton.edu:" + hpc_base_dir],
        'verbosity':2
    }

def task_transfer_data_hpc():
    # code
    # data
    # dodo.py
    # NOTE: make sure vpn is connected
    for i in sizes:
        yield {'name': "transferring data " + i,
                'actions': ["scp -r " + data_dir_base + i + "* " +
                "bcooley@adroit.princeton.edu:" + hpc_data_dir + i],
                'verbosity':2
                }

def task_transfer_code_hpc():
    yield {
        'name': "transfering code to hpc...",
        'actions':["scp -r source/* " + "bcooley@adroit.princeton.edu:" + hpc_source_dir,
        "scp -r " + code_dir + "* " + "bcooley@adroit.princeton.edu:" + hpc_code_dir],
        'verbosity':2
    }

def task_compile_results():
    yield {
        'name': "compile_results",
        'params':[{'name':'size',
		      'long':'size',
		      'type':str,
		      'default':'mid/'}],
        'actions':["cd " + code_dir + "; python 02_results_compile.py"],
        'verbosity':2
    }

def task_sync_results():
    # for mil_off
    # doit sync_results:sync_results --mil_off True
    yield {
        'name': "sync_results",
        'params':[{'name':'mil_off',
		      'long':'mil_off',
		      'type':str,
		      'default':'False'}],
        'actions':["cd " + code_dir + "; python d_sync_hpc.py " + "%(mil_off)s"],
        'verbosity':2
    }

def task_counterfactuals():
    yield {
        'name': "computing counterfactuals...",
        'actions':["cd " + code_dir + "; python 03_counterfactuals.py"],
        'verbosity':2
    }

def task_build_figs():
    for i in fig_files:
        yield {
        'name': "building figure " + i + "...",
        'actions':["cd " + code_dir + "; Rscript " + i],
        'verbosity':2
    }
