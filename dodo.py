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

csv_dir_base = "~/Dropbox\ \(Princeton\)/1_Papers/tpsp/01_data/"
results_dir_base = csv_dir_base + "results/"
data_dir_base = csv_dir_base + "data/"

sizes = ["mini/", "mid/", "large/"]

code_dir = "01_analysis/"

hpc_base_dir = "~/tpsp/"
hpc_data_dir = hpc_base_dir + "data/"
hpc_source_dir = hpc_base_dir + "source/"
hpc_code_dir = hpc_base_dir + "code/"

def task_source():
    yield {
        'name': "migrating templates...",
        'actions': ["mkdir -p templates",
                    "cp -a " + templatePath + "cooley-paper-template.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-plain.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-latex-beamer.tex " + "templates/",
                    "mkdir -p source/",
                    "cp -a " + source_path_esc + "python/ source/"]
    }

def task_results():
    # first: conda activate python3
    yield {
        'name': "collecting results...",
        'actions':["python " + code_dir + "results.py local"],
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
	"""build slides"""
	yield {
		'name': "writing slides...",
		'actions': ["R --slave -e \"rmarkdown::render('tpsp_slides.Rmd', output_file='index.html')\""],
		'verbosity': 2,
	}

def task_methods_slides():
	yield {
		'name': "building methods slides...",
		'actions':["R --slave -e \"rmarkdown::render(\'" + "tpsp_methods_slides.rmd" + "\', output_file=\'" + "tpsp_methods_slides.pdf" +"\')\""]
	}

def task_notes():
    notesFiles = helpers.getFiles("notes/")
    for i in range(len(notesFiles)):
        fName = notesFiles[i].split(".")[0]
        suffix = notesFiles[i].split(".")[1]
        if suffix == "md":
            yield {
                'name': notesFiles[i],
                'actions':["pandoc --template=templates/cooley-plain.latex -o " +
                            fName + ".pdf " + notesFiles[i]]
            }

def task_setup_dirs():
    for i in sizes:
        yield {
            'name': "setting up results directory " + i,
            'actions':["mkdir -p " + results_dir_base + i]
        }

def task_transfer_hpc():
    # code
    # data
    # dodo.py
    yield {
        'name': "transfering files to hpc...",
        'actions':["scp -r " + code_dir + "* " + "bcooley@adroit.princeton.edu:" + hpc_code_dir,
        "scp -r source/* " + "bcooley@adroit.princeton.edu:" + hpc_source_dir,
        "scp -r slurm/* " + "bcooley@adroit.princeton.edu:" + hpc_base_dir]
    }
    for i in sizes:
        yield {'name': "transferring data " + i,
                'actions': ["scp -r " + data_dir_base + i + "* " +
                "bcooley@adroit.princeton.edu:" + hpc_data_dir + i]}
