import os
import sys

helpersPath = os.path.expanduser("~/Dropbox (Princeton)/14_Software/python/")
sys.path.insert(1, helpersPath)

import helpers

templatePath = "~/Dropbox\ \(Princeton\)/8_Templates/"
github = "~/GitHub/tpsp"
website_docs = "~/Dropbox\ \(Princeton\)/5_CV/website/static/docs"
website_docs_github = "~/Github/brendancooley.github.io/docs"
templatePath = "~/Dropbox\ \(Princeton\)/8_Templates/"

def task_source():
    yield {
        'name': "migrating templates...",
        'actions': ["mkdir -p templates",
                    "cp -a " + templatePath + "cooley-paper-template.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-plain.latex " + "templates/",
                    "cp -a " + templatePath + "cooley-latex-beamer.tex " + "templates/"]
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