import os

def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
	
def files(path):
    for file in os.listdir(path):
        if os.path.isfile(os.path.join(path, file)):
            yield file

def getFiles(folder, full=True):
	allF = files(folder)
	if full==True:
		return [folder + i for i in allF if not i.startswith('.')]
	else:
		return [i for i in allF if not i.startswith('.')]