import urllib.request
import tarfile

# download the PBMC data
url = "http://cb.csail.mit.edu/cb/densvis/datasets/zheng_pbmc_data.tar.gz"
file_name = "./zheng_pbmc_data.tar.gz"

urllib.request.urlretrieve (url, file_name)

# extract the data
tar = tarfile.open(file_name, "r:gz")
tar.extractall()
tar.close()