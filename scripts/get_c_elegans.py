### Jianshu Zhao, jianshu.zhao@gatech.edu

import urllib.request
import tarfile

# download the C. elegans data
url = "http://cb.csail.mit.edu/cb/densvis/datasets/packer_c-elegans_data.tar.gz"
file_name = "./packer_c-elegans_data.tar.gz"

urllib.request.urlretrieve (url, file_name)

# extract the data
tar = tarfile.open(file_name, "r:gz")
tar.extractall()
tar.close()

