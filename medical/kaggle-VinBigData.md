vinBigData:
https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/data?select=test
kaggle competitions download -c vinbigdata-chest-xray-abnormalities-detection


## At colab.google.com is simple:

Create api key
and yout colab:

from google.colab import files
files.upload() #upload kaggle.json

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json

!kaggle kernels list — user YOUR_USER — sort-by dateRun

!kaggle competitions download -c DATASET

!unzip -q train.csv.zip -d .
!unzip -q test.csv.zip -d .
!ls