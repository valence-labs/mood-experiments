import datamol as dm


"""
Where results and data are saved to
"""
CACHE_DIR = dm.fs.get_cache_dir("MOOD")

"""
For the downstream applications (optimization and virtual screening)
we save all related data to this directory
"""
DOWNSTREAM_APPS_DATA_DIR = "https://storage.valencelabs.com/mood-data/downstream_applications/"

"""
Where the results of MOOD are saved
"""
RESULTS_DIR = "https://storage.valencelabs.com/mood-data/results/"

"""
The two downstream applications we consider for MOOD as application areas of molecular scoring
"""
SUPPORTED_DOWNSTREAM_APPS = ["virtual_screening", "optimization"]

"""
Where data related to specific datasets is saved
"""
DATASET_DATA_DIR = "https://storage.valencelabs.com/mood-data/datasets/"


"""The number of epochs to train NNs for"""
NUM_EPOCHS = 100
