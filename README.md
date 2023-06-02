<div align="center">
    <img src="docs/images/logo.svg">
    <h3>Molecular Out-Of-Distribution Generalization</h3>
    <p>
        Close the testing-deployment gap in molecular scoring.
    </p>
</div>

---

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI]()]()

Python repository with all the code that was used for the MOOD paper.

## Setup
We recommend you to use `mamba` ([learn more](https://github.com/mamba-org/mamba)).

```shell
mamba env create -n mood -f env.yml 
conda activate mood
pip install -e . 
```

## Overview
The repository is set-up to make the results easy to reproduce. If you get stuck or like to learn more, please feel free to open an issue.

### CLI
After installation, the MOOD CLI can be used to reproduce the results.
```shell
mood --help
```

### Data
All data has been made available in a public GCP bucket. See [`gs://mood-data-public`](https://storage.googleapis.com/mood-data-public/).

### Code
- `mood/`: This is the main part of the codebase. It contains Python implementations of several reusable components and defines the CLI.
- `notebooks/`: Notebooks were used to visualize and otherwise explore the results. All plots in the paper can be reproduced through these notebooks.
- `scripts/`: Generally more messy pieces of code that were used to generate (intermediate) results.

## Use the MOOD protocol
<div align="center">
    <img src="docs/images/protocol.png" width="80%">
</div>

One of the main results of the MOOD paper, is the MOOD protocol. This protocol helps to close the testing-deployment gap in molecular scoring by finding the most representative splitting method. To make it easy for others to experiment with this protocol, we made an effort to make it easy to use.

```python
import datamol as dm
from sklearn.model_selection import ShuffleSplit
from mood.splitter import PerimeterSplit, MaxDissimilaritySplit, PredefinedGroupShuffleSplit, MOODSplitter

# Load your data
smiles = ...
X = ...
y = ...

# Load your deployment data
X_deployment = ...

scaffolds = [dm.to_smiles(dm.to_scaffold_murcko(dm.to_mol(smi))) for smi in smiles]
candidate_splitters = {
    "Random": ShuffleSplit(),
    "Scaffold": PredefinedGroupShuffleSplit(groups=scaffolds),
    "Perimeter": PerimeterSplit(),
    "Maximum Dissimilarity": MaxDissimilaritySplit(),
}

mood_splitter = MOODSplitter(candidate_splitters)
mood_splitter.fit(X, y, X_deployment=X_deployment)

for train, test in mood_splitter.split(X, y):
    # Work your magic!
    ...
```

## How to cite
Please cite MOOD if you use it in your research: [![DOI]()]()

```
# TODO: Bibtex
```

