# DomainAdaptFTIR

## Description
This repository contains a Python implementation of the DomainAdaptFTIR method described in the paper entitled "**Title**" (doi url).

## Installation
The package is available via PyPI:
```bash
pip install DomainAdaptFTIR
```

## Usage
For detailed information on how to initialize the DomainAdaptFTIR method and configure its parameters, please refer to the `example_usage.ipynb` Jupyter Notebook and the code documentation in the `DomainAdaptFTIR.py` file. The basic usage format is as follows:
```python
from DomainAdaptFTIR import DomainAdaptFTIR

# Initialize an instance of DomainAdaptFTIR with a calibration set
dapter = DomainAdaptFTIR(spectra=<your_spectra_to_augment>,
              subject_ids=<your_subject_ids_for_spectra_to_augment>,
              calibration_set=<your_calibration_set>,
              calibration_ids=<your_calibration_set_subject_ids>)

# Create synthetic samples based on an input set of training samples X and associated sample labels y
X_gen, y_gen = DomainAdaptFTIR.run(n_per_subject_id=<your_n_per_subject_id>)

```

## Citation
> Citation goes here