# AdaptFTIR

## Description
This repository contains a Python implementation of the AdaptFTIR method described in the paper entitled "**Bridging Spectral Gaps: Cross-Device Model Generalization in Blood-Based Infrared Spectroscopy**" (https://doi.org/10.1021/acs.analchem.5c00185).

## Installation
The package is available via PyPI:
```bash
pip install AdaptFTIR
```

## Usage
For detailed information on how to initialize the AdaptFTIR method and configure its parameters, please refer to the `example_usage.ipynb` Jupyter Notebook and the code documentation in the `AdaptFTIR.py` file. The basic usage format is as follows:
```python
from AdaptFTIR import AdaptFTIR

# Initialize an instance of AdaptFTIR with a calibration set
adapter = AdaptFTIR(spectra=<your_spectra_to_augment>,
              subject_ids=<your_subject_ids_for_spectra_to_augment>,
              calibration_set=<your_calibration_set>,
              calibration_ids=<your_calibration_set_subject_ids>)

# Create synthetic samples based on an input set of training samples X and associated sample labels y
X_gen, y_gen = AdaptFTIR.run(n_per_subject_id=<your_n_per_subject_id>)

```

## Citation
> F. Nemeth, N. Leopold-Kerschbaumer, D. Debreceni, F. Fleischmann, K. Borbely, D. Mazurencu-Marinescu-Pele, T. Bocklitz, M. Žigman, K. Kepesidis, Bridging Spectral Gaps: Cross-Device Model Generalization in Blood-Based Infrared Spectroscopy, Analytical Chemistry (2025). https://doi.org/10.1021/acs.analchem.5c00185
