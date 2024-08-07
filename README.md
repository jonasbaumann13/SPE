# SPE
gui and api for Single Photon Event evaluation for X-rays on 2d detectors like CCDs or CMOS. Information can be found here:
    https://doi.org/10.1039/C8JA00212F
![SPE principle](https://github.com/jonasbaumann13/SPE/blob/main/pictures/pee_cover_github.png?raw=true)

# Installation
1. for installation copy the repository directory to hard drive
2. install python (anaconda is recommended)
3. install required packages from requirements.txt or environment.yml

for package installation and / or new environment creation check out the "create_new_conda_environment.txt"
in your (virtual) environment also install pytorch (https://pytorch.org/get-started/locally/) if you want to use neural network support. 

# Run the GUIs
GUIs are located in folder "gui" and can be started from (anaconda) shell by 
```console
 python evares_EXA.py
```
and
```console
 python evares_SPE.py
```

# Access SPE algorithms with a script
It is recommended to use the spe_class in base/spe_module.py. For usage see docstring there. The actual
algorithms working directly on numpy arrays can be found in base/spe.py

# Neural Network support
There is the option to use a neural network for photon event evaluation. However, to use it, pytorch needs to be installed. Check out https://pytorch.org/get-started/locally/

# How to Cite
J. Baumann, R. Gnewkow, S. Staeck, V. Szwedowski-Rammert, C. Schlesiger, I. Mantouvalou and B. Kanngießer 'Photon event evaluation for conventional pixelated detectors in energy-dispersive X-ray applications', J. Anal. At. Spectrom., 2018, 33, 2043–2052; DOI: 10.1039/c8ja00212f.