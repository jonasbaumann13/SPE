# SPE
gui and api for Single Photon Event evaluation for X-rays on 2d detectors like CCDs or CMOS

# Installation
1. for installation copy the repository directory to hard drive
2. install python (anaconda is recommended)
3. install required packages from requirements.txt
for package installation and / or new environment creation check out the "create_new_conda_environment.txt"

# Run the GUIs
The GUIs can be started from (anaconda) shell by "python evares_EXA.py" and "python evares_SPE.py", respectively.

# Access SPE algorithms with a script
It is recommended to use the spe_class in base/spe_module.py. For usage see docstring there. The actual
algorithms working directly on numpy arrays can be found in base/spe.py

# Neural Network support
There is the option to use a neural network for photon event evaluation. However, to use it, pytorch needs to be installed.
See "requirements.txt" for further information.