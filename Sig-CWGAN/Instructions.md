This folder contains the script to generate artificial paths from the Sig-CWGAN model as well as the script to evaluate (compute the stylized features and first four moments of) the generated paths.

Additionally, it includes the following scripts (with slight modifications to the original ones) to obtain comparable results to the report:
  - requirements.yml is the modified installation file for the dedicated environment. It basically contains the version of each library used during experiments.
  - Some additional and required commands and libraries were:
    conda env create -f requirements.yml  || conda env update -f requirements.yml
    sudo apt install g++
    sudo apt install gcc
    Cmake (library to solve some problems during installation of certain libraries)
    conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
    pip install signatory==1.2.6.1.7.1 --no-cache-dir --force-reinstall
    
  - test_metrics.py which includes the extra stylized features functions. This file has to be replaced in the folder .\lib
  - base.py which includes the extra stylized features as metrics during training. This file has to be replaced in the folder .\lib\algos
