This folder contains the script to generate artificial paths from the Sig-CWGAN model as well as the script to evaluate (compute the stylized features and first four moments of) the generated paths.

Additionally, it includes the following scripts (with slight modifications to the original ones) to obtain comparable results to the report:
  - test_metrics.py which includes the extra stylized features functions. This file has to be replaced in the folder .\lib
  - base.py which includes the extra stylized features as metrics during training. This file has to be replaced in the folder .\lib\algos
