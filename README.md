Candescence - GRACE - Software to capture the essence of Candida albicans morphologies. :microscope: :crystal_ball:

GRACE here refers to a collaboration with Case, Westman et al. (2023) in the Cowen Lab at Toronto. Here we used transfer learning from the original Candesence to optimize it for the specific type of microscopy images they have for the GRACE conditional knock-out collection for Candida albicans.

This is an effort of the [Hallett lab](https://mikehallett.science). The full project is described at our [OSF Candescence](https://osf.io/qdxbp/) repository.

The original project is described in our bioRxiv manuscript Bettauer et al. (2021) bioRxiv #445299.

This repository houses the code for building the Candescence object-detection/object-classification classifiers for both the so-called Macrophage and tissue culture (TC) image compendia.


In high level terms, you will need the following. 

-- Python 3.7; R 3.6  and many libraries for each. 
-- You will need PyTorch and torchvision. 
-- [MMDETECTION](https://mmdetection.readthedocs.io/en/latest/) and Open-MMLab



You will have to define effectively four paths that are referred to throughout our code:

CANDESCENCE: this points to the root of the directories downloaded. <br>
ROOT: the file path to this repo on your machine<br>
TOOL: this points to the MMDETECTION executable.<br>
