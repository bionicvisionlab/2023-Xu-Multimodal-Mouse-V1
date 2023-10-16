[![license](https://img.shields.io/badge/License-BSD%202--Clause-blue.svg)](https://github.com/bionicvisionlab/2023-Xu-Multimodal-Mouse-V1/blob/master/LICENSE)
[![Data](https://img.shields.io/badge/data-osf.io-lightgrey.svg)]([https://osf.io/s2udz/](https://doi.org/10.17605/OSF.IO/MSP3A))
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2023.05.30.542912v1-orange)](https://doi.org/10.1101/2023.05.30.542912v1)

# Multimodal Deep Learning Model Unveils Behavioral Dynamics of V1 Activity in Freely Moving Mice

Please cite as:

> A Xu, Y Hou, CM Niell, M Beyeler (2023). Multimodal deep learning model unveils behavioral dynamics of V1 activity in freely moving mice.
> *Advances in Neural Information Processing Systems (NeurIPS) 2023*

![image](https://github.com/bionicvisionlab/2023-Xu-Multimodal-Mouse-V1/assets/5214334/b467ea16-3e3c-447b-b335-cdeb180c6f7f)

Preprint available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2023.05.30.542912v1).
The data used to train the models (mouse-data-10-segment-split-70-30-48ms.zip) and the trained model weights (table_1.zip, table_2.zip, table_3.zip) can be found at [OSF](https://doi.org/10.17605/OSF.IO/MSP3A).

Dependencies:
* PyTorch
* NumPy
* SciPy
* Kornia

#### Instructions

The code mostly consists of self-contained Jupyter notebooks. 
* `mouse_model`: Dataset and evaluation utilities that are commonly used in the following Jupyter notebooks.
* `train_cnn_shifter_table_1.ipynb`: Training and evaluation of the CNN model in Table 1.
* `train_autoencoder_shifter_table_1.ipynb`: Training and evaluation of the autoencoder model in Table 1.
* `train_resnet_shifter_table_1.ipynb`: Training and evaluation of the ResNet model in Table 1.
* `train_efficientnet_shifter_table_1.ipynb`: Training and evaluation of the EfficientNet model in Table 1.
* `train_sensorium_shifter_table_1&2.ipynb`: Training and evaluation of the Sensorium and Sensorium+ models in Table 1 and Table 2. Please refer to [Sensorium](https://github.com/sinzlab/sensorium) for dependencies.
* `train_cnn_gru_shifter_table_2.ipynb`: Training and evaluation of the models with different behavioral feature sets in Table 2 (rows 1 - 4).
* `train_cnn_gru_shifter_table_2_sens_orig.ipynb`: Training and evaluation of the models with behavioral feature set S.
* `train_cnn_gru_shifter_table_3.ipynb`: Training and evaluation of the models in Table 3.
* `fig-3-scatter-vis.ipynb`: Plotting Figure 3.
* `gradient_ascent_cnn_gru_shifter_fig_4.ipynb`: Gradient ascent analysis behind Figure 4.
* `fig-4-visual-rf.ipynb`: Plotting Figure 4.
* `saliency_map_cnn_gru_shifter_fig_5.ipynb`: Saliency map analysis behind Figure 5.
* `fig-5-saliency.ipynb`: Plotting Figure 5.
* `gradient_ascent_cnn_gru_shifter_fixed_beh_fig_c1.ipynb`: Gradient ascent analysis behind Figure C1.
* `fig-c1-visual-rf-fixed-beh.ipynb`: Plotting Figure C1.
