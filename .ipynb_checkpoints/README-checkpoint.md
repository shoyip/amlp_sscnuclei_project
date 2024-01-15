# Segmentation of Skeletal Stem Cells Nuclei in Phase Contrast Images

This is the repository containing the files for the project work of
the [Advanced Machine Learning for Physics course](https://elearning.uniroma1.it/enrol/index.php?id=16240)
(AY 2022/2022) at Sapienza University of Rome taught by prof. Stefano Giagu.

## Content of the repository

The repository contains:
- the `assets` folder containing images used in the notebooks
- the `SSCNuclei_Project.ipynb` Jupyter notebook containing the code for loading the data, preparing the model and training it.

## Setup the project

In order to run the code it is needed to first define the environment.

By using the Anaconda software we can create an environment by issuing the following command

```bash
$ conda create --name <env> --file requirements.txt
```

This will install all the needed requirements. Running

```bash
$ jupyter notebook
```

will open up the web page showing the folder in the default browser, then clicking on the `SSCNuclei_Project.ipynb` file will open up the main notebook containing the code.

## Impressum

The lab work has been performed by Antonio Culla, Domenico Caudo, Biagio Palmisano and Shoichi Yip.

The segmentation process has been devised by Shoichi Yip.

The project was coordinated by Stefania Melillo and Leonardo Parisi.

This project belongs to the ERC RG.BIO project of the CoBBS Lab (Collective Behaviour in Biological Systems Laboratory) at CNR ISC - Sapienza.

![](assets/logo_black.png)