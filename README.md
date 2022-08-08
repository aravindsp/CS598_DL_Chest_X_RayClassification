# Multi-Label Chest X-Ray Classification - CS 598 DL Project

Authors
--------------
- Aravind Pillai


Our goal in this paper is to develop a lightweight solution to detect 14 different chest conditions from an X ray image. Given an X-ray image as input, our classifier outputs a label vector indicating which of 14 disease classes does the image fall into. For training, we used dataset consisting of 224,316 chest radiographs of 65,240 patients who underwent a radiographic examination from Stanford University Medical Center between October 2002 and July 2017.


Experiment set up
-----------------
We used AWS Deep Learning AMI (Ubuntu 18.04), g4dn.2xlarge for training and prediction.

Execute Notebook
-----------------
- source activate pytorch_latest_p37     
- jupyter nbconvert --to=notebook --inplace --ExecutePreprocessor.enabled=True cs-598-multi-labelchest-x-ray-classification.ipynb > new.log 2>&1 & >> disown   



