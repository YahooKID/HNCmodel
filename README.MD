This is a code repository for "_An innovative and efficient diagnostic prediction flow for head and neck cancer: a deep learning approach for multi-modal survival analysis prediction based on text and multi-center PET/CT images._"

The code is currently in the preliminary open-source state, and we are working on improving this repository.

The current implementation of the Mean Absolute Error (MAE) part is based on https://github.com/facebookresearch/mae. orz

The fine-tuning process that follows is based on https://github.com/salesforce/LAVIS. orz

Due to some conflicts in the environments of these two repositories (timm@mae==0.3.2 and timm@lavis==0.4.12), we recommend setting up two separate environments to avoid unnecessary issues. Please refer to the README of each open-source project for instructions on setting up the respective environments.

For the mae project, please replace /util/petct_dataset.py in this repository with the code from the mae project, and modify and replace the dataset in main_pretrain.py.

For the lavis project, please override risk_model.py and loss.py in this project. Load the risk_model and Loss into a new BaseModel and register them. Then, register util/pre_dataset.py from this project into lavis. For more details, refer to the documentation of the lavis project. Afterward, you can start training directly by following the training script of Blip2.

However, modifying the lavis project can be challenging. We have attempted to implement a new independent training code, such as survival_analysis_train.py. This code re-implements the training process to achieve maximum compatibility with the QFormer structure in the lavis project. The code has been tested in our project and works fine. However, since it has not been thoroughly tested, there may be some issues. Please feel free to raise an issue if you encounter any problems.

We will make every effort to improve this code, but due to busy work and life, we may not be able to complete this process quickly. We appreciate your understanding.

As the MAE part mentioned in this paper is a generative network, there is a potential risk of leaking facial information of individuals in the Head and Neck section of the training set. Therefore, we are unable to open-source the training weights temporarily. We are exploring other possibilities for open-sourcing the weights. (QAQ & orz)

Some experience we have:
1. If you have less data, maybe you can try Vit-S and adjust the lr.
2. augment is important but not necessary.
3. Q-Former is a little difficult to train, On different dataset maybe you should try different parameters.
4. If you want to deploy it but not just use it for research, maybe you can try load the Q-Former official pretrained weight.
5. Maybe you can try crop the image to ensure the proportion of foreground become higher, it can improve the performance of the net, we will release our preprocess(before dataset-loader) code in the close future, it's a little confused now.

Anyway, core filelist:   
1. main_pretrain.py for pretrain   
2. survival_analysis_train.py for train follow stream.   
3. prepare_dataset_demo.py offers a template of how to generate a dataset in our study, you can try to adjust it.
