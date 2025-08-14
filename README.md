<div align="center">

# SMART: Relation-Aware Learning of Geometric Representations for Knowledge Graphs
</div>

This is the PyTorch implementation of SMART. SMART is developed in the [Rotate3D framework](XXXX) , to which we refer the users for more details.  

## Overview
SMART is a knowledge graph embedding model (KGE) that can learn relation-specific elementary geometric transformations (EGTs.) 
The EGTs  supported by SMART are translation, rotation, reflection, and scaling. 
Each relation is mapped to one of those EGTs. 
This is achieved by a 3-step approach for KGEs: training, adaptive learning and freezing. 
In the training phase, all EGTs contribute equally to each relation which allows the embeddings to form. 
In the adaptive learning phase, attention scores to the EGTs per relation are adapted. 
Finally, during freezing a hard selection of an EGT per relation is performed and the model continues to improve with this particular EGT.

XXXX

## Link Prediction

This command train the SMART model on wn18rr dataset with GPU 0.
```
CUDA_VISIBLE_DEVICES=0 python -u codes/run.py --do_train \
 --cuda \
 --do_valid \
 --do_test \
 --data_path data/wn18rr \
 --model smartFW \
 -n 512 -b 1024 -d 250 \
 -g 9.0 -a 0 \
 -lr 0.0001 --max_steps 260000 \
 --test_batch_size 16 -reg 0.1  -p 2 \
 --pretraining_steps 120000 --training_steps 50000 --finetuning_steps 90000\
  -threshold 0 
```

The maximum step (max_steps) is set to the sum of pretraining_steps, training_steps and finetuning_steps, which 
are the number of iterations during the training, adaptive learning and freezing phases, respectively.
The run.sh script provides an easy way to search for hyperparameters or to evaluate our models on the KG benchmark datasets.

The code below provides scripts to train and evaluate the SMART model and its variants on the wn18rr knowledge graph.

###### Evaluate the base SMART model
    bash runs.sh train smartFW wn18rr 0 0 1024 512 250 9 0 0.0001 260000 16 0.1 2 120000 50000 90000 0 --disable_adv 
This runs the standard `SMART` model with all relation-specific transformations active and no threshold filtering (threshold = 0). 
The `--disable_adv ` flag deactivates the self-adversarial negative sampling since the temperature (`-a`) is set to 0.

###### Evaluate SMART_> (Thresholded SMART)
    bash runs.sh train smartFW wn18rr 0 0 1024 512 250 9 1.0 0.0001 260000 16 0.1 2 120000 50000 90000 0.25
This runs `SMART_>`, where each relation selects only the transformations whose weights exceed a given threshold (here 0.25).

###### Evaluate SMARTm (majority voted EGT for all relations)
    bash runs.sh train smartFW wn18rr 0 0 512 512 250 9 1.0 0.0001 260000 16 0 2 120000 50000 90000 0 --single_egt
This runs `SMARTm`, which assigns to all relations the majority-voted single transformation (EGT) by using the `--single_egt` flag.



## Citation
```
@inproceedings{
  key,
  title={},
  author={},
  booktitle={},
  pages={},
  year={}
}
```
