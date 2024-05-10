# skateMAE

Source code for "SkateMAE: Synthetic Data for Skateboard Rotation Estimation"

SkateMAE is a a project in skateboard rotation estimation. The original idea
was that if such a model existed it could be used in parallel with a 
human pose estimation model to create a generalizable skateboard trick
classification model. Previous trick classification models have been restricted
by the training labels, and employed end-to-end architectures. Using rotation estimation
is not only more generalizable (i.e. any flatground trick can generally be defined
as some rotationn of the board and some rotation of the skateboarder), but it's also 
more explainable. 

# Gathering the data

All the data used in this project was either downloaded from Youtube, in which case I
don't have permission to distribute it, or was generated synthetically, the code for which 
is in this repository (gen_synth_data.py). This requires a different set of dependencies than what's 
listed in requirements.txt, because Pytorch3D has weird annoying dependencies. 

I've provided a small test set to benchmark against and verify
the provided checkpoints from the ablation study.

# Training

Assuming you've collected sufficient real world data as well as generated synthetic data,
First pretrain the model with mae_pretrain.py then train with train_mae.py.

# Testing

I've provided model checkpoints and a small test set to test against. All of this is in the /test directory. 

Run the testing script with:

```bash
python --model_path <path to skateMAE model checkpoint> --frames_path <path to directory of test images> --poses_path <path to csv file containing rotation labels>
```