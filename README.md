# Crystal
Defending Privacy Against More Knowledgeable Membership Inference Attackers

This repository contains code for "Defending Privacy Against More Knowledgeable Membership Inference Attackers". 

# Setting

Required python tool : Python 3.5, Pytorch 1.2.0, torchvision 0.3.0, torchtext 0.6.0, numpy, argparse (1.1), scipy (1.2.0). GPU support (you can also comment related code if not using GPU). 


# Dataset description: 

The CIFAR-10 dataset is needed at the data folder

# Code usage: 

We provide the sample code on the CIFAR-10. You can imitate our program to run on your data set.

All the configs is in the config folder, including the epochs, learning rate, training data size and so on.

If you want to run the CIFAR-10 dataset, you can directly run runs.py (python runs.py) after installing python tools. All the settings keep the same as the paper. 

# Citation
If you use this method or this code in your paper, then please cite it:

```
@inproceedings{yin2021defending,
  title={Defending Privacy Against More Knowledgeable Membership Inference Attackers},
  author={Yin, Yu and Chen, Ke and Shou, Lidan and Chen, Gang},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={2026--2036},
  year={2021}
}
```

# License
This project is CC-BY-NC-licensed.

# Acknowledgements

Our implementation uses the source code from the following repositories:
* membership_inference_attack(https://github.com/AdrienBenamira/membership_inference_attack#membership_inference_attack)
* mixup(https://github.com/facebookresearch/mixup-cifar10)

# References
* R. Shokri, M. Stronati, C. Song, and V. Shmatikov. Membership Inference Attacks against Machine Learning Models in IEEE Symposium on Security and Privacy, 2017.
* Hongyi Zhang and Moustapha Cisse and Yann N. Dauphin and David Lopez-Paz. mixup: Beyond Empirical Risk Minimization in ICLR, 2018
