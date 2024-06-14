# Causal SSL

### Introduction
When citing this research, please use:
```
@misc{ibrahim2024semisupervised,
      title={Semi-Supervised Learning for Deep Causal Generative Models}, 
      author={Yasin Ibrahim and Hermione Warr and Konstantinos Kamnitsas},
      year={2024},
      eprint={2403.18717},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
Ibrahim, Y., Warr, H., Kamnitsas, K.: Semi-Supervised Learning for Deep Causal Generative Models. arXiv preprint arXiv:2403.18717 (2024)


### Running Code
To install the required packages, run:
```
pip install -r requirements.txt
```

To train the generative model, navigate to the ```/src``` folder and run:
```
bash run_local.sh experiment_name
```

By default, this is set to use the coloured Morpho-MNIST data as described in the paper. To generate new semi-synthetic data with custom relationships between the variables, edit or create a new colour generator function in the ```mnist.py``` file in the main directory, and then run:
```
python mnist.py [-h] [--size No. labelled samples] [--random Random labelling]
```
Currently, the variables available are:
 - Thickness
 - Intensity
 - Foreground (digit) colour
 - Background colour
 - Digit

Hyperparameters can be edited in the ```/src/hps.py``` file. These include:

- **labelled** : The proportion of data that is labelled (for MIMIC data)
- **random**: Whether to use random labelling 
- **seed**: Set random seed
- **scm_thresh**: When to start using predicted outputs
- **reg_thresh**: When to start counterfactual regularisation
- **rw**: Counterfactual regularisation weight
- **zrw**: Latent variable weight in counterfactual regularisation
- **beta**: KL penalty weight
- **cw**: Classifier weight

### Method Outline

![](figures/main_model.svg) 

### Example Outputs

<img src="figures/mimic_cfs.svg" height="250"> <img src="figures/mnist.svg" height="250">

### Acknowledgements
Code for the generative model is adapted from:
- [https://github.com/biomedia-mira/causal-gen](https://github.com/biomedia-mira/causal-gen)\
      De Sousa Ribeiro, F., Xia, T., Monteiro, M., Pawlowski, N., Glocker, B.: High Fidelity Image Counterfactuals with Probabilistic Causal Models. In: Proceedings of the 40th International Conference on Machine Learning. vol. 202, pp. 7390-7425 (2023)

Code for the custom MNIST data generation is based on:
- [https://github.com/dccastro/Morpho-MNIST](https://github.com/dccastro/Morpho-MNIST)\
      Castro, D. C., Tan, J., Kainz, B., Konukoglu, E., & Glocker, B. (2019). Morpho-MNIST: Quantitative Assessment and Diagnostics for Representation Learning. Journal of Machine Learning Research, 20(178).

- [https://github.com/salesforce/corr_based_prediction](https://github.com/salesforce/corr_based_prediction)\
      Arpit, D., Xiong, C. and Socher, R., 2019. Predicting with high correlation features. arXiv preprint arXiv:1910.00164.
