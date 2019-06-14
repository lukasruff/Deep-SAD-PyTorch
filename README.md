# Deep SAD: A Deep Semi-supervised Anomaly Detection method
This repository will provide a [PyTorch](https://pytorch.org/) implementation of the *Deep SAD* method presented in our paper ”Deep Semi-Supervised Anomaly Detection”.

**6 Jun 2019: I'm in the process of cleaning up the code. The full code will be released shortly.**


## Citation and Contact
You find a preprint of the Deep Semi-Supervised Anomaly Detection paper on arXiv 
[https://arxiv.org/abs/1906.02694](https://arxiv.org/abs/1906.02694).

If you find our work useful, please also cite the paper:
```
@article{ruff2019,
  title     = {Deep Semi-Supervised Anomaly Detection},
  author    = {Ruff, Lukas and Vandermeulen, Robert A. and G{\"o}rnitz, Nico and Binder, Alexander and M{\"u}ller, Emmanuel and M{\"u}ller, Klaus-Robert and Kloft, Marius},
  journal   = {arXiv preprint arXiv:1906.02694},
  year      = {2019}
}
```

If you would like get in touch, just drop us an email to [contact@lukasruff.com](mailto:contact@lukasruff.com).


## Abstract
> > Deep approaches to anomaly detection have recently shown promising results over shallow approaches on high-dimensional data. Typically anomaly detection is treated as an unsupervised learning problem. In practice however, one may have---in addition to a large set of unlabeled samples---access to a small pool of labeled samples, e.g. a subset verified by some domain expert as being normal or anomalous. Semi-supervised approaches to anomaly detection make use of such labeled data to improve detection performance. Few deep semi-supervised approaches to anomaly detection have been proposed so far and those that exist are domain-specific. In this work, we present Deep SAD, an end-to-end methodology for deep semi-supervised anomaly detection. Using an information-theoretic perspective on anomaly detection, we derive a loss motivated by the idea that the entropy for the latent distribution of normal data should be lower than the entropy of the anomalous distribution. We demonstrate in extensive experiments on MNIST, Fashion-MNIST, and CIFAR-10 along with other anomaly detection benchmark datasets that our approach is on par or outperforms shallow, hybrid, and deep competitors, even when provided with only few labeled training data.


## License
MIT
