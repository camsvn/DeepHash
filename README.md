# DeepHash and triplet learning with online triplet mining

PyTorch implementation of DeepHash and triplet networks for learning embeddings.

DeepHash is a model used to create Binary encodings of images for that can be used in image retrival systems.

# Installation

Requires [pytorch](http://pytorch.org/) 1.3 with torchvision 


# Code structure

- **datasets.py**
  - *SiameseMNIST* class - wrapper for a MNIST-like dataset, returning random positive and negative pairs
  - *TripletMNIST* class - wrapper for a MNIST-like dataset, returning random triplets (anchor, positive and negative)
  - *BalancedBatchSampler* class - BatchSampler for data loader, randomly chooses *n_classes* and *n_samples* from each class based on labels
  - *TripletCifar* cass - wrapper for the CIFAR dataset, returning random triplets (anchor, positive and negative)
  - *BalancedBatchSamplerCifar* cass - BatchSampler for dataloader, CIFAR doesn't work with the other sampler hence this one.
- **networks.py**
  - *EmbeddingNet* - base network for encoding images into embedding vector
  - *ClassificationNet* - wrapper for an embedding network, adds a fully connected layer and log softmax for classification
  - *SiameseNet* - wrapper for an embedding network, processes pairs of inputs
  - *TripletNet* - wrapper for an embedding network, processes triplets of inputs
- **losses.py**
  - *ContrastiveLoss* - contrastive loss for pairs of embeddings and pair target (same/different)
  - *TripletLoss* - triplet loss for triplets of embeddings
  - *OnlineContrastiveLoss* - contrastive loss for a mini-batch of embeddings. Uses a *PairSelector* object to find positive and negative pairs within a mini-batch using ground truth class labels and computes contrastive loss for these pairs
  - *OnlineTripletLoss* - triplet loss for a mini-batch of embeddings. Uses a *TripletSelector* object to find triplets within a mini-batch using ground truth class labels and computes triplet loss
- **trainer.py**
  - *fit* - unified function for training a network with different number of inputs and different types of loss functions
- **metrics.py**
  - Sample metrics that can be used with *fit* function from *trainer.py*
- **utils.py**
  - *PairSelector* - abstract class defining objects generating pairs based on embeddings and ground truth class labels. Can be used with *OnlineContrastiveLoss*.
    - *AllPositivePairSelector, HardNegativePairSelector* - PairSelector implementations
  - *TripletSelector* - abstract class defining objects generating triplets based on embeddings and ground truth class labels. Can be used with *OnlineTripletLoss*.
    - *AllTripletSelector*, *HardestNegativeTripletSelector*, *RandomNegativeTripletSelector*, *SemihardNegativeTripletSelector* - TripletSelector implementations



# References

[1] Raia Hadsell, Sumit Chopra, Yann LeCun, [Dimensionality reduction by learning an invariant mapping](http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf), CVPR 2006

[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. [Facenet: A unified embedding for face recognition and clustering.](https://arxiv.org/abs/1503.03832) CVPR 2015

[3] Alexander Hermans, Lucas Beyer, Bastian Leibe, [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/pdf/1703.07737), 2017

[4] Brandon Amos, Bartosz Ludwiczuk, Mahadev Satyanarayanan, [OpenFace: A general-purpose face recognition library with mobile applications](http://reports-archive.adm.cs.cmu.edu/anon/2016/CMU-CS-16-118.pdf), 2016

[5] Yi Sun, Xiaogang Wang, Xiaoou Tang, [Deep Learning Face Representation by Joint Identification-Verification](http://papers.nips.cc/paper/5416-deep-learning-face-representation-by-joint-identification-verification), NIPS 2014
