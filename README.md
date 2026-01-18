# Remote Sensing Image Captioning avec ResNet50 + Transformer

## Vue d'ensemble

Ce projet implémente un système de génération automatique de descriptions (captions) pour des images de télédétection (remote sensing). Il utilise une architecture encoder-decoder basée sur ResNet50 pour l'extraction de features visuelles et un Transformer pour la génération de texte.

### Objectif
Générer automatiquement des descriptions textuelles précises pour des images satellites et aériennes du dataset RSICD (Remote Sensing Image Captioning Dataset).


##  Architecture du modèle

### Encoder : ResNet50 pré-entraîné
- **Backbone** : ResNet-50 pré-entraîné sur ImageNet
- **Extraction de features** : Couches convolutionnelles (jusqu'à la dernière couche avant FC)
- **Pooling** : Pooling (1x1)
- **Projection** : 49 vecteurs de 256 dimensions
- **Fine-tuning** : Les 2 derniers blocs ResNet sont entraînables

### Decoder : Transformer
- **Architecture** : Transformer Decoder (6 couches)
- **Dimensions** :
  - Embedding : 256
  - Decoder : 512
  - Attention heads : 8
- **Positional Encoding** : Encodage sinusoïdal pour les positions des tokens
- **Mécanisme d'attention** : Multi-head attention avec masque causal
- **Dropout** : 0.1

### Paramètres du modèle
- **Total de paramètres** : ~50,5M
- **Paramètres entraînables** : ~49,1M
- **Paramètres gelés** : ~1,4M


## Dataset : RSICD

Le **Remote Sensing Image Captioning Dataset (RSICD)** contient des images satellites avec descriptions textuelles.

### Statistiques
- **Total d'images** : 10 921 images
- **Distribution** :
  - Train : 8 734 images (80%)
  - Validation : 1 094 images (10%)
  - Test : 1 093 images (10%)

### Caractéristiques des captions
- **Nombre total de captions** : 43 670 (pour le train set)
- **Moyenne de captions par image** : ~5
- **Longueur moyenne des captions** : Variable (max observé ~36 tokens)
- **Taille du vocabulaire** : 1 434 mots (fréquence minimale : 3)

### Preprocessing
- **Résolution des images** : 224×224 pixels
- **Normalisation** : Moyenne et écart-type ImageNet
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Tokens spéciaux** : `<PAD>`, `<START>`, `<END>`, `<UNK>`


## 📈 Résultats d'évaluation

### Métriques sur le Test Set

| Métrique | Score | Description |
|----------|-------|-------------|
| **BLEU-1** | 0.6448 | Précision des unigrammes |
| **BLEU-2** | 0.4762 | Précision des bigrammes |
| **BLEU-3** | 0.3758 | Précision des trigrammes |
| **BLEU-4** | 0.3025 | Précision des 4-grammes |
| **METEOR** | 0.2605 | Métrique alignement sémantique |
| **ROUGE-L** | 0.4771 | Longest Common Subsequence |
| **CIDEr** | 0.8326 | Consensus-based metric |



## Entraînement

### Hyperparamètres
```python
Epochs: 50
Batch size: 64 (train), 128 (val), 1 (test)
Optimizer: Adam
  - Encoder LR: 1e-4
  - Decoder LR: 4e-4
Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
Loss function: CrossEntropyLoss (ignore PAD tokens)
Gradient clipping: max_norm=5.0
