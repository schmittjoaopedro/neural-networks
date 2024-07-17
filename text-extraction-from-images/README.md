# Text Extraction from Images

Paper: [TextOCR: Towards large-scale end-to-end reasoning for arbitrary-shaped scene text](https://arxiv.org/abs/2105.05486)

Dataset: [Kaggle](https://www.kaggle.com/datasets/robikscube/textocr-text-extraction-from-images-dataset)


# Mask TextSpotter v3

General Architecture:
* RestNet-50 - backbone
* Segmentation Proposal Network (SPN) - text proposals (Segmentation Label Generation)
* Fast R-CNN - refining proposals
* Text Instance Segmentation Module - accurate detection
* Character Segmentation Module - recognition
* Spatial Attentional Model - recognition
