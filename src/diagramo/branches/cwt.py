"""CWT: original seven-channel normalized Morlet log-magnitude images."""
from diagramo.features import cwt as transform

ENCODERS = ('resnet50', 'efficientnet_b4', 'vit_b_16', 'vgg16')
FUSION_ENCODER = 'efficientnet_b4'
