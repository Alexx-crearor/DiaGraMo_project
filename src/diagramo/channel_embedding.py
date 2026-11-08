"""Label-free adapters and aggregation, with no fitting on the study cohort."""
import torch
from torch import nn


class FixedChannelAdapter(nn.Module):
    """Fixed 1x1 convolution: geometry / contact / orientation-and-time.

    Channel order: x, y, pressure, tilt, azimuth, time, pen_status.
    Every input participates; each RGB output is a convex average in [0,1].
    This is a prespecified lossy map, not a learned or invertible adapter.
    """
    def __init__(self):
        super().__init__()
        matrix=torch.zeros(3,7,1,1)
        for row,indices in enumerate(((0,1),(2,6),(3,4,5))):
            matrix[row,list(indices),0,0]=1/len(indices)
        self.register_buffer('weight',matrix)

    def forward(self,images):
        if images.ndim!=4 or images.shape[1]!=7:
            raise ValueError('Expected B x 7 x H x W images')
        return torch.nn.functional.conv2d(images,self.weight)


def channel_rgb(images):
    """One separate grayscale RGB image per channel, task-major/channel-minor."""
    if images.ndim!=4 or images.shape[1]!=7:
        raise ValueError('Expected B x 7 x H x W images')
    return images.reshape(-1,1,*images.shape[-2:]).repeat(1,3,1,1)


def aggregate_channel_embeddings(features,batch_size):
    """Keep channels separate by concatenation; mean is a matched-cost control."""
    if features.ndim!=2 or features.shape[0]!=batch_size*7:
        raise ValueError('Expected seven channel embeddings per task')
    separate=features.reshape(batch_size,7,-1)
    return separate.reshape(batch_size,-1),separate.mean(1)
