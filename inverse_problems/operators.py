import torch
import torch.nn.functional as F

class Inpainting:
    def __init__(self, mask): self.mask=mask
    def A(self,x): return x*self.mask

class SuperResolution:
    def __init__(self, scale): self.scale=scale
    def A(self,x):
        lr=F.interpolate(x,scale_factor=1/self.scale,mode='bilinear',align_corners=False)
        return lr

class DenoiseIdentity:
    def A(self,x): return x

class SparseObservation:
    def __init__(self, idx): self.idx=idx
    def A(self,xflat): return xflat[:,self.idx]
