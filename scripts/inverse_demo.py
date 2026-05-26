import argparse
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
from inverse_problems.operators import Inpainting, SuperResolution, DenoiseIdentity
from inverse_problems.posterior_sampling import posterior_guidance_step

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',required=True)
    ap.add_argument('--task',choices=['inpainting','sr','denoise'],default='denoise')
    ap.add_argument('--out',default='inverse_out.png')
    args=ap.parse_args()
    x=read_image(args.input).float()/255.0
    x=x.unsqueeze(0)
    if args.task=='inpainting':
        mask=torch.ones_like(x); mask[:,:,:,x.shape[-1]//3:2*x.shape[-1]//3]=0
        op=Inpainting(mask); y=op.A(x)
    elif args.task=='sr':
        op=SuperResolution(4); y=op.A(x)
    else:
        op=DenoiseIdentity(); y=x+0.05*torch.randn_like(x)
    z=torch.randn_like(x)
    for _ in range(50):
        z,_=posterior_guidance_step(z,y,op,sigma=0.05,step_size=0.1)
    save_image(z.clamp(0,1),args.out)
