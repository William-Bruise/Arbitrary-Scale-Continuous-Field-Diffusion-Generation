import argparse
from pathlib import Path
import torch
from torchvision.utils import save_image
from models.continuous_diffusion import ContinuousFieldDenoiser
from trainers.ddpm_trainer import GaussianDiffusion, sample_coords


def denoise(model,diff,coords,steps,seed,device):
    torch.manual_seed(seed)
    b=1;n=coords.size(1)
    x=torch.randn(b,n,3,device=device)
    g=torch.randn(b, model.global_encoder[-1].out_features, device=device)
    for i in reversed(range(steps)):
        t=torch.full((b,),i,device=device,dtype=torch.long)
        eps=model(x,coords,t,g)
        a=diff.ac[t][:,None,None]
        x0=(x-(1-a).sqrt()*eps)/(a.sqrt()+1e-8)
        if i>0:
            z=torch.randn_like(x)
            a_prev=diff.ac[t-1][:,None,None]
            x=a_prev.sqrt()*x0 + (1-a_prev).sqrt()*z
        else:
            x=x0
    return x

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--outdir',default='runs/sample')
    ap.add_argument('--resolutions',default='64x64,128x128,192x96')
    ap.add_argument('--seed',type=int,default=0)
    args=ap.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    ck=torch.load(args.ckpt,map_location=device)
    cfg=ck['cfg']
    model=ContinuousFieldDenoiser(cfg['global_latent_dim'],cfg['hidden_dim'],cfg['depth']).to(device)
    model.load_state_dict(ck['model']); model.eval()
    diff=GaussianDiffusion(cfg['diffusion_steps'],device)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for rs in args.resolutions.split(','):
            h,w=[int(x) for x in rs.split('x')]
            n=h*w
            coords=sample_coords(1,n,0.0,device)
            vals=denoise(model,diff,coords,cfg['diffusion_steps'],args.seed,device)
            img=vals.reshape(1,h,w,3).permute(0,3,1,2).clamp(-1,1)
            save_image((img+1)/2, Path(args.outdir,f'seed{args.seed}_{h}x{w}.png'))
