import argparse, os, torch, matplotlib.pyplot as plt
from .continuous_field import ContinuousGaussianField
from .model import LatentUNetDenoiser
from .diffusion import DDPMCoefficients

def save_multires(field, coeff, out_path, sizes=(28,42,56,84)):
    imgs=[field.render(coeff,s,s)[0].detach().cpu() for s in sizes]
    fig,axes=plt.subplots(len(imgs[0]),len(sizes),figsize=(3*len(sizes),3*len(imgs[0])))
    if imgs[0].shape[0]==1: axes=[axes]
    for r in range(len(imgs[0])):
        for c,(im,s) in enumerate(zip(imgs,sizes)):
            axes[r][c].imshow(im[r], cmap='gray' if im.shape[0]==1 else None, vmin=0, vmax=1)
            axes[r][c].set_title(f"{s}x{s} ch{r}"); axes[r][c].axis('off')
    plt.tight_layout(); fig.savefig(out_path); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True); p.add_argument('--outdir',default='runs/sample'); p.add_argument('--device',default='cpu'); a=p.parse_args()
    os.makedirs(a.outdir,exist_ok=True); ck=torch.load(a.ckpt,map_location=a.device)
    ch=ck.get('channels',1); k=ck['num_basis']
    field=ContinuousGaussianField(k,ck.get('sigma',0.08),ch,a.device).to(a.device)
    model=LatentUNetDenoiser(k,ch,ck.get('unet_base',64)).to(a.device); model.load_state_dict(ck['model']); model.eval()
    diff=DDPMCoefficients(ck['timesteps'],device=a.device)
    coeff=diff.sample(model,1,ch*k,a.device)
    if ck.get('normalize_coeffs',False): coeff=coeff*ck['coeff_std'].to(a.device).unsqueeze(0)+ck['coeff_mean'].to(a.device).unsqueeze(0)
    save_multires(field,coeff.reshape(1,ch,k),f"{a.outdir}/sample_multires.png")

if __name__=='__main__': main()
