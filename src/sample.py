import argparse, os, torch, matplotlib.pyplot as plt
from .continuous_field import ContinuousGaussianField
from .model import LatentUNetDenoiser
from .diffusion import DDPMCoefficients

def default_multiscales(base_size: int):
    return [base_size * m for m in (1, 2, 3, 4)]

def save_multires(field, coeff, out_path, base_size: int):
    sizes = default_multiscales(base_size)
    imgs=[field.render(coeff,s,s)[0].detach().cpu() for s in sizes]
    fig,axes=plt.subplots(1,len(sizes),figsize=(4*len(sizes),4))
    if len(sizes)==1: axes=[axes]
    for ax,im,s in zip(axes,imgs,sizes):
        if im.shape[0]==1:
            ax.imshow(im[0], cmap='gray', vmin=0, vmax=1)
        else:
            ax.imshow(im.permute(1,2,0).clamp(0,1))
        ax.set_title(f"{s}x{s}")
        ax.axis('off')
    plt.tight_layout(); fig.savefig(out_path); plt.close(fig)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ckpt',required=True); p.add_argument('--outdir',default='runs/sample'); p.add_argument('--device',default='cpu'); a=p.parse_args()
    os.makedirs(a.outdir,exist_ok=True); ck=torch.load(a.ckpt,map_location=a.device)
    ch=ck.get('channels',1); k=ck['num_basis']; base_size=ck.get('image_size',28)
    field=ContinuousGaussianField(k,ck.get('sigma',0.08),ch,a.device).to(a.device)
    model=LatentUNetDenoiser(k,ch,ck.get('unet_base',64)).to(a.device); model.load_state_dict(ck['model']); model.eval()
    diff=DDPMCoefficients(ck['timesteps'],device=a.device)
    coeff=diff.sample(model,1,ch*k,a.device)
    if ck.get('normalize_coeffs',False): coeff=coeff*ck['coeff_std'].to(a.device).unsqueeze(0)+ck['coeff_mean'].to(a.device).unsqueeze(0)
    save_multires(field,coeff.reshape(1,ch,k),f"{a.outdir}/sample_multires.png",base_size=base_size)

if __name__=='__main__': main()
