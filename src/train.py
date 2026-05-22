import argparse, os, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .dataset import make_dataset
from .continuous_field import ContinuousGaussianField, fit_coeffs_to_image
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

def estimate_coeff_stats(field, dl, device, max_batches=100):
    sums=sq_sums=None; count=0
    with torch.no_grad():
        for bi,(x,_) in enumerate(dl):
            if bi>=max_batches: break
            x=x.to(device); c=fit_coeffs_to_image(field,x).reshape(x.shape[0],-1)
            sums=c.sum(0) if sums is None else sums+c.sum(0)
            sq_sums=(c**2).sum(0) if sq_sums is None else sq_sums+(c**2).sum(0)
            count += c.shape[0]
    mean=sums/max(count,1); var=sq_sums/max(count,1)-mean**2
    return mean, torch.sqrt(var.clamp_min(1e-8))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='mnist'); p.add_argument('--data-root',default='./data'); p.add_argument('--image-size',type=int,default=32)
    p.add_argument('--epochs',type=int,default=10); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--timesteps',type=int,default=200); p.add_argument('--num-basis',type=int,default=144); p.add_argument('--sigma',type=float,default=0.08)
    p.add_argument('--device',default='auto', help='auto|cpu|cuda|cuda:0'); p.add_argument('--outdir',default='runs/full_train'); p.add_argument('--sample-every',type=int,default=500)
    p.add_argument('--ckpt-every',type=int,default=1000); p.add_argument('--num-workers',type=int,default=2); p.add_argument('--unet-base',type=int,default=64)
    p.add_argument('--normalize-coeffs',action='store_true'); p.add_argument('--stats-batches',type=int,default=100)
    args=p.parse_args()
    if args.device == 'auto':
      args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[device] using {args.device}')
    os.makedirs(f"{args.outdir}/samples",exist_ok=True); os.makedirs(f"{args.outdir}/checkpoints",exist_ok=True); os.makedirs(f"{args.outdir}/logs",exist_ok=True)
    ds=make_dataset(args.dataset,args.data_root,True,args.image_size); dl=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,drop_last=True)
    sample_x,_=next(iter(dl)); channels=sample_x.shape[1]
    field=ContinuousGaussianField(args.num_basis,args.sigma,channels,args.device).to(args.device)
    model=LatentUNetDenoiser(args.num_basis,channels,args.unet_base).to(args.device)
    diff=DDPMCoefficients(args.timesteps,device=args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    mean=torch.zeros(channels*args.num_basis,device=args.device); std=torch.ones(channels*args.num_basis,device=args.device)
    if args.normalize_coeffs: mean,std=estimate_coeff_stats(field,dl,args.device,args.stats_batches)
    step=0; log=f"{args.outdir}/logs/train_log.txt"
    for e in range(1,args.epochs+1):
      for x,_ in dl:
        x=x.to(args.device); coeff=fit_coeffs_to_image(field,x); flat=coeff.reshape(x.shape[0],-1)
        train=(flat-mean)/std if args.normalize_coeffs else flat
        t=torch.randint(0,args.timesteps,(x.shape[0],),device=args.device); noise=torch.randn_like(train); x_t=diff.q_sample(train,t,noise)
        pred=model(x_t,t); loss=((pred-noise)**2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step+=1
        if step%50==0 or step==1:
          msg=f"epoch={e}/{args.epochs} step={step} loss={loss.item():.6f} coeffs={tuple(train.shape)} x_t={tuple(x_t.shape)}"; print(msg); open(log,'a').write(msg+'\n')
        if step%args.sample_every==0:
          sc=diff.sample(model,1,channels*args.num_basis,args.device); sc=sc*std.unsqueeze(0)+mean.unsqueeze(0) if args.normalize_coeffs else sc
          save_multires(field,sc.reshape(1,channels,args.num_basis),f"{args.outdir}/samples/step_{step}_multires.png")
        if step%args.ckpt_every==0:
          torch.save({'model':model.state_dict(),'epoch':e,'steps':step,'dataset':args.dataset,'image_size':args.image_size,'channels':channels,'num_basis':args.num_basis,'sigma':args.sigma,'unet_base':args.unet_base,'timesteps':args.timesteps,'normalize_coeffs':args.normalize_coeffs,'coeff_mean':mean.cpu(),'coeff_std':std.cpu()},f"{args.outdir}/checkpoints/step_{step}.pt")
    torch.save({'model':model.state_dict(),'epoch':args.epochs,'steps':step,'dataset':args.dataset,'image_size':args.image_size,'channels':channels,'num_basis':args.num_basis,'sigma':args.sigma,'unet_base':args.unet_base,'timesteps':args.timesteps,'normalize_coeffs':args.normalize_coeffs,'coeff_mean':mean.cpu(),'coeff_std':std.cpu()},f"{args.outdir}/checkpoints/final.pt")

if __name__=='__main__': main()
