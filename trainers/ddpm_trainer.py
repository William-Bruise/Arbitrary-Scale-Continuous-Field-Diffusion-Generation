from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets.image_folder_dataset import ImageFolderDataset
from models.continuous_diffusion import ContinuousFieldDenoiser


class GaussianDiffusion:
    def __init__(self, T=1000, device='cpu'):
        self.T=T
        betas=torch.linspace(1e-4,0.02,T,device=device)
        alphas=1-betas
        self.ac=torch.cumprod(alphas,0)

    def q_sample(self, x0, t, eps):
        a=self.ac[t][:,None,None]
        return a.sqrt()*x0 + (1-a).sqrt()*eps


def sample_coords(b, n, irregular_ratio=0.5, device='cpu'):
    r = torch.rand(1).item()
    if r < irregular_ratio:
        return torch.rand(b,n,2,device=device)
    s = int(n**0.5)
    ys,xs=torch.meshgrid(torch.linspace(0,1,s,device=device), torch.linspace(0,1,s,device=device), indexing='ij')
    c=torch.stack([xs.flatten(),ys.flatten()],-1)
    c=c[None].repeat(b,1,1)
    return c


def query_image(img, coords):
    g = coords*2-1
    g = g[:,None,:,:]
    out = F.grid_sample(img, g, mode='bilinear', align_corners=True)
    return out[:, :, 0, :].permute(0,2,1)


class Trainer:
    def __init__(self, cfg):
        self.cfg=cfg
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        ds=ImageFolderDataset(f"{cfg.data_root}/{cfg.dataset}", 'train', cfg.image_size)
        self.loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,drop_last=True)
        self.model=ContinuousFieldDenoiser(cfg.global_latent_dim,cfg.hidden_dim,cfg.depth).to(self.device)
        self.opt=torch.optim.AdamW(self.model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
        self.diff=GaussianDiffusion(cfg.diffusion_steps,self.device)

    def train(self):
        Path(self.cfg.output_dir,'checkpoints').mkdir(parents=True, exist_ok=True)
        step=0
        while step < self.cfg.steps:
            for x in self.loader:
                x=x.to(self.device)
                b=x.size(0)
                n=torch.randint(self.cfg.min_points,self.cfg.max_points+1,(1,)).item()
                coords_f=sample_coords(b,n,self.cfg.use_irregular_ratio,self.device)
                x0_f=query_image(x,coords_f)
                t=torch.randint(0,self.cfg.diffusion_steps,(b,),device=self.device)
                eps=torch.randn_like(x0_f)
                xt=self.diff.q_sample(x0_f,t,eps)
                g=self.model.encode_global(x)
                pred=self.model(xt,coords_f,t,g)
                loss_main=F.mse_loss(pred,eps)

                n2=max(self.cfg.min_points,n//4)
                coords_c=coords_f[:,:n2]
                x0_c=x0_f[:,:n2]; eps_c=eps[:,:n2]; xt_c=xt[:,:n2]
                pred_c=self.model(xt_c,coords_c,t,g)
                loss_cons=F.mse_loss(pred_c,eps_c)
                loss=loss_main+self.cfg.consistency_weight*loss_cons
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                step+=1
                if step%100==0: print(f"step={step} loss={loss.item():.4f}")
                if step%self.cfg.ckpt_every==0:
                    torch.save({'model':self.model.state_dict(),'cfg':self.cfg.__dict__}, Path(self.cfg.output_dir,'checkpoints',f'step_{step}.pt'))
                if step>=self.cfg.steps: break
        torch.save({'model':self.model.state_dict(),'cfg':self.cfg.__dict__}, Path(self.cfg.output_dir,'checkpoints','final.pt'))
