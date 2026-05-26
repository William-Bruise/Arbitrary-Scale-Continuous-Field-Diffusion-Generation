import argparse
import torch
from metrics.consistency import overlap_consistency

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=16); args=ap.parse_args()
    a=torch.randn(args.n,3); b=a.clone();
    idx=torch.arange(args.n)
    print('overlap_l1', overlap_consistency(a,idx,b,idx).item())
