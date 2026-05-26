import argparse
from utils.config import load_config
from trainers.ddpm_trainer import Trainer

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',required=True)
    args=ap.parse_args()
    cfg=load_config(args.config)
    Trainer(cfg).train()
