import torch

def posterior_guidance_step(x, y, op, sigma, step_size):
    x = x.detach().requires_grad_(True)
    pred = op.A(x)
    loss = ((pred-y)**2).mean()/(2*sigma*sigma)
    grad = torch.autograd.grad(loss,x)[0]
    return (x - step_size*grad).detach(), loss.item()
