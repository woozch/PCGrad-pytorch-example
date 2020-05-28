import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import random
from itertools import accumulate


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.freeze = nn.Linear(10, 10)  # freeze module
        self.base1 = nn.Linear(10, 10)
        self.task1 = nn.Linear(10, 2)
        self.task2 = nn.Linear(10, 2)

        for p in self.freeze.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = F.relu(self.freeze(x), inplace=True)
        x = F.relu(self.base1(x), inplace=True)
        t1 = self.task1(x)
        t2 = self.task2(x)
        return t1, t2


def normal_backward(net, optimizer, X, y, loss_layer=nn.CrossEntropyLoss()):
    num_tasks = len(y)  # T
    losses = []
    for i in range(num_tasks):
        optimizer.zero_grad()
        result = net(X)
        loss = loss_layer(result[i], y[i])
        losses += [loss, ]

    tot_loss = sum(losses)
    tot_loss.backward()
    return losses


def PCGrad_backward(net, optimizer, X, y, loss_layer=nn.CrossEntropyLoss()):
    grads_task = []
    grad_shapes = [p.shape if p.requires_grad is True else None
                   for group in optimizer.param_groups for p in group['params']]
    grad_numel = [p.numel() if p.requires_grad is True else 0
                  for group in optimizer.param_groups for p in group['params']]
    num_tasks = len(y)  # T
    losses = []

    # calculate gradients for each task
    for i in range(num_tasks):
        optimizer.zero_grad()
        result = net(X)
        loss = loss_layer(result[i], y[i])
        losses.append(loss)
        loss.backward()

        devices = [
            p.device for group in optimizer.param_groups for p in group['params']]

        grad = [p.grad.detach().clone().flatten() if (p.requires_grad is True and p.grad is not None)
                else None for group in optimizer.param_groups for p in group['params']]

        # fill zero grad if grad is None but requires_grad is true
        grads_task.append(torch.cat([g if g is not None else torch.zeros(
            grad_numel[i], device=devices[i]) for i, g in enumerate(grad)]))

    # shuffle gradient order
    random.shuffle(grads_task)

    # gradient projection
    grads_task = torch.stack(grads_task, dim=0)  # (T, # of params)
    proj_grad = grads_task.clone()

    def _proj_grad(grad_task):
        for k in range(num_tasks):
            inner_product = torch.sum(grad_task*grads_task[k])
            proj_direction = inner_product / (torch.sum(
                grads_task[k]*grads_task[k])+1e-12)
            grad_task = grad_task - torch.min(
                proj_direction, torch.zeros_like(proj_direction)) * grads_task[k]
        return grad_task

    proj_grad = torch.sum(torch.stack(
        list(map(_proj_grad, list(proj_grad)))), dim=0)  # (of params, )

    indices = [0, ] + [v for v in accumulate(grad_numel)]
    params = [p for group in optimizer.param_groups for p in group['params']]
    assert len(params) == len(grad_shapes) == len(indices[:-1])
    for param, grad_shape, start_idx, end_idx in zip(params, grad_shapes, indices[:-1], indices[1:]):
        if grad_shape is not None:
            param.grad[...] = proj_grad[start_idx:end_idx].view(grad_shape)  # copy proj grad

    return losses

if __name__ == '__main__':

    net = Net()
    net.train()
    optimizer = opt.SGD(net.parameters(), lr=0.01)
    num_task = 2
    num_iterations = 50000
    for it in range(num_iterations):
        X = torch.rand(20, 10) - \
            torch.cat([torch.zeros(10, 10), torch.ones(10, 10)])
        y = [torch.cat([torch.zeros(10,), torch.ones(10,)]).long(),
             torch.cat([torch.ones(10,), torch.zeros(10,)]).long()]
        losses = PCGrad_backward(net, optimizer, X, y)
        # losses = normal_backward(net, optimizer, X, y)
        optimizer.step()
        if it % 100 == 0:
            print("iter {} total loss: {}".format(
                it, sum([l.item() for l in losses])))
