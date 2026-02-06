from torch.optim import Optimizer
from torch.optim.optimizer import required


class GD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        self.lr = lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-group['lr'], d_p)
        return loss

    def adjust_learning_rate(self, round_i):
        lr = self.lr * (0.5 ** (round_i // 10))
        for param_group in self.param_groups:
            param_group['lr'] = lr
 
    def soft_decay_learning_rate(self, dataset, dirichlet):
        if dirichlet == 0.01 and dataset == 'cifar10':
            self.lr *= 0.96
        elif dirichlet == 0.01 and dataset == 'cifar100':
            self.lr *= 0.96
        elif dataset == 'cifar100':
            self.lr *= 0.98
        elif dataset == 'svhn':
            self.lr *= 0.98
        else:
            self.lr *= 0.9992
        # self.lr *= 0.99   
        for param_group in self.param_groups:
            param_group['lr'] = self.lr
    def soft_decay_learning_rate2(self):
        self.lr *= 0.95
        for param_group in self.param_groups:
            param_group['lr'] = self.lr
            
    def inverse_prop_decay_learning_rate(self, round_i):
        # self.lr *= 0.99
        # for param_group in self.param_groups:
        #     param_group['lr'] = self.lr
        for param_group in self.param_groups:
            param_group['lr'] = self.lr / (round_i + 1)

    def set_lr(self, lr):
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']


class LrdGD(GD):
    def __init__(self, params, lr=required, weight_decay=0):
        super(LrdGD, self).__init__(params, lr, weight_decay)

    def step(self, lr, closure=None):
        """Performs a single optimization step.

        Arguments:
            lr: learning rate
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-lr, d_p)
        return loss
