import torch
try:
    from apex import amp
except ImportError:
    pass

def loss_backward(args, loss, optimizer):
    if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
    if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

class FGM():
    def __init__(self, model, param_name, alpha=1.0):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha

    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                assert name in self.data
                param.data = self.data[name]
        self.data = {}

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.adversarial()
        loss = self.model(**inputs)[0]
        loss_backward(args, loss, optimizer)
        self.restore_param_data()


class PGD():
    def __init__(self, model, param_name, alpha=0.3, epsilon=1.0, K=3):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.K = K

    def backup_param_data(self):
        self.data = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.data[name] = param.data.clone()

    def restore_param_data(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.data = self.data[name]

    def backup_param_grad(self):
        self.grad = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                self.grad[name] = param.grad.clone()

    def restore_param_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                param.grad = self.grad[name]


    def adversarial(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.param_name in name:
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    perturbation = self.alpha * param.grad / norm
                    param.data.add_(perturbation)
                    param.data = self.project(name, param.data)

    def project(self, param_name, param_data):
        eta = torch.clamp(param_data - self.data[param_name])
        norm = torch.norm(eta)
        if norm > self.epsilon:
            eta = self.epsilon * eta / norm
        return self.data[param_name] + eta

    def adversarial_training(self, args, inputs, optimizer):
        self.backup_param_data()
        self.backup_param_grad()
        for k in range(self.K):
            self.adversarial()
            if k != self.K - 1:
                self.model.zero_grad()
            else:
                self.restore_param_grad()
            loss = self.model(**inputs)[0]
            loss_backward(args, loss, optimizer)
        self.restore_param_data()


class FreeAT():
    def __init__(self, model, param_name, alpha=0.3, epsilon=1.0, K=3):
        self.model = model
        self.param_name = param_name
        self.alpha = alpha
        self.epsilon = epsilon
        self.K = K
