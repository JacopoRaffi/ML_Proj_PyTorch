import torch.optim as optim

class Training:
    def __init__(self, net):
        '''
        Implementation of a Moudle to train PyTorch neural network
    
        Attributes
        ----------
        nn : torch.nn.Module neural network
            PyTorch Sequential module with linear layers (each one with its activation functions)

        optim: torch.optim
            PyTorch optimizer, can be SGD or Adamax
        
        lr_scheduler: torch.lr_scheduler.LinearLR
            PyTorch scheduler for linear decay of learning rate

        '''

        self.nn = net
        self.optim = None
        self.lr_scheduler = None

    def train(self, train_data, train_labels, val_data, val_labels, 
              batch_size, loss, max_epochs):
        pass

    def set_SGD_hyperparameters(self, lr:float, momentum:float, weight_decay:float, nesterov:bool, lr_decay_step:int = 0):
        '''
        Set hyperparameters for SGD training

        Parameter
        ---------
        lr: float
            learning rate 
        momentum: float
            momentum factor
        weight_decay: float
            Lambda Tikhonov regularization (L2 penality)
        nesterov: bool
            enable Nesterov momentum
        lr_decay_step: float
            Number of steps for Linear Decay of the learning rate

        Returns
        -------
        return: -

        '''

        self.optim = optim.SGD(self.nn.parameters(), lr=lr, momentum=momentum, 
                               weight_decay=weight_decay, nesterov=nesterov)
        
        if lr_decay_step:
            self.scheduler = optim.lr_scheduler.LinearLR(self.optim, start_factor=1, end_factor=0.01, total_iters=lr_decay_step)


    def set_Adamax_hyperparameters(self, lr, betas, eps, weight_decay):
        '''
        Set hyperparameters for SGD training

        Parameter
        ---------
        lr: float
            learning rate 
        betas: Tuple[float, float]
            coefficients used for computing running averages of gradient and its square
        eps: float
            term added to the denominator to improve numerical stability
        weight_decay: float
            Lambda Tikhonov regularization (L2 penality)

        Returns
        -------
        return: -

        '''
        self.optim = optim.Adamax(self.nn.parameters(), lr, betas, eps, weight_decay)
