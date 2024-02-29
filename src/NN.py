import torch.nn as nn

# TODO: implement reset method 

class SequentialNN(nn.Module):
    '''
    Implementation of a simple sequential Neural Network
    
    Attributes
    ----------
    topology : torch.nn.Sequential
        PyTorch Sequential module with linear layers (each one with its activation functions)
    '''
    
    def __init__(self, layers): # layers = [{'input': int, 'output':int, 'act_fun': 'fun_name'}]
        '''
        NN initialisation

        Parameter
        ---------
        layers: list
            the layers topology of the NN
            layers has the form: [{'input': int, 'output':int, 'act_fun': 'fun_name'}]
            e.g. 
            [{'input': 5, 'output': 10, 'act_fun': 'Sigmoid'}, {'input': 10, 'output': 3, 'act_fun': 'Identity'}]

        Returns
        -------
        return: -
        '''

        super(SequentialNN, self).__init__()

        nn_layers = []

        for layer in layers:
            nn_layers.append(nn.Linear(layer['input'], layer['output']))
            nn_layers.append(getattr(nn, layer['act_fun'])())

        self.topology = nn.Sequential(*nn_layers)

    def forward(self, x):
        '''
        Compute the Neural Network's output

        Parameter
        ---------
        x: tensor
            the input of the Neural Network

        Returns
        -------
        return: tensor
            return the output of the neural network
        '''
        
        return self.topology(x)