import torch

## Hook function to store the inputs of a given module
class SaveInput:
    def __init__(self):
        self.inputs = []

    def __call__(self, module, module_in, module_out):
        self.inputs.append(module_in)

    def clear(self):
        self.inputs = []

## Function calculating the entropy for a given layer
def get_entropy_layer(x, bool, model_class_name):
    
    x_positive = x>=torch.tensor(0)
    x_positive_on_mini_batch = torch.sum(x_positive, dim=0)

    x_negative = x<torch.tensor(0)
    x_negative_on_mini_batch = torch.sum(x_negative, dim=0)

    proba_plus = x_positive_on_mini_batch/x_positive.shape[0]
    proba_minus = x_negative_on_mini_batch/x_negative.shape[0]

    if bool:
        ## The tensors are not stored in the same way between Swin and the others.
        if  model_class_name == "SwinTransformer":
            while len(proba_plus.shape) > 1:
                proba_plus = torch.mean(proba_plus,dim=0)
                proba_minus = torch.mean(proba_minus,dim=0)
        else:
            while len(proba_plus.shape) > 1:
                proba_plus = torch.mean(proba_plus,dim=1)
                proba_minus = torch.mean(proba_minus,dim=1)

    ## Shape : [Number of Neurons]
    entropy = -proba_plus*torch.log2(torch.clamp(proba_plus, min=1e-10))-proba_minus*torch.log2(torch.clamp(proba_minus, min=1e-10))
    
    return(entropy)
