import torch
from tqdm import tqdm
from utils import get_entropy_layer

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, hooks, save_input, args):

    model.train()
    running_loss=0
    running_real_entropy_for_each_layer = torch.zeros(len(hooks.keys()))
    correct=0
    total=0

    ## Boolean for tensor reduction (Here always true, but can be False for LeNet network)
    bool_mean=False
    model_class_name = model.__class__.__name__
    if model_class_name == "ResNet" or model_class_name == "ResNet_orig" or model_class_name == 'SwinTransformer' or model_class_name == 'MobileNetV2' or model.__class__.__name__ == "VGG":
        bool_mean=True

    for data in tqdm(train_loader):
        inputs,labels=data[0].to(device),data[1].to(device)
        outputs=model(inputs)

        ## Hook filled!
        ## Let's calculate the entropy of each layer:
        real_entropy_for_each_layer = torch.zeros(len(hooks.keys()))
        for k in range(len(hooks.keys())):
            # mean over all the neurons of the layer
            real_entropy_for_each_layer[k] = torch.mean(get_entropy_layer(save_input.inputs[k][0], bool_mean, model_class_name))

        ## Clear the Hook for Memory
        save_input.clear()

        loss =loss_fn(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        for l in range(len(running_real_entropy_for_each_layer)):
            running_real_entropy_for_each_layer[l]+= real_entropy_for_each_layer[l].item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)

        if args.dataset == "Flowers-102" or args.dataset == "DTD":
            correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
        else:
            correct += predicted.eq(labels).sum().item()
    
    train_loss=running_loss/len(train_loader)
    train_acc=100.*correct/total
    real_entropy_for_each_layer = running_real_entropy_for_each_layer/len(train_loader)
    return train_acc, train_loss, real_entropy_for_each_layer

def val_entropy_one_epoch(model, val_loader, loss_fn, device, hooks, save_input, args):

    model.eval()
    running_loss=0
    running_real_entropy_for_each_layer = torch.zeros(len(hooks.keys()))
    correct=0
    total=0

    ## Boolean for tensor reduction (Here always true, but can be False for LeNet network)
    bool_mean=False
    model_class_name = model.__class__.__name__
    if model_class_name == "ResNet" or model_class_name == "ResNet_orig" or model_class_name == 'SwinTransformer' or model_class_name == 'MobileNetV2' or model.__class__.__name__ == "VGG":
        bool_mean=True

    with torch.no_grad():
        for data in tqdm(val_loader):
            inputs,labels=data[0].to(device),data[1].to(device)
            outputs=model(inputs)

            ## Hook filled!
            real_entropy_for_each_layer = torch.zeros(len(hooks.keys()))
            for k in range(len(hooks.keys())):
                # mean over all the neurons of the layer
                real_entropy_for_each_layer[k] = torch.mean(get_entropy_layer(save_input.inputs[k][0], bool_mean, model_class_name)) 

            save_input.clear()

            loss =loss_fn(outputs,labels)
            
            running_loss += loss.item()

            for l in range(len(running_real_entropy_for_each_layer)):
                running_real_entropy_for_each_layer[l]+= real_entropy_for_each_layer[l].item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            if args.dataset == "Flowers-102" or args.dataset == "DTD":
                correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
            else:
                correct += predicted.eq(labels).sum().item()
        
    val_loss=running_loss/len(val_loader)
    val_accu=100.*correct/total
    real_entropy_for_each_layer = running_real_entropy_for_each_layer/len(val_loader)

    return val_accu, val_loss, real_entropy_for_each_layer
