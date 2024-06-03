import torch 
from tqdm import tqdm 

def eval(model, test_loader, loss_fn, device, save_input):
    model.eval()
    
    running_loss=0
    correct=0
    total=0    
    with torch.no_grad():
        for data in tqdm(test_loader):
            images,labels=data[0].to(device),data[1].to(device)
            outputs=model(images)
            loss= loss_fn(outputs,labels)
            running_loss+=loss.item()     
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            save_input.clear()
    test_loss=running_loss/len(test_loader)
    accu=100.*correct/total

    print('Test Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))
    return(accu, test_loss)