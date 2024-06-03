import os 
import argparse
import torch
import torchvision
import random
import numpy as np
import re 

from models.ResNet import ResNet, BasicBlock, ResNet_orig
from models.Mobilenetv2 import MobileNetV2
from utils import SaveInput
from train import train_one_epoch, val_entropy_one_epoch
from test import eval
from dataloaders.cifar10 import get_cifar10
from dataloaders.tinyimagenet200 import get_tinyimagenet200
from dataloaders.pacs import get_pacs
from dataloaders.vlcs import get_vlcs
from dataloaders.aircraft import get_aircraft
from dataloaders.dtd import get_dtd
from dataloaders.flower import get_flowers102

def main():
    parser = argparse.ArgumentParser(description='EASIER')
    parser.add_argument('--model', default='ResNet-18',  help='model (default: ResNet-18)')
    parser.add_argument('--dataset', default='CIFAR-10',  help='dataset (default : CIFAR-10)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='seed (default: 0)')
    parser.add_argument('--optimizer', default="SGD",  help='Optimizer Adam/SGD (default: SGD)')
    parser.add_argument('--momentum', type=float, default=0.9,  help='Momentum (default: 0.9)')
    parser.add_argument('--epochs', type=int, default=160,  help='Epochs (default: 160)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size (default:128)')
    parser.add_argument('--lr', type=float, default=0.1,  help='Learning Rate (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.1,  help='Drop factor (default: 0.1)')
    parser.add_argument('--milestones', default="80,120",  type=lambda s: [int(item) for item in s.split(',')], help='Milestones (default: "80,120")')
    parser.add_argument('--wd', type=float, default=1e-4,  help='Weight decay (default: 1e-4)')
    parser.add_argument('--dir_to_save_checkpoint',type=str, default='checkpoints/', help="directory to save checkpoint (default:'checkpoints/')")
    parser.add_argument('--device', type=int, default=0, help='GPU id (default: 0)')
    parser.add_argument('--root', default="data/", help='root of dataset (default:"data/")')
    parser.add_argument('--distributed', default=False, help='distributed (default : False)')
    parser.add_argument('--mixup_alpha', type=float, default=0.8, help='mixup_alpha (default: 0.8)')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='cutmix_alpha (default: 1.0)')
    parser.add_argument('--use_v2', default=False, help='use_v2 (default: False)')
    parser.add_argument('--workers', type=int, default=10, help='workers (default:10)')
    args = parser.parse_args()
    
    ## Getting the desired GPU
    cuda = "cuda:"+str(args.device)
    device = torch.device(cuda)

    ## SEEDING
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":16:8"
    
    ## DATASET:
    if args.dataset == "CIFAR-10":
        train_loader, val_loader, test_loader, num_classes, _ = get_cifar10(args, size_train_set = 0.9, get_train_sampler = False, transform_train = True)
    elif args.dataset == "Tiny-ImageNet-200":
        train_loader, val_loader, test_loader, num_classes, _ = get_tinyimagenet200(args, get_train_sampler = False, transform_train = True)
    elif args.dataset == "PACS":
        train_loader, val_loader, test_loader, num_classes, _ = get_pacs(args, get_train_sampler = False, transform_train = True)
    elif args.dataset == "VLCS":
        train_loader, val_loader, test_loader, num_classes, _ = get_vlcs(args, get_train_sampler = False, transform_train = True)
    elif args.dataset == 'Flowers-102':
        train_loader, val_loader, test_loader, num_classes, _ = get_flowers102(args, get_train_sampler=False)
    elif args.dataset == 'DTD':
        train_loader, val_loader, test_loader, num_classes, _ = get_dtd(args, get_train_sampler=False)
    elif args.dataset == 'Aircraft':
        train_loader, val_loader, test_loader, num_classes, _ = get_aircraft(args, get_train_sampler=False)
    
    ## MODEL:
    if args.model == "ResNet-18":
        if args.dataset == "CIFAR-10" or args.dataset == "PACS" or args.dataset == "VLCS":
            model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
        elif args.dataset == "Tiny-ImageNet-200":
            model = ResNet_orig(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
        elif args.dataset == "Flowers-102" or args.dataset == "DTD" or args.dataset == "Aircraft":
            model = torchvision.models.resnet18(weights = True)
            model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)
            torch.save({'model_state_dict': model.state_dict()},"pretrained_ResNet-18.pt")
            del model  
            model = ResNet_orig(BasicBlock, [2, 2, 2, 2], num_classes=num_classes).to(device)
            checkpoint = torch.load("pretrained_ResNet-18.pt", map_location = device)
            model.load_state_dict(checkpoint['model_state_dict'])
            del checkpoint

    elif args.model == "MobileNetv2":
        if args.dataset == "CIFAR-10":
            model = MobileNetV2().to(device)
        else:
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            model.classifier[1] = torch.nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
            model.to(device)

    elif args.model == "Swin-T":
        model = torchvision.models.swin_t(weights = True).to(device)
        model.head = torch.nn.Linear(in_features=model.head.in_features, out_features=num_classes).to(device)

    elif args.model == "VGG16":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)
        model.classifier[6] = torch.nn.Linear(in_features =4096, out_features = num_classes )
        model.to(device)
        
    ## Loss
    loss_fn=torch.nn.CrossEntropyLoss()
        
    ## inplace = False is important to calculate the entropy with the Hook
    if args.model == "MobileNetv2":
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU6: ## MobileNetv2
                change_name= re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f'model.{change_name} = torch.nn.ReLU6(inplace=False)')
    elif args.model == 'VGG16' or args.model == "ResNet-18":
        for name, module in model.named_modules():
            if type(module) == torch.nn.ReLU: ## VGG16 or ResNet18
                change_name= re.sub(r'\.(\d+)', r'[\1]', name)
                exec(f'model.{change_name} = torch.nn.ReLU(inplace=False)')

    ## Hook
    save_input = SaveInput()
    hooks = {}
    for name, module in model.named_modules():
        if type(module) == torch.nn.ReLU: ## ResNet
            hooks[name] = module.register_forward_hook(save_input)

        elif type(module) == torch.nn.ReLU6: ## MobileNetv2
            hooks[name] = module.register_forward_hook(save_input)

        elif type(module) == torch.nn.GELU: ## Swin-T
            hooks[name] = module.register_forward_hook(save_input)

    
    ## Iterating over the number of activation
    number_of_activation = len(hooks.keys())
    for number_of_possible_replacement in range(number_of_activation):
        
        name_run = "EASIER_"+args.dataset+"_"+args.model+"_iteration"+str(number_of_possible_replacement)
        
        if args.optimizer == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        ## TRAINING:
        for epoch in range(1,args.epochs+1):
            train_acc, train_loss, entropy_for_each_layer = train_one_epoch(model, train_loader, optimizer, loss_fn, device, hooks, save_input, args)
            val_acc, val_loss, val_entropy_for_each_layer = val_entropy_one_epoch(model, val_loader, loss_fn, device, hooks, save_input, args)
            test_acc, test_loss = eval(model, test_loader, loss_fn, device, save_input)

            torch.save({               
                    'epoch': epoch,
                    'train_loss':train_loss, 'train_acc':train_acc,
                    'val_loss':val_loss, 'val_acc':val_acc,
                    'test_loss':test_loss, 'test_acc':test_acc, 
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},os.path.join(args.dir_to_save_checkpoint,"checkpoint_"+name_run+"_last_epoch.pt"))
            
            if args.optimizer == "SGD":
                scheduler.step()

        ## END OF TRAINING: REMOVE ONE ACTIVATION
        
        ## Put lowest entropy to zero to remove this layer
        entropy_for_each_layer[np.argmin(entropy_for_each_layer)]=0

        ## Check if a layer is removable and change rectifier to Identity:
        layers_to_remove=[]
        hooks_to_remove=[]
        store_position_associated_with_key=[]
        for this_layer in range(len(hooks.keys())):
            if entropy_for_each_layer[this_layer]==0: 
                store_position_associated_with_key.append(this_layer)
                name = list(hooks.keys())[this_layer]
                hooks_to_remove.append(name)
                if args.model == "ResNet-18" or args.model == "MobileNetv2" or args.model == "VGG16":
                    change_name= re.sub(r'\.(\d+)', r'[\1]', name)
                elif args.model == "Swin-T":
                    change_name= re.sub(r'\.(\d+)\.(\d+)\.', r'[\1][\2].', name)
                    change_name= re.sub(r'\.(\d+)$', r'[\1]', change_name)
                layers_to_remove.append(change_name)
        for name in hooks_to_remove:
            hooks[name].remove()
            hooks.pop(name)
        for name in layers_to_remove:
            exec(f'model.{name} = torch.nn.Identity()')
        
        ## After removal, save model
        torch.save(model, os.path.join(args.dir_to_save_checkpoint,name_run+"_after_removal"))
   
if __name__ == '__main__':
    main()
