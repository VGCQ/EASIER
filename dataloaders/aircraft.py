import torch
from torchvision import datasets, transforms


def get_aircraft(args, valid_size = 0, get_train_sampler = False, transform_train = True):
    re_size = 512
    crop_size = 224
    if transform_train:
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    test_transform = transforms.Compose([
            transforms.Resize((re_size, re_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
    train_dataset = datasets.FGVCAircraft(root=args.root + "/FGVCAircraft/raw/",
                                 split='train',
                                 transform=train_transform,
                                 download=True)
    val_dataset = datasets.FGVCAircraft(root=args.root + "/FGVCAircraft/raw/",
                                 split='val',
                                 transform=test_transform,
                                 download=True)
    
    dataset_labels = train_dataset.classes
    num_classes = len(train_dataset.classes)
    

    train_sampler = None
    validation_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None and transform_train),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=args.workers > 0)

    valid_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, sampler=validation_sampler,
                                                   persistent_workers=args.workers > 0)
    

    test = datasets.FGVCAircraft(root=args.root + "/FGVCAircraft/raw/",
                            split='test',
                            transform=test_transform)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True, sampler=test_sampler,
                                                  persistent_workers=args.workers > 0)
    
    if(get_train_sampler):
        return train_dataloader, valid_dataloader, test_dataloader, train_sampler, num_classes, dataset_labels
    
    return train_dataloader, valid_dataloader, test_dataloader, num_classes, dataset_labels
