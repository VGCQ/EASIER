import torch
from torchvision import datasets, transforms
from transforms import get_mixup_cutmix
from torch.utils.data.dataloader import default_collate



def get_dtd(args, get_train_sampler = False, transform_train = True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    spatial_size = 224

    if transform_train:
        train_transforms = transforms.Compose([
                    transforms.Resize(9*spatial_size//8),
                    # transforms.RandomCrop(spatial_size),
                    transforms.RandomResizedCrop(spatial_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
    else:
        train_transforms = transforms.Compose([
                        transforms.Resize(9*spatial_size//8),
                        transforms.CenterCrop(spatial_size),
                        transforms.ToTensor(),
                        normalize,
                    ])
        
    test_transforms = transforms.Compose([
                        transforms.Resize(9*spatial_size//8),
                        transforms.CenterCrop(spatial_size),
                        transforms.ToTensor(),
                        normalize,
                    ])

    
    train = datasets.DTD(root=args.root + "/DTD/raw/",
                                split="train",
                                transform=train_transforms,
                                download=True)
    
    dataset_labels = train.classes
    num_classes = len(train.classes)

    valid = datasets.DTD(root=args.root + "/DTD/raw/",
                                split="val",
                                transform=test_transforms)
    
    

    train_sampler = None
    validation_sampler = None
    test_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train)
        validation_sampler = torch.utils.data.distributed.DistributedSampler(valid, shuffle=False, drop_last=True)

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate


    
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                                   num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                   persistent_workers=args.workers > 0, collate_fn=collate_fn if transform_train else None)

    valid_dataloader = torch.utils.data.DataLoader(valid, batch_size=args.batch_size, shuffle=False,
                                                   num_workers=args.workers, pin_memory=True, sampler=validation_sampler,
                                                   persistent_workers=args.workers > 0, collate_fn=collate_fn if transform_train else None)
    

    test = datasets.DTD(root=args.root + "/DTD/raw/",
                            split="test",
                            transform=test_transforms)
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True, sampler=test_sampler,
                                                  persistent_workers=args.workers > 0)
    
    if(get_train_sampler):
        return train_dataloader, valid_dataloader, test_dataloader, train_sampler, num_classes, dataset_labels
    
    return train_dataloader, valid_dataloader, test_dataloader, num_classes, dataset_labels
