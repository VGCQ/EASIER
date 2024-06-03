import torch
import torchvision
from torchvision import datasets, transforms
import os
import numpy as np
from PIL import ImageFile


def get_pacs(args, get_train_sampler = False, transform_train = True):
    
    data_path = args.root

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Randomly cropping the images
        transforms.RandomHorizontalFlip(),  # Randomly apply horizontal flipping
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),  # Random color jittering
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [torchvision.datasets.ImageFolder(os.path.join(data_path, domain), transform=transform) 
                    for domain in ['cartoon', 'art_painting', 'photo', 'sketch']]
    dataset = torch.utils.data.ConcatDataset(datasets)

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_split = int(0.8 * len(dataset))  # 80% for training
    val_test_split = int(0.9 * len(dataset))  # 10% for validation, 10% for testing
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_test_split]
    test_indices = indices[val_test_split:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    num_classes = len(dataset.datasets[0].classes)

    dataset_labels = dataset.datasets[0].classes
    
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
    
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                  num_workers=args.workers, pin_memory=True, sampler=test_sampler,
                                                  persistent_workers=args.workers > 0)
    
    if(get_train_sampler):
        return train_dataloader, valid_dataloader, test_dataloader, train_sampler, num_classes, dataset_labels
    
    return train_dataloader, valid_dataloader, test_dataloader, num_classes, dataset_labels
