# Details on the learning strategies

The training hyperparameters used in the experiments are presented in Table 4.

|  Model  |  Dataset   |  Epochs |  Batch |  Opt. |  Mom. |  LR |  Milestones |  Drop Factor |  Weight Decay |
|:---------------:|:-------------------:|:----------------:|:---------------:|:--------------:|:--------------:|:------------:|:--------------------:|:---------------------:|:----------------------:|
| ResNet-18     | CIFAR-10          | 160            | 128           | SGD          | 0.9          | 0.1        | [80, 120]          | 0.1                 | 1e-4                 |
| Swin-T        | CIFAR-10          | 160            | 128           | SGD          | 0.9          | 0.001      | [80, 120]          | 0.1                 | 1e-4                 |
| MobileNetv2   | CIFAR-10          | 160            | 128           | SGD          | 0.9          | 0.1        | [80, 120]          | 0.1                 | 1e-4                 |
| VGG-16        | CIFAR-10          | 160            | 128           | SGD          | 0.9          | 0.01       | [80, 120]          | 0.1                 | 1e-4                 |
| ResNet-18     | Tiny-ImageNet-200 | 160            | 128           | SGD          | 0.9          | 0.1        | [80, 120]          | 0.1                 | 1e-4                 |
| Swin-T        | Tiny-ImageNet-200 | 160            | 128           | SGD          | 0.9          | 0.001      | [80, 120]          | 0.1                 | 1e-4                 |
| MobileNetv2   | Tiny-ImageNet-200 | 160            | 128           | SGD          | 0.9          | 0.1        | [80, 120]          | 0.1                 | 1e-4                 |
| VGG-16        | Tiny-ImageNet-200 | 160            | 128           | SGD          | 0.9          | 0.01       | [80, 120]          | 0.1                 | 1e-4                 |
| ResNet-18     | PACS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| Swin-T        | PACS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| MobileNetv2   | PACS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| VGG-16        | PACS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| ResNet-18     | VLCS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| Swin-T        | VLCS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| MobileNetv2   | VLCS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| VGG-16        | VLCS              | 30             | 16            | SGD          | 0.9          | 0.001      | [24]               | 0.1                 | 5e-4                 |
| ResNet-18     | Flowers-102       | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| Swin-T        | Flowers-102       | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| MobileNetv2   | Flowers-102       | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| VGG-16        | Flowers-102       | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| ResNet-18     | DTD               | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| Swin-T        | DTD               | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| MobileNetv2   | DTD               | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| VGG-16        | DTD               | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| ResNet-18     | Aircraft          | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| Swin-T        | Aircraft          | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| MobileNetv2   | Aircraft          | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
| VGG-16        | Aircraft          | 50             | 16            | Adam         |              | 1e-4       |                    |                     | 0                    |
<p align="center">
   Table 4: The different employed learning strategies.
</p>

CIFAR-10 is augmented with per-channel normalization, random horizontal flipping, and random shifting by up to four pixels in any direction.
For the datasets of DomainBed, the images are augmented with per-channel normalization, random horizontal flipping, random cropping, and resizing to 224. The brightness, contrast, saturation, and hue are also randomly affected with a factor fixed to 0.4.
Tiny-ImageNet-200 is augmented with per-channel normalization and random horizontal flipping.
Moreover, the images of Flowers-102 are augmented with per-channel normalization, random horizontal and vertical flipping combined with a random rotation, and cropped to 224. DTD and Aircraft are augmented with random horizontal and vertical flipping, and with per-channel normalization.

Following [Liao et al.](https://openaccess.thecvf.com/content/ICCV2023W/RCV/papers/Liao_Can_Unstructured_Pruning_Reduce_the_Depth_in_Deep_Neural_Networks_ICCVW_2023_paper.pdf) and [Quétu et al.](https://arxiv.org/pdf/2303.01213), on CIFAR-10 and Tiny-ImageNet-200, all the models are trained for 160 epochs, optimized with SGD, having momentum 0.9, batch size 128, and weight decay 1e-4. The learning rate is decayed by a factor of 0.1 at milestones 80 and 120. The initial learning rate ranges from 0.1 for ResNet-18 and MobileNetv2, 0.01 for VGG-16 to 1e-3 for Swin-T.
Moreover, on PACS and VLCS, all the models are trained for 30 epochs, optimized with SGD, having momentum 0.9, a learning rate of 1e-3 decayed by a factor 0.1 at milestone 24, batch size 16, and weight decay 5e-4.
Furthermore, on Aircraft, DTD, and Flowers-102, all the models are trained following a transfer learning strategy. Indeed, each model is initialized with their pre-trained weights on ImageNet, trained for 50 epochs, optimized with Adam, having a learning rate 1e-4 and batch size 16.

For ResNet-18, MobileNetv2, and VGG-16 all the ReLU-activated layers are taken into account. For Swin-T, all the GELU-activated layers are considered.

The results for Layer Folding are obtained using the same aforementioned training policies, with the hyper-parameters declared in [Dror et al.](https://arxiv.org/pdf/2106.09309). Even if the method to find a good set of hyperparameters is not provided by the authors, for the datasets on which Layer Folding was not evaluated in the original work, we tried our best to select a good set of hyperparameters. Moreover, to allow a fair comparison with our method regarding the number of layers removed, we did not use their thresholding policy to select the number of removable layers. Instead, we have chosen to remove the top-k PReLU-activated layers having the most linear slope.

# Detailed Results

For each architecture, the test performance (top-1) and the number of removed layers (Rem.) obtained by EASIER for each iteration are displayed in Table 5 for CIFAR-10, in Table 6 for Tiny-ImageNet-200, in Table 7 for PACS, in Table 8 for VLCS, in Table 9 for Flowers-102, in Table 10 for DTD, in Table 11 for Aircraft.

|                                                                            | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:-------------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                                 |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|   | 92,47         | 0/17     | 91,66      | 0/12   | 93,65           | 0/35     | 93,50      | 0/15     |
|                                                                                 | 92,39         | 1/17     | 91,95      | 1/12   | 93,60           | 1/35     | 93,54      | 1/15     |
|                                                                                 | 92,82         | 2/17     | 91,84      | 2/12   | 93,76           | 2/35     | 93,44      | 2/15     |
|                                                                                 | 92,15         | 3/17     | 91,57      | 3/12   | 93,67           | 3/35     | 93,58      | 3/15     |
|                                                                                 | 92,47         | 4/17     | 91,73      | 4/12   | 93,26           | 4/35     | 93,51      | 4/15     |
|                                                                                 | 91,97         | 5/17     | 91,40      | 5/12   | 93,14           | 5/35     | 93,62      | 5/15     |
|                                                                                 | 92,22         | 6/17     | 91,25      | 6/12   | 93,22           | 6/35     | 93,42      | 6/15     |
|                                                                                 | 92,40         | 7/17     | 91,41      | 7/12   | 92,99           | 7/35     | 93,12      | 7/15     |
|                                                                                 | 92,10         | 8/17     | 90,83      | 8/12   | 93,16           | 8/35     | 93,61      | 8/15     |
|                                                                                 | 90,35         | 9/17     | 90,12      | 9/12   | 92,91           | 9/35     | 92,65      | 9/15     |
|                                                                                 | 89,94         | 10/17    | 87,79      | 10/12  | 92,79           | 10/35    | 93,04      | 10/15    |
|                                                                                 | 86,53         | 11/17    | 87,96      | 11/12  | 93,03           | 11/35    | 92,44      | 11/15    |
|                                                                                 | 84,52         | 12/17    |            |        | 93,16           | 12/35    | 91,46      | 12/15    |
|                                                                                 | 80,69         | 13/17    |            |        | 92,79           | 13/35    | 90,65      | 13/15    |
|                                                                                 | 80,95         | 14/17    |            |        | 92,79           | 14/35    | 90,38      | 14/15    |
|                                                                                 | 78,98         | 15/17    |            |        | 92,77           | 15/35    |            |          |
|                                                                                 | 77,74         | 16/17    |            |        | 92,45           | 16/35    |            |          |
|                                                                                 |               |          |            |        | 92,05           | 17/35    |            |          |
|                                                                                 |               |          |            |        | 91,80           | 18/35    |            |          |
|                                                                                 |               |          |            |        | 90,89           | 19/35    |            |          |
|                                                                                 |               |          |            |        | 90,79           | 20/35    |            |          |
|                                                                                 |               |          |            |        | 90,84           | 21/35    |            |          |
|                                                                                 |               |          |            |        | 89,92           | 22/35    |            |          |
|                                                                                 |               |          |            |        | 90,33           | 23/35    |            |          |
|                                                                                 |               |          |            |        | 90,44           | 24/35    |            |          |
|                                                                                 |               |          |            |        | 89,42           | 25/35    |            |          |
|                                                                                 |               |          |            |        | 88,98           | 26/35    |            |          |
|                                                                                 |               |          |            |        | 89,56           | 27/35    |            |          |
|                                                                                 |               |          |            |        | 88,74           | 28/35    |            |          |
|                                                                                 |               |          |            |        | 88,43           | 29/35    |            |          |
|                                                                                 |               |          |            |        | 87,19           | 30/35    |            |          |
|                                                                                 |               |          |            |        | 87,67           | 31/35    |            |          |
|                                                                                 |               |          |            |        | 86,88           | 32/35    |            |          |
|                                                                                 |               |          |            |        | 84,97           | 33/35    |            |          |
|                                                                                 |               |          |            |        | 62,56           | 34/35    |            |          |

<p align="center">
   Table 5: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on CIFAR-10.
</p>


|                                                                                      |  ResNet-18  |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:----------------------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                                          |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|   | 41,26         | 0/17     | 75,78      | 0/12   | 46,54           | 0/35     | 63,94      | 0/15     |
|                                                                                          | 41,10         | 1/17     | 70,94      | 1/12   | 47,52           | 1/35     | 63,32      | 1/15     |
|                                                                                          | 40,62         | 2/17     | 69,62      | 2/12   | 46,62           | 2/35     | 62,12      | 2/15     |
|                                                                                          | 41,46         | 3/17     | 68,46      | 3/12   | 48,22           | 3/35     | 61,34      | 3/15     |
|                                                                                          | 40,42         | 4/17     | 67,06      | 4/12   | 47,36           | 4/35     | 60,44      | 4/15     |
|                                                                                          | 40,42         | 5/17     | 66,68      | 5/12   | 47,52           | 5/35     | 59,06      | 5/15     |
|                                                                                          | 35,84         | 6/17     | 65,80      | 6/12   | 47,88           | 6/35     | 58,60      | 6/15     |
|                                                                                          | 36,30         | 7/17     | 63,90      | 7/12   | 48,44           | 7/35     | 57,60      | 7/15     |
|                                                                                          | 33,82         | 8/17     | 61,10      | 8/12   | 48,04           | 8/35     | 56,80      | 8/15     |
|                                                                                          | 34,04         | 9/17     | 56,76      | 9/12   | 47,76           | 9/35     | 56,04      | 9/15     |
|                                                                                          | 33,02         | 10/17    | 56,30      | 10/12  | 47,96           | 10/35    | 55,40      | 10/15    |
|                                                                                          | 31,58         | 11/17    | 52,26      | 11/12  | 47,58           | 11/35    | 53,18      | 11/15    |
|                                                                                          | 30,92         | 12/17    |            |        | 47,78           | 12/35    | 51,24      | 12/15    |
|                                                                                          | 33,88         | 13/17    |            |        | 48,00           | 13/35    | 48,02      | 13/15    |
|                                                                                          | 33,62         | 14/17    |            |        | 48,96           | 14/35    | 43,78      | 14/15    |
|                                                                                          | 33,90         | 15/17    |            |        | 48,88           | 15/35    |            |          |
|                                                                                          | 31,12         | 16/17    |            |        | 48,80           | 16/35    |            |          |
|                                                                                          |               |          |            |        | 48,54           | 17/35    |            |          |
|                                                                                          |               |          |            |        | 49,72           | 18/35    |            |          |
|                                                                                          |               |          |            |        | 49,50           | 19/35    |            |          |
|                                                                                          |               |          |            |        | 49,84           | 20/35    |            |          |
|                                                                                          |               |          |            |        | 50,14           | 21/35    |            |          |
|                                                                                          |               |          |            |        | 49,18           | 22/35    |            |          |
|                                                                                          |               |          |            |        | 49,88           | 23/35    |            |          |
|                                                                                          |               |          |            |        | 49,88           | 24/35    |            |          |
|                                                                                          |               |          |            |        | 49,92           | 25/35    |            |          |
|                                                                                          |               |          |            |        | 48,98           | 26/35    |            |          |
|                                                                                          |               |          |            |        | 49,16           | 27/35    |            |          |
|                                                                                          |               |          |            |        | 48,80           | 28/35    |            |          |
|                                                                                          |               |          |            |        | 45,70           | 29/35    |            |          |
|                                                                                          |               |          |            |        | 45,44           | 30/35    |            |          |
|                                                                                          |               |          |            |        | 45,30           | 31/35    |            |          |
|                                                                                          |               |          |            |        | 41,20           | 32/35    |            |          |
|                                                                                          |               |          |            |        | 35,50           | 33/35    |            |          |
|                                                                                          |               |          |            |        | 18,16           | 34/35    |            |          |

<p align="center">
   Table 6: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on Tiny-ImageNet-200.
</p>

|                                                                        | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:---------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                             |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|   | 79,70         | 0/17     | 97,10      | 0/12   | 95,60           | 0/35     | 95,40      | 0/15     |
|                                                                             | 86,60         | 1/17     | 95,90      | 1/12   | 95,30           | 1/35     | 95,70      | 1/15     |
|                                                                             | 86,40         | 2/17     | 95,90      | 2/12   | 95,40           | 2/35     | 95,20      | 2/15     |
|                                                                             | 88,30         | 3/17     | 93,80      | 3/12   | 94,80           | 3/35     | 95,20      | 3/15     |
|                                                                             | 89,00         | 4/17     | 94,30      | 4/12   | 95,20           | 4/35     | 95,50      | 4/15     |
|                                                                             | 87,90         | 5/17     | 94,60      | 5/12   | 94,50           | 5/35     | 94,20      | 5/15     |
|                                                                             | 87,70         | 6/17     | 92,50      | 6/12   | 94,50           | 6/35     | 93,00      | 6/15     |
|                                                                             | 88,30         | 7/17     | 91,80      | 7/12   | 94,40           | 7/35     | 93,00      | 7/15     |
|                                                                             | 87,30         | 8/17     | 91,50      | 8/12   | 94,20           | 8/35     | 91,70      | 8/15     |
|                                                                             | 88,30         | 9/17     | 84,80      | 9/12   | 93,70           | 9/35     | 90,20      | 9/15     |
|                                                                             | 87,70         | 10/17    | 85,90      | 10/12  | 93,80           | 10/35    | 91,40      | 10/15    |
|                                                                             | 86,80         | 11/17    | 86,20      | 11/12  | 93,60           | 11/35    | 89,90      | 11/15    |
|                                                                             | 87,10         | 12/17    |            |        | 92,70           | 12/35    | 87,40      | 12/15    |
|                                                                             | 84,30         | 13/17    |            |        | 92,10           | 13/35    | 85,90      | 13/15    |
|                                                                             | 80,90         | 14/17    |            |        | 92,20           | 14/35    | 85,50      | 14/15    |
|                                                                             | 75,10         | 15/17    |            |        | 92,40           | 15/35    |            |          |
|                                                                             | 52,00         | 16/17    |            |        | 88,60           | 16/35    |            |          |
|                                                                             |               |          |            |        | 89,80           | 17/35    |            |          |
|                                                                             |               |          |            |        | 89,60           | 18/35    |            |          |
|                                                                             |               |          |            |        | 89,00           | 19/35    |            |          |
|                                                                             |               |          |            |        | 91,00           | 20/35    |            |          |
|                                                                             |               |          |            |        | 89,10           | 21/35    |            |          |
|                                                                             |               |          |            |        | 88,30           | 22/35    |            |          |
|                                                                             |               |          |            |        | 88,40           | 23/35    |            |          |
|                                                                             |               |          |            |        | 87,90           | 24/35    |            |          |
|                                                                             |               |          |            |        | 86,40           | 25/35    |            |          |
|                                                                             |               |          |            |        | 86,30           | 26/35    |            |          |
|                                                                             |               |          |            |        | 87,60           | 27/35    |            |          |
|                                                                             |               |          |            |        | 87,30           | 28/35    |            |          |
|                                                                             |               |          |            |        | 86,60           | 29/35    |            |          |
|                                                                             |               |          |            |        | 79,50           | 30/35    |            |          |
|                                                                             |               |          |            |        | 67,10           | 31/35    |            |          |
|                                                                             |               |          |            |        | 63,30           | 32/35    |            |          |
|                                                                             |               |          |            |        | 62,90           | 33/35    |            |          |
|                                                                             |               |          |            |        | 43,80           | 34/35    |            |          |

<p align="center">
   Table 7: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on PACS.
</p>

|                                                                         | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:---------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                             |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|  | 68,13         | 0/17     | 83,04      | 0/12   | 81,36           | 0/35     | 82,76      | 0/15     |
|                                                                             | 70,83         | 1/17     | 82,01      | 1/12   | 79,50           | 1/35     | 81,64      | 1/15     |
|                                                                             | 72,04         | 2/17     | 81,45      | 2/12   | 79,59           | 2/35     | 81,92      | 2/15     |
|                                                                             | 71,48         | 3/17     | 79,87      | 3/12   | 77,82           | 3/35     | 79,59      | 3/15     |
|                                                                             | 71,39         | 4/17     | 78,75      | 4/12   | 78,56           | 4/35     | 80,99      | 4/15     |
|                                                                             | 71,67         | 5/17     | 78,19      | 5/12   | 78,10           | 5/35     | 78,94      | 5/15     |
|                                                                             | 71,20         | 6/17     | 79,12      | 6/12   | 77,17           | 6/35     | 78,84      | 6/15     |
|                                                                             | 71,11         | 7/17     | 76,51      | 7/12   | 76,79           | 7/35     | 78,66      | 7/15     |
|                                                                             | 70,92         | 8/17     | 78,19      | 8/12   | 77,82           | 8/35     | 78,29      | 8/15     |
|                                                                             | 71,85         | 9/17     | 78,19      | 9/12   | 78,19           | 9/35     | 78,56      | 9/15     |
|                                                                             | 71,30         | 10/17    | 75,68      | 10/12  | 76,14           | 10/35    | 76,51      | 10/15    |
|                                                                             | 71,02         | 11/17    | 73,63      | 11/12  | 76,61           | 11/35    | 75,77      | 11/15    |
|                                                                             | 70,83         | 12/17    |            |        | 76,23           | 12/35    | 75,49      | 12/15    |
|                                                                             | 69,80         | 13/17    |            |        | 78,56           | 13/35    | 76,05      | 13/15    |
|                                                                             | 70,27         | 14/17    |            |        | 76,51           | 14/35    | 70,18      | 14/15    |
|                                                                             | 54,24         | 15/17    |            |        | 76,79           | 15/35    |            |          |
|                                                                             | 55,27         | 16/17    |            |        | 74,84           | 16/35    |            |          |
|                                                                             |               |          |            |        | 74,18           | 17/35    |            |          |
|                                                                             |               |          |            |        | 74,74           | 18/35    |            |          |
|                                                                             |               |          |            |        | 73,44           | 19/35    |            |          |
|                                                                             |               |          |            |        | 75,58           | 20/35    |            |          |
|                                                                             |               |          |            |        | 73,90           | 21/35    |            |          |
|                                                                             |               |          |            |        | 72,88           | 22/35    |            |          |
|                                                                             |               |          |            |        | 73,07           | 23/35    |            |          |
|                                                                             |               |          |            |        | 74,74           | 24/35    |            |          |
|                                                                             |               |          |            |        | 72,69           | 25/35    |            |          |
|                                                                             |               |          |            |        | 73,07           | 26/35    |            |          |
|                                                                             |               |          |            |        | 72,88           | 27/35    |            |          |
|                                                                             |               |          |            |        | 71,48           | 28/35    |            |          |
|                                                                             |               |          |            |        | 72,88           | 29/35    |            |          |
|                                                                             |               |          |            |        | 71,85           | 30/35    |            |          |
|                                                                             |               |          |            |        | 71,02           | 31/35    |            |          |
|                                                                             |               |          |            |        | 71,76           | 32/35    |            |          |
|                                                                             |               |          |            |        | 64,96           | 33/35    |            |          |
|                                                                             |               |          |            |        | 52,47           | 34/35    |            |          |


<p align="center">
   Table 8: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on VLCS.
</p>

|                                                                                | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:----------------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                                    |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|  | 88,88         | 0/17     | 92,70      | 0/12   | 88,50           | 0/35     | 86,47      | 0/15     |
|                                                                                    | 88,23         | 1/17     | 93,10      | 1/12   | 89,56           | 1/35     | 86,32      | 1/15     |
|                                                                                    | 87,49         | 2/17     | 92,71      | 2/12   | 89,79           | 2/35     | 88,81      | 2/15     |
|                                                                                    | 87,53         | 3/17     | 88,93      | 3/12   | 90,31           | 3/35     | 88,32      | 3/15     |
|                                                                                    | 87,25         | 4/17     | 90,93      | 4/12   | 89,79           | 4/35     | 84,63      | 4/15     |
|                                                                                    | 84,57         | 5/17     | 88,89      | 5/12   | 89,82           | 5/35     | 85,69      | 5/15     |
|                                                                                    | 83,43         | 6/17     | 87,49      | 6/12   | 89,27           | 6/35     | 84,14      | 6/15     |
|                                                                                    | 63,41         | 7/17     | 85,92      | 7/12   | 89,02           | 7/35     | 82,60      | 7/15     |
|                                                                                    | 73,12         | 8/17     | 85,10      | 8/12   | 87,93           | 8/35     | 84,62      | 8/15     |
|                                                                                    | 64,58         | 9/17     | 80,39      | 9/12   | 88,49           | 9/35     | 81,53      | 9/15     |
|                                                                                    | 70,81         | 10/17    | 80,73      | 10/12  | 88,37           | 10/35    | 80,68      | 10/15    |
|                                                                                    | 72,22         | 11/17    | 82,05      | 11/12  | 87,02           | 11/35    | 77,46      | 11/15    |
|                                                                                    | 67,28         | 12/17    |            |        | 87,56           | 12/35    | 75,15      | 12/15    |
|                                                                                    | 65,33         | 13/17    |            |        | 86,60           | 13/35    | 74,84      | 13/15    |
|                                                                                    | 28,17         | 14/17    |            |        | 87,01           | 14/35    | 69,47      | 14/15    |
|                                                                                    | 32,85         | 15/17    |            |        | 86,26           | 15/35    |            |          |
|                                                                                    | 20,15         | 16/17    |            |        | 86,06           | 16/35    |            |          |
|                                                                                    |               |          |            |        | 83,56           | 17/35    |            |          |
|                                                                                    |               |          |            |        | 82,39           | 18/35    |            |          |
|                                                                                    |               |          |            |        | 82,83           | 19/35    |            |          |
|                                                                                    |               |          |            |        | 83,61           | 20/35    |            |          |
|                                                                                    |               |          |            |        | 82,92           | 21/35    |            |          |
|                                                                                    |               |          |            |        | 82,40           | 22/35    |            |          |
|                                                                                    |               |          |            |        | 82,92           | 23/35    |            |          |
|                                                                                    |               |          |            |        | 81,36           | 24/35    |            |          |
|                                                                                    |               |          |            |        | 81,14           | 25/35    |            |          |
|                                                                                    |               |          |            |        | 74,63           | 26/35    |            |          |
|                                                                                    |               |          |            |        | 68,30           | 27/35    |            |          |
|                                                                                    |               |          |            |        | 69,62           | 28/35    |            |          |
|                                                                                    |               |          |            |        | 68,08           | 29/35    |            |          |
|                                                                                    |               |          |            |        | 57,78           | 30/35    |            |          |
|                                                                                    |               |          |            |        | 58,74           | 31/35    |            |          |
|                                                                                    |               |          |            |        | 43,10           | 32/35    |            |          |
|                                                                                    |               |          |            |        | 40,97           | 33/35    |            |          |
|                                                                                    |               |          |            |        | 20,28           | 34/35    |            |          |


<p align="center">
   Table 9: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on Flowers-102.
</p>

|                                                                        | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:--------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                            |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|  | 60,53         | 0/17     | 67,50      | 0/12   | 64,41           | 0/35     | 64,20      | 0/15     |
|                                                                            | 61,97         | 1/17     | 70,05      | 1/12   | 63,09           | 1/35     | 64,57      | 1/15     |
|                                                                            | 61,22         | 2/17     | 67,02      | 2/12   | 62,93           | 2/35     | 64,73      | 2/15     |
|                                                                            | 62,02         | 3/17     | 66,54      | 3/12   | 62,39           | 3/35     | 64,57      | 3/15     |
|                                                                            | 59,73         | 4/17     | 63,67      | 4/12   | 62,82           | 4/35     | 63,62      | 4/15     |
|                                                                            | 58,99         | 5/17     | 62,23      | 5/12   | 63,67           | 5/35     | 59,84      | 5/15     |
|                                                                            | 57,55         | 6/17     | 58,99      | 6/12   | 63,83           | 6/35     | 59,95      | 6/15     |
|                                                                            | 53,40         | 7/17     | 58,94      | 7/12   | 62,45           | 7/35     | 58,30      | 7/15     |
|                                                                            | 49,26         | 8/17     | 58,19      | 8/12   | 61,17           | 8/35     | 58,19      | 8/15     |
|                                                                            | 52,29         | 9/17     | 57,18      | 9/12   | 62,29           | 9/35     | 56,01      | 9/15     |
|                                                                            | 50,48         | 10/17    | 52,45      | 10/12  | 60,11           | 10/35    | 54,10      | 10/15    |
|                                                                            | 52,61         | 11/17    | 53,19      | 11/12  | 61,97           | 11/35    | 49,41      | 11/15    |
|                                                                            | 49,68         | 12/17    |            |        | 60,80           | 12/35    | 47,29      | 12/15    |
|                                                                            | 41,86         | 13/17    |            |        | 60,74           | 13/35    | 47,34      | 13/15    |
|                                                                            | 32,55         | 14/17    |            |        | 60,59           | 14/35    | 41,54      | 14/15    |
|                                                                            | 33,35         | 15/17    |            |        | 61,01           | 15/35    |            |          |
|                                                                            | 18,94         | 16/17    |            |        | 61,54           | 16/35    |            |          |
|                                                                            |               |          |            |        | 61,60           | 17/35    |            |          |
|                                                                            |               |          |            |        | 59,15           | 18/35    |            |          |
|                                                                            |               |          |            |        | 60,96           | 19/35    |            |          |
|                                                                            |               |          |            |        | 59,89           | 20/35    |            |          |
|                                                                            |               |          |            |        | 57,02           | 21/35    |            |          |
|                                                                            |               |          |            |        | 57,07           | 22/35    |            |          |
|                                                                            |               |          |            |        | 52,07           | 23/35    |            |          |
|                                                                            |               |          |            |        | 53,83           | 24/35    |            |          |
|                                                                            |               |          |            |        | 55,64           | 25/35    |            |          |
|                                                                            |               |          |            |        | 54,20           | 26/35    |            |          |
|                                                                            |               |          |            |        | 43,88           | 27/35    |            |          |
|                                                                            |               |          |            |        | 43,67           | 28/35    |            |          |
|                                                                            |               |          |            |        | 43,83           | 29/35    |            |          |
|                                                                            |               |          |            |        | 32,18           | 30/35    |            |          |
|                                                                            |               |          |            |        | 32,50           | 31/35    |            |          |
|                                                                            |               |          |            |        | 30,90           | 32/35    |            |          |
|                                                                            |               |          |            |        | 28,62           | 33/35    |            |          |
|                                                                            |               |          |            |        | 14,89           | 34/35    |            |          |


<p align="center">
   Table 10: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on DTD.
</p>

|                                                                             | ResNet-18 |       | Swin-T |     | MobileNetv2 |       | VGG-16 |       |
|:-------------------------------------------------------------------------------:|:-------------:|:--------:|:----------:|:------:|:---------------:|:--------:|:----------:|:--------:|
|                                                                                 |  top-1      |  Rem.  |  top-1   |  Rem. |  top-1        |  Rem.  |  top-1    |  Rem.  |
|  | 73,36         | 0/17     | 76,39      | 0/12   | 73,36           | 0/35     | 75,85      | 0/15     |
|                                                                                 | 72,79         | 1/17     | 76,81      | 1/12   | 73,00           | 1/35     | 78,13      | 1/15     |
|                                                                                 | 71,59         | 2/17     | 75,97      | 2/12   | 73,69           | 2/35     | 76,69      | 2/15     |
|                                                                                 | 71,14         | 3/17     | 74,65      | 3/12   | 72,13           | 3/35     | 76,63      | 3/15     |
|                                                                                 | 70,03         | 4/17     | 76,15      | 4/12   | 72,55           | 4/35     | 75,70      | 4/15     |
|                                                                                 | 71,17         | 5/17     | 74,59      | 5/12   | 71,74           | 5/35     | 66,52      | 5/15     |
|                                                                                 | 71,14         | 6/17     | 74,74      | 6/12   | 71,20           | 6/35     | 69,70      | 6/15     |
|                                                                                 | 65,47         | 7/17     | 74,44      | 7/12   | 70,30           | 7/35     | 67,21      | 7/15     |
|                                                                                 | 61,36         | 8/17     | 72,13      | 8/12   | 68,98           | 8/35     | 65,32      | 8/15     |
|                                                                                 | 60,85         | 9/17     | 71,62      | 9/12   | 69,31           | 9/35     | 64,12      | 9/15     |
|                                                                                 | 63,19         | 10/17    | 69,76      | 10/12  | 68,92           | 10/35    | 56,83      | 10/15    |
|                                                                                 | 66,22         | 11/17    | 71,41      | 11/12  | 69,52           | 11/35    | 55,42      | 11/15    |
|                                                                                 | 62,89         | 12/17    |            |        | 69,16           | 12/35    | 57,82      | 12/15    |
|                                                                                 | 39,12         | 13/17    |            |        | 67,60           | 13/35    | 49,23      | 13/15    |
|                                                                                 | 35,85         | 14/17    |            |        | 67,60           | 14/35    | 52,57      | 14/15    |
|                                                                                 | 28,11         | 15/17    |            |        | 67,15           | 15/35    |            |          |
|                                                                                 | 15,63         | 16/17    |            |        | 63,97           | 16/35    |            |          |
|                                                                                 |               |          |            |        | 64,66           | 17/35    |            |          |
|                                                                                 |               |          |            |        | 63,46           | 18/35    |            |          |
|                                                                                 |               |          |            |        | 64,30           | 19/35    |            |          |
|                                                                                 |               |          |            |        | 62,26           | 20/35    |            |          |
|                                                                                 |               |          |            |        | 63,61           | 21/35    |            |          |
|                                                                                 |               |          |            |        | 62,68           | 22/35    |            |          |
|                                                                                 |               |          |            |        | 52,75           | 23/35    |            |          |
|                                                                                 |               |          |            |        | 55,57           | 24/35    |            |          |
|                                                                                 |               |          |            |        | 53,89           | 25/35    |            |          |
|                                                                                 |               |          |            |        | 51,67           | 26/35    |            |          |
|                                                                                 |               |          |            |        | 52,09           | 27/35    |            |          |
|                                                                                 |               |          |            |        | 51,85           | 28/35    |            |          |
|                                                                                 |               |          |            |        | 39,27           | 29/35    |            |          |
|                                                                                 |               |          |            |        | 40,08           | 30/35    |            |          |
|                                                                                 |               |          |            |        | 42,96           | 31/35    |            |          |
|                                                                                 |               |          |            |        | 22,17           | 32/35    |            |          |
|                                                                                 |               |          |            |        | 18,15           | 33/35    |            |          |
|                                                                                 |               |          |            |        | 6,69            | 34/35    |            |          |



<p align="center">
   Table 11: Test performance (top-1) and the number of removed layers (Rem.) of EASIER on Aircraft.
</p>
