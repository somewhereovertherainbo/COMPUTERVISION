
import os
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import v2
num_workers = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: v2.Compose,
    batch_size: int,
    num_workers: int = num_workers):
  
  """
  Creates training and testing DataLoaders

  Creates a PyTorch Dataset and then a DataLoader using the data from training and testing directories.

  Args:
    train_dir: Path to the directory containing training data
    test_dir: Path to the directory containing testing data
    transform: torchvision transforms to be performed on training and testing data
    batch_size: number of images to load in a single batch
    num_workers: An integer for number of workersper DataLoader

  Returns:
    A Tuple of (train_dataloader, test_dataloader, class_names)
    Where class_names is a list of target class
    Example usage:
      train_dataloadaer, test_dataloader, class_names = create_dataloaders(train_dir = path/to/train_dir,
                                                                           test_dir = path/to/test_dir,
                                                                           transform = v2.Compose([]),
                                                                           batch_size = 32,
                                                                           num_workers = 2)


  """
  train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=transform, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

  test_data = datasets.ImageFolder(root=test_dir,
                                  transform=transform)

  class_names = train_data.classes

  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True,
                                pin_memory = True) 

  test_dataloader = DataLoader(dataset=test_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              pin_memory = True)

  return train_dataloader, test_dataloader, class_names
