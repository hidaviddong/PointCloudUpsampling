import open3d as o3d
import numpy as np
import h5py
import torch
import MinkowskiEngine as ME
import os
import time
from torch.utils.data.sampler import Sampler


def loadh5(filedir, color_format='rgb'):
  """Load coords & feats from h5 file.

  Arguments: file direction

  Returns: coords & feats.
  """
  pc = h5py.File(filedir, 'r')['data'][:]

  coords = pc[:,0:3].astype('int32')

  if color_format == 'rgb':
    feats = pc[:,3:6]/255. 
  elif color_format == 'yuv':
    R, G, B = pc[:, 3:4], pc[:, 4:5], pc[:, 5:6]
    Y = 0.257*R + 0.504*G + 0.098*B + 16
    Cb = -0.148*R - 0.291*G + 0.439*B + 128
    Cr = 0.439*R - 0.368*G - 0.071*B + 128
    feats = np.concatenate((Y,Cb,Cr), -1)/256.
  elif color_format == 'geometry':
    feats = np.expand_dims(np.ones(coords.shape[0]), 1)
  elif color_format == 'None':
    return coords
    
  feats = feats.astype('float32')

  return coords, feats

def loadply(filedir):
  """Load coords & feats from ply file.
  
  Arguments: file direction.
  
  Returns: coords & feats.
  """
  pcd = o3d.io.read_point_cloud(filedir)
  
  coords = np.asarray(pcd.points)
  feats = np.asarray(pcd.colors)
    
  return coords, feats

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path = self.files[idx]
        coords,feats = loadply(file_path)
        return (coords, feats)
        
class PartialPlyDataset(torch.utils.data.Dataset):
    def __init__(self, files, parts=5):
        """
        Initialize the dataset
        :param files: List of ply file paths
        :param parts: Number of parts to divide each file into
        """
        self.files = files
        self.parts = parts
        self.current_file_index = -1
        self.current_coords = None
        self.current_feats = None
#         print(f"Initializing dataset with {len(files)} files, each divided into {parts} parts")

    def __len__(self):
        """Return the total length of the dataset (number of files * parts per file)"""
        return len(self.files) * self.parts

    def __getitem__(self, idx):
        """
        Get the data for a specific index
        :param idx: Index
        :return: Tuple of (coords, feats)
        """
        file_index = idx // self.parts
        part = idx % self.parts

        if file_index != self.current_file_index or self.current_coords is None:
            self.load_new_file(file_index)

        total_points = len(self.current_coords)
        points_per_part = total_points // self.parts
        start = part * points_per_part
        end = (part + 1) * points_per_part if part < self.parts - 1 else total_points

#         print(f"Fetching part {part + 1}/{self.parts} of file {file_index + 1}/{len(self.files)}")
#         print(f"Point cloud range: {start} to {end}, total points: {end - start}")

        return (self.current_coords[start:end], self.current_feats[start:end])

    def load_new_file(self, file_index):
        """
        Load a new ply file
        :param file_index: Index of the file to load
        """
        self.current_file_index = file_index
        file_path = self.files[file_index]
#         print(f"Loading new file: {file_path}")
        self.current_coords, self.current_feats = loadply(file_path)
#         print(f"File loaded, total points: {len(self.current_coords)}")
        

def collate_pointcloud_fn(batch):
    coords, feats = batch[0]
    coords_batch = ME.utils.batched_coordinates([coords])
    feats_batch = torch.from_numpy(feats).float()
    return coords_batch, feats_batch

def make_data_loader(files, batch_size, shuffle, num_workers, repeat):
    
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_pointcloud_fn, 
        'pin_memory': True,
        'drop_last': False
    }
    
    
    dataset = PartialPlyDataset(files,2)
#     dataset = Dataset(files)
    
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    
    loader = torch.utils.data.DataLoader(dataset, **args)
    
    return loader

