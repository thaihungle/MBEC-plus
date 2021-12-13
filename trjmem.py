import torch
import torch.optim as optim
from torch.nn import Parameter
import pyflann
import numpy as np
import random

def inverse_distance(d, epsilon=1e-3):
    return 1 / (d + epsilon)

class TrjMemory:
    def __init__(self, mem_dim, num_neighbors, max_memory, lr):
        self.mem_dim = mem_dim
        self.num_neighbors = num_neighbors
        self.max_memory = max_memory
        self.lr = lr
        self.keys = None
        self.values = None
        pyflann.set_distance_type('euclidean')  # squared euclidean actually
        self.kdtree = pyflann.FLANN()
        # key_cache stores a cache of all keys that exist in the memory
        # This makes memory updates efficient
        self.key_cache = {}
        # stale_index is a flag that indicates whether or not the index in self.kdtree is stale
        # This allows us to only rebuild the kdtree index when necessary
        self.stale_index = True
    

        # Keys and value to be inserted into self.keys and self.values when commit_insert is called
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None

     

    def get_mem_size(self):
        if self.keys is not None:
            return len(self.keys)
        return 0

    def get_index(self, key):
        """
        If key exists in the memory, return its index
        Otherwise, return None
        """
        # print(key.data.cpu().numpy().shape)
        if self.key_cache.get(tuple(key.detach().cpu().numpy()[0])) is not None:
            if self.stale_index:
                self.commit_insert()
            return int(self.kdtree.nn_index(key.detach().cpu().numpy(), 1)[0][0])
        else:
            return None

    def update(self, value, index):
        """
        Set self.values[index] = value
        """
        self.values[index] = value

    def insert(self, key, value):
        """
        Insert key, value pair into memory (do not support batch insert)
        """
        if torch.cuda.is_available():
            if self.keys_to_be_inserted is None:
                # Initial insert
                self.keys_to_be_inserted = key.clone().detach().cuda()
                self.values_to_be_inserted = value.clone().detach().cuda()
            else:
                self.keys_to_be_inserted = torch.cat(
                    [self.keys_to_be_inserted.cuda(), key.clone().detach().cuda()], 0)
                self.values_to_be_inserted = torch.cat(
                    [self.values_to_be_inserted.cuda(), value.clone().detach().cuda()], 0)
        else:
            if self.keys_to_be_inserted is None:
                # Initial insert
                self.keys_to_be_inserted = key.clone().detach()
                self.values_to_be_inserted = value.clone().detach()
            else:
                self.keys_to_be_inserted = torch.cat(
                    [self.keys_to_be_inserted, key.clone().detach()], 0)
                self.values_to_be_inserted = torch.cat(
                    [self.values_to_be_inserted, value.clone().detach()], 0)
        self.key_cache[tuple(key.detach().cpu().numpy()[0])] = 0
        self.stale_index = True

    def commit_insert(self):
   
        if self.keys is None:
            self.keys = self.keys_to_be_inserted
            self.values = self.values_to_be_inserted
        elif self.keys_to_be_inserted is not None:
            self.keys = torch.cat([self.keys, self.keys_to_be_inserted], 0)
            self.values = torch.cat([self.values, self.values_to_be_inserted], 0)



        if len(self.keys) > self.max_memory:
            # Expel oldest key to maintain total memory
            for key in self.keys[:-self.max_memory]:
                del self.key_cache[tuple(key.detach().cpu().numpy())]
            self.keys = self.keys[-self.max_memory:]
            self.values = self.values[-self.max_memory:]
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None
        self.kdtree.build_index(self.keys.cpu().numpy(),  algorithm="kdtree")
        self.stale_index = False



    def read(self, lookup_key, is_learning=False):
        """
        Perform memory lookup, support batch query
        is_learning: read during minimize td error --> do not count as usage
        """

        lookup_indexesb, distb = self.kdtree.nn_index(
            lookup_key.detach().cpu().numpy(), min(self.num_neighbors, len(self.keys)),  algorithm="kdtree")
        indexes = torch.LongTensor(lookup_indexesb).view(-1)

        old_indexes = indexes
        vshape = self.values.shape
        if len(vshape)==2:
            indexes=indexes.unsqueeze(-1).repeat(1, vshape[1])

        kvalues = inverse_distance(torch.tensor(distb).to(lookup_key.device))
        kvalues = kvalues/torch.sum(kvalues, dim=-1, keepdim=True)

        values = self.values.gather(0, indexes.to(lookup_key.device))

        if len(vshape)==2:
            values = values.reshape(lookup_key.shape[0],vshape[1], -1)
            kvalues = kvalues.unsqueeze(1)
        else:
            values = values.reshape(lookup_key.shape[0],-1)
        if random.random() > 0.7:
            return torch.max(values, dim=-1)[0].detach()
 
        return torch.sum(kvalues*values, dim=-1)



    def write(self, lookup_key, R):
        """
        support batch
        """
        lookup_indexesb, distb = self.kdtree.nn_index(
            lookup_key.detach().cpu().numpy(), min(self.num_neighbors, len(self.keys)), algorithm="kdtree")

        for b, lookup_indexes in enumerate(lookup_indexesb):
            ks = []
            kernel_sum = 0
            if self.num_neighbors == 1 and len(lookup_indexes.shape)==1:
                lookup_indexes=[lookup_indexes]
                distb=[distb]
            if isinstance(lookup_indexes, np.int32):
                lookup_indexes = [lookup_indexes]
                distb=[distb]


            for i, index in enumerate(lookup_indexes):
      
                curv = self.values[int(index)]
                kernel_val = inverse_distance(distb[b][i])
                kernel_sum += kernel_val
                ks.append((index,kernel_val, curv))
            for index, kernel_val, curv in ks:
                self.update((R-curv)*kernel_val/kernel_sum*self.lr + curv, index)


