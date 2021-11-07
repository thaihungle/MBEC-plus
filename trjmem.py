import torch
import torch.optim as optim
from torch.nn import Parameter
import pyflann
import numpy as np
import random

def inverse_distance(d, epsilon=1e-3):
    return 1 / (d + epsilon)

class TrjMemory:
    def __init__(self, kernel, num_neighbors, max_memory, lr):
        self.kernel = kernel
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
        # indexes_to_be_updated is the set of indexes to be updated on a call to update_params
        # This allows us to rebuild only the keys of key_cache that need to be rebuilt when necessary
        self.indexes_to_be_updated = set()

        # Keys and value to be inserted into self.keys and self.values when commit_insert is called
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None

        # Move recently used lookup indexes
        # These should be moved to the back of self.keys and self.values to get LRU property
        self.move_to_back = set()

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
        if self.key_cache.get(tuple(key.data.cpu().numpy()[0])) is not None:
            if self.stale_index:
                self.commit_insert()
            return int(self.kdtree.nn_index(key.data.cpu().numpy(), 1)[0][0])
        else:
            return None

    def update(self, value, index):
        """
        Set self.values[index] = value
        """
        values = self.values.data
        values[index] = value[0].data
        self.values = Parameter(values)
        self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)

    def insert(self, key, value):
        """
        Insert key, value pair into memory (do not support batch insert)
        """
        if torch.cuda.is_available():
            if self.keys_to_be_inserted is None:
                # Initial insert
                self.keys_to_be_inserted = key.data.cuda()
                self.values_to_be_inserted = value.data.cuda()
            else:
                self.keys_to_be_inserted = torch.cat(
                    [self.keys_to_be_inserted.cuda(), key.data.cuda()], 0)
                self.values_to_be_inserted = torch.cat(
                    [self.values_to_be_inserted.cuda(), value.data.cuda()], 0)
        else:
            if self.keys_to_be_inserted is None:
                # Initial insert
                self.keys_to_be_inserted = key.data
                self.values_to_be_inserted = value.data
            else:
                self.keys_to_be_inserted = torch.cat(
                    [self.keys_to_be_inserted, key.data], 0)
                self.values_to_be_inserted = torch.cat(
                    [self.values_to_be_inserted, value.data], 0)
        self.key_cache[tuple(key.data.cpu().numpy()[0])] = 0
        self.stale_index = True

    def commit_insert(self):
        if self.keys is None:
            self.keys = self.keys_to_be_inserted
            self.values = self.values_to_be_inserted
        elif self.keys_to_be_inserted is not None:
            self.keys = torch.cat([self.keys.data, self.keys_to_be_inserted], 0)
            self.values = torch.cat([self.values.data, self.values_to_be_inserted], 0)

        # Move most recently used key-value pairs to the back
        if len(self.move_to_back) != 0:
            self.keys = torch.cat([self.keys.data[list(set(range(len(
                self.keys))) - self.move_to_back)], self.keys.data[list(self.move_to_back)]], 0)
            self.values = torch.cat([self.values.data[list(set(range(len(
                self.values))) - self.move_to_back)], self.values.data[list(self.move_to_back)]], 0)
            self.move_to_back = set()

        if len(self.keys) > self.max_memory:
            # Expel oldest key to maintain total memory
            for key in self.keys[:-self.max_memory]:
                del self.key_cache[tuple(key.data.cpu().numpy())]
            self.keys = self.keys[-self.max_memory:].data
            self.values = self.values[-self.max_memory:].data
        self.keys_to_be_inserted = None
        self.values_to_be_inserted = None
        # self.optimizer = optim.RMSprop([self.keys, self.values], lr=self.lr)
        self.kdtree.build_index(self.keys.data.cpu().numpy(), algorithm='kdtree')
        self.stale_index = False


    def nearest(self, lookup_key):
        lookup_indexesb, distb = self.kdtree.nn_index(
            lookup_key.data.cpu().numpy(), 1)
        indexes = torch.LongTensor(lookup_indexesb).view(-1)
        values = torch.tensor(self.values).gather(0, indexes.to(lookup_key.device))
        return values.reshape(lookup_key.shape[0], -1)



    def read(self, lookup_key, is_learning=False):
        """
        Perform memory lookup, support batch query
        is_learning: read during minimize td error --> do not count as usage
        """
        lookup_indexesb, distb = self.kdtree.nn_index(
            lookup_key.data.cpu().numpy(), min(self.num_neighbors, len(self.keys)))
        indexes = torch.LongTensor(lookup_indexesb).view(-1)

        old_indexes = indexes
        vshape = torch.tensor(self.values).shape
        if len(vshape)==2:
            indexes=indexes.unsqueeze(-1).repeat(1, vshape[1])

        kvalues = inverse_distance(torch.tensor(distb).to(lookup_key.device))
        kvalues = kvalues/torch.sum(kvalues, dim=-1, keepdim=True)

        values = torch.tensor(self.values).gather(0, indexes.to(lookup_key.device))

        if len(vshape)==2:
            values = values.reshape(lookup_key.shape[0],vshape[1], -1)
            kvalues = kvalues.unsqueeze(1)
        else:
            values = values.reshape(lookup_key.shape[0],-1)
        if random.random() > 0.7:
            return torch.max(values, dim=-1)[0].detach()
        if not is_learning:
            self.move_to_back.update(old_indexes.numpy())
        return torch.sum(kvalues*values, dim=-1)



    def write(self, lookup_key, R, update_flag=False):
        """
        support batch
        If update_flag == True, add the nearest neighbor indexes to self.indexes_to_be_updated
        """
        lookup_indexesb, distb = self.kdtree.nn_index(
            lookup_key.data.cpu().numpy(), min(self.num_neighbors, len(self.keys)))

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
                if update_flag:
                    self.indexes_to_be_updated.add(int(index))
                else:
                    self.move_to_back.add(int(index))
                curv = self.values[int(index)]
                kernel_val = inverse_distance(distb[b][i])
                kernel_sum += kernel_val
                ks.append((index,kernel_val, curv))
            for index, kernel_val, curv in ks:
                self.update((R-curv)*kernel_val/kernel_sum*self.lr + curv, index)


    def update_params(self):
        """
        Use self.indexes_to_be_updated to update self.key_cache accordingly and rebuild the index of self.kdtree
        """
        for index in self.indexes_to_be_updated:
            del self.key_cache[tuple(self.keys[index].data.cpu().numpy())]
        for index in self.indexes_to_be_updated:
            self.key_cache[tuple(self.keys[index].data.cpu().numpy())] = 0
        self.indexes_to_be_updated = set()
        self.kdtree.build_index(self.keys.data.cpu().numpy())
        self.stale_index = False
