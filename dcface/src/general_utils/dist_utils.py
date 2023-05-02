import os
import pickle
import torch
import functools
import torch.distributed as dist


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ["LOCAL_RANK"])

def all_gather(data, src):
    if dist.is_initialized():
        dist.all_gather(data, src)
    else:
        data[0] = src

def split_tensor(x, rank=None):
    """
    extract the tensor for a corresponding "worker" in the batch dimension
    Args:
        x: (n, c)
    Returns: x: (n_local, c)
    """
    n = len(x)
    if rank is None:
        rank = get_local_rank()
    world_size = get_world_size()
    # print(f'rank: {rank}/{world_size}')
    per_rank = n // world_size
    return x[rank * per_rank:(rank + 1) * per_rank]


def barrier():
    if dist.is_initialized():
        dist.barrier()
    else:
        pass

def broadcast(data, src):
    if dist.is_initialized():
        dist.broadcast(data, src)
    else:
        pass

def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    else:
        return 0

def chunk_size(size, rank, world_size):
    extra = rank < size % world_size
    return size // world_size + extra