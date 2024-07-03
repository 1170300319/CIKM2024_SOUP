import math
import random
from typing import TypeVar, Optional, Iterator
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, Sampler
import torch.distributed as dist

__all__ = ["DisMultiSetSampler", "MyDistributedSampler"]

T_co = TypeVar('T_co', covariant=True)


class DisMultiSetSampler(Sampler[T_co]):
    '''
        Args:
            sample_len_map: A map, containing information of sampler numbers of each dataset.
            For example:{'U2Q':60, 'QAC':30, 'Q2Q':10, 'I2Q':10}
    '''

    def __init__(self, dataset: Dataset, sample_len_map=None, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        # super(DistributedSampler, self).__init__(dataset)
        # DistributedSampler.__init__(self, dataset, shuffle=shuffle)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.alpha = 1.0  # traditional采样比例
        self.tra_times = 2  # traditional数据的重复次数
        self.sample_len_map = sample_len_map
        # calculate the sample length for all dataset
        self.subset_size = sum([self.sample_len_map[each] if each != 'traditional' else 0 for each in self.sample_len_map])
        # 采样
        if 'traditional' in self.sample_len_map:
            if 'text' in self.sample_len_map:
                self.subset_size += int(self.tra_times * self.alpha * self.sample_len_map['text'])
            if 'Q2Q' in self.sample_len_map:
                self.subset_size += int(self.tra_times * self.alpha * self.sample_len_map['Q2Q'])

        if self.subset_size > len(self.dataset):
            raise RuntimeError("Total sample size bigger than total dataset length")
        # calculate the total length for all dataset
        self.subset_length = self.dataset.getLengthAllDataset()
        self.dataset_names = [each for each in self.subset_length]
        for each in self.dataset_names:
            if self.subset_length[each] < self.sample_len_map[each]:
                raise RuntimeError("sample size bigger than dataset length for " + each)
        # self.num_samples = math.ceil(self.subset_size / self.num_replicas)
        self.num_samples = math.floor(self.subset_size / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        # raise RuntimeError(self.total_size, self.num_samples, self.num_replicas)

        indices_list = []
        traditional_list = []
        sub_len = 0
        # 如果有traditional的任务，则取U2Q，QAC，Q2Q数据集的20%出来组合成新数据集
        if self.shuffle:
            # raise RuntimeError('dataset_names: ', self.dataset_names)
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
            for each in self.dataset_names:
                if each == 'traditional':
                    continue

                each_indices = torch.randperm(self.subset_length[each], generator=g)  # type: ignore[arg-type]
                each_indices += sub_len  # 多个数据集之间的bias
                each_indices = each_indices.tolist()  # 不能提前转list

                if each in ['text', 'Q2Q']:
                    traditional_list += each_indices[:int(self.sample_len_map[each]*self.alpha)]

                # indices_list.append(each_indices)  # 裁剪
                indices_list.append(each_indices[:self.sample_len_map[each]])  # 裁剪
                sub_len += self.subset_length[each]
        else:
            # indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
            for each in self.dataset_names:
                if each == 'traditional':
                    continue

                each_indices = torch.tensor(list(range(self.subset_length[each])))  # 偷懒统一格式
                each_indices += sub_len
                each_indices = each_indices.tolist()

                if each in ['text', 'Q2Q']:
                    traditional_list += each_indices[:int(self.sample_len_map[each]*self.alpha)]

                # indices_list.append(each_indices)  # 裁剪
                indices_list.append(each_indices[:self.sample_len_map[each]])
                sub_len += self.subset_length[each]

        if 'traditional' in self.dataset_names:
            # sublength已经算过了
            # 加上一个统一的bias，以确定这些数据是traditional的
            # traditional_list = torch.tensor(traditional_list)  # 偷懒统一格式
            # traditional_list += sub_len
            # traditional_list = traditional_list.tolist()
            # indices_list.append(traditional_list)
            traditional_list_c = torch.tensor(traditional_list)  # 偷懒统一格式
            for i in range(1, self.tra_times):
                traditional_list_c = torch.concat((traditional_list_c, torch.tensor(traditional_list)), dim=0)

            traditional_list_c += sub_len
            traditional_list_c = traditional_list_c.tolist()
            indices_list.append(traditional_list_c)

        indices = []
        for each in indices_list:
            indices += each
        random.shuffle(indices)  # 再次打乱，打乱任务之间的顺序

        # remove tail of data to make it evenly divisible.
        # drop last先不管了
        indices = indices[:self.total_size]
        if len(indices) != self.total_size:
            raise RuntimeError("length indices not equal total size", len(indices), self.total_size)
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError("length indices not equal num samples", len(indices), self.num_replicas)
        # assert len(indices) == self.num_samples

        # raise RuntimeError("indices_list", indices[:100],  indices[len(indices)-100:], self.subset_length, self.sample_len_map, len(indices))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class MyDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size and that any instance of it always
        returns the same elements in the same order.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.

    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> # xdoctest: +SKIP
        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
            # self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        # raise RuntimeError(self.total_size, self.num_samples, self.num_replicas)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # if not self.drop_last:
        #     # add extra samples to make it evenly divisible
        #     padding_size = self.total_size - len(indices)
        #     if padding_size <= len(indices):
        #         indices += indices[:padding_size]
        #     else:
        #         indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        # else:
        # remove tail of data to make it evenly divisible.
        indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        # raise RuntimeError("indices_list", indices[:100],  indices[len(indices)-100:], len(indices))

        # raise RuntimeError("indices_list", indices)

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch

