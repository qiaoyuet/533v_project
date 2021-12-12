from torch.utils.data import Dataset
import torch
import os
import pickle
import numpy as np
from problems.tsp.state_tsp import StateTSP
from problems.tsp.state_tsptw import StateTSPTW
from utils.beam_search import beam_search


class TSP(object):

    NAME = 'tsp'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
            torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
            pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        d = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        return (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSP.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):

        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSP.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class TSPDataset(Dataset):
    
    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset+num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [torch.FloatTensor(size, 2).uniform_(0, 1) for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]


class TSPTW(object):
    NAME = 'tsptw'

    @staticmethod
    def _get_penalty(tw_lb_sorted, tw_ub_sorted, pair_dist, lam):
        cur_time = 0
        penalty = 0
        for i in range(len(pair_dist)):
            penalty += max(0, cur_time - tw_ub_sorted[i])
            cur_time += pair_dist[i]
            if cur_time < tw_lb_sorted[i + 1]:
                cur_time = tw_lb_sorted[i + 1]
        penalty += max(0, cur_time - tw_ub_sorted[-1])
        cost = cur_time + lam * penalty
        return cost, cur_time, penalty

    @staticmethod
    def lam_scheduler(lam):
        pass

    @classmethod
    def get_costs(cls, dataset, pi, lam):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) ==
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        # Gather dataset in order of tour
        loc = dataset['loc']
        d = loc.gather(1, pi[..., None].expand(*pi.size(), loc.size(-1)))

        # Length is distance (L2-norm of difference) from each next location from its prev and of last from first
        # obj_dist = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2).sum(1) + (d[:, 0] - d[:, -1]).norm(p=2, dim=1)

        tw_lb, tw_ub = dataset['tw_lb'].unsqueeze(-1), dataset['tw_ub'].unsqueeze(-1)
        tw_lb_sorted = tw_lb.gather(1, pi[..., None].expand(*pi.size(), tw_lb.size(-1)))
        tw_ub_sorted = tw_ub.gather(1, pi[..., None].expand(*pi.size(), tw_ub.size(-1)))
        pair_dist = (d[:, 1:] - d[:, :-1]).norm(p=2, dim=2)
        tw_lb_sorted = tw_lb_sorted.squeeze(2).cpu().detach().numpy()
        tw_ub_sorted = tw_ub_sorted.squeeze(2).cpu().detach().numpy()
        pair_dist = pair_dist.cpu().detach().numpy()
        assert tw_lb_sorted.shape[1] == tw_ub_sorted.shape[1] == pair_dist.shape[1] + 1
        costs, dists, penalties = [], [], []
        for i in range(pair_dist.shape[0]):
            cost, dist, penalty = cls._get_penalty(tw_lb_sorted[i], tw_ub_sorted[i], pair_dist[i], lam)
            costs.append(cost)
            dists.append(dist)
            penalties.append(penalty)
        costs = torch.from_numpy(np.array(costs)).cuda()
        dists = torch.from_numpy(np.array(dists)).cuda()
        penalties = torch.from_numpy(np.array(penalties)).cuda()
        return costs, dists, penalties, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return TSPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateTSPTW.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = TSPTW.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


def make_instance(args):
    loc, tw_center, tw_width, tw_lb, tw_ub, *args = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float),
        'tw_center': torch.tensor(tw_center, dtype=torch.float),
        'tw_width': torch.tensor(tw_width, dtype=torch.float),
        'tw_lb': torch.tensor(tw_lb, dtype=torch.float),
        'tw_ub': torch.tensor(tw_ub, dtype=torch.float)
    }


def generate_instance(size):
    # 20 * np.sqrt(2) = 28.3
    loc = torch.FloatTensor(size, 2).uniform_(0, 1)
    tw_center = torch.FloatTensor(size).uniform_(0, 15)
    tw_width = torch.FloatTensor(size).uniform_(1e-3, 15)
    tw_lb = tw_center - tw_width
    tw_lb[tw_lb < 0] = 0
    tw_ub = tw_center + tw_width
    return {
        'loc': loc,
        'tw_center': tw_center,
        'tw_width': tw_width,
        'tw_lb': tw_lb,
        'tw_ub': tw_ub
    }


class TSPTWDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(TSPTWDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [make_instance(args) for args in data[offset:offset+num_samples]]
        else:
            self.data = [generate_instance(size) for _ in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
