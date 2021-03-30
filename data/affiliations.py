import torch
import torch.nn.functional as F
from data.cluster import sample_labels
from data.mvn import MultivariateNormalDiag
import math

def read_affiliations_data(s, device):
    return torch.load(s, map_location=device)

# note that reading the affiliations every time is going to be slow; could be factored out later
def sample_affiliations(B, N, K,
        rand_N=True, rand_K=True,
        device='cpu',
        datafile = 'data/affiliations/v1.pt'):

    N = torch.randint(int(0.3*N), N, [1], dtype=torch.long).item() \
            if rand_N else N

    # read file of all affiliations clusters
    dataset = read_affiliations_data(datafile, device)

    #TODO: subsample these to (B, N, K):
    # labels = sample_labels(B, N, K, alpha=alpha, rand_K=rand_K, device=device)
    # params = mvn.sample_params([B, K], device=device)
    # gathered_params = torch.gather(params, 1,
    #         labels.unsqueeze(-1).repeat(1, 1, params.shape[-1]))
    # X = mvn.sample(gathered_params)
    # if onehot:
    #     labels = F.one_hot(labels, K)
    # dataset = {'X':X, 'labels':labels}
    choice = torch.randint(0, len(dataset), [1], dtype=torch.long).item()
    return dataset[choice]

if __name__ == '__main__':

    ds = sample_affiliations(10, 300, 4, rand_K=True)
    print(ds['ll'])
