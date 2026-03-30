import torch

def small_world_connectivity(dist, c, l):

    dist_min = torch.min(dist, dim=1).values[:, None]
    dist_max = torch.max(dist, dim=1).values[:, None]
    dist_norm = (dist - dist_min) / (dist_max - dist_min + 1e-8)
    conn_prob = c * torch.exp(-(dist_norm / l) ** 2)
    input_conn = torch.where(torch.rand_like(conn_prob) < conn_prob, conn_prob, torch.zeros_like(conn_prob))
    if dist.shape[0] == dist.shape[1]:
        input_conn.fill_diagonal_(0)
    return input_conn
