import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize_adj(adj):
    eye = torch.eye(adj.size(0), device=adj.device, dtype=adj.dtype)
    adj = adj + eye
    rowsum = adj.sum(1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

class CausalMGAE(nn.Module):
    def __init__(self, in_features=1, hidden_dim=16):
        super().__init__()
        self.w_enc = nn.Linear(in_features, hidden_dim)
        self.w_dec = nn.Linear(hidden_dim, in_features)

    def forward(self, x_speed, x_order, adj_spatial):
        mask = torch.sigmoid(x_order - 1.0)
        x_masked = x_speed * mask
        norm_adj = normalize_adj(adj_spatial)
        h_enc = F.relu(torch.einsum("bsni,nm->bsmi", self.w_enc(x_masked), norm_adj))
        x_tilde = torch.sigmoid(torch.einsum("bsni,nm->bsmi", self.w_dec(h_enc), norm_adj))
        return x_tilde

class DualGraphMHA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_spatial, adj_semantic):
        batch_steps, num_nodes, dim = x.shape
        q = self.q_linear(x).view(batch_steps, num_nodes, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_steps, num_nodes, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_steps, num_nodes, self.nhead, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        spatial_mask = (adj_spatial <= 0).view(1, 1, num_nodes, num_nodes)
        scores_spatial = scores.masked_fill(spatial_mask, -6e4)
        attn_spatial = F.softmax(scores_spatial, dim=-1)
        out_spatial = torch.matmul(attn_spatial, v).transpose(1, 2).reshape(batch_steps, num_nodes, dim)
        attn_semantic = F.softmax(scores, dim=-1)
        attn_semantic = attn_semantic * adj_semantic.view(1, 1, num_nodes, num_nodes)
        attn_semantic = attn_semantic / (attn_semantic.sum(dim=-1, keepdim=True) + 1e-9)
        out_semantic = torch.matmul(attn_semantic, v).transpose(1, 2).reshape(batch_steps, num_nodes, dim)
        out_fused = self.out_proj(torch.cat([out_spatial, out_semantic], dim=-1))
        x = self.norm1(x + self.dropout(out_fused))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x

class CCSTMTHead(nn.Module):
    def __init__(self, hidden_dim, seq_len=12):
        super().__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(hidden_dim * seq_len, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.pi_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.mu_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.theta_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Softplus())
        self.v_layer = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU())
        nn.init.constant_(self.pi_layer[0].bias, -math.log((1.0 - 0.7) / 0.7))

    def forward(self, x):
        batch_size, num_nodes, seq_len, hidden_dim = x.shape
        x = x.reshape(batch_size, num_nodes, seq_len * hidden_dim)
        x = self.shared_fc(x)
        pi = self.pi_layer(x).squeeze(-1)
        mu = self.mu_layer(x).squeeze(-1) + 1e-4
        theta = self.theta_layer(x).squeeze(-1) + 1e-4
        v_hat = self.v_layer(x).squeeze(-1)
        return pi, mu, theta, v_hat

class CC_STMT(nn.Module):
    def __init__(self, num_nodes, input_dim=5, hidden_dim=64, num_layers=2, nhead=8, seq_len=12):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.mgae = CausalMGAE(in_features=1, hidden_dim=16)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.mha_layers = nn.ModuleList([DualGraphMHA(hidden_dim, nhead) for _ in range(num_layers)])
        self.decoder = CCSTMTHead(hidden_dim, seq_len)

    def forward(self, x_order, x_speed, x_enroute, x_weather, adj_spatial, adj_semantic):
        batch_size, seq_len, num_nodes, _ = x_order.shape
        x_speed_tilde = self.mgae(x_speed, x_order, adj_spatial)
        x = torch.cat([x_order, x_speed_tilde, x_enroute, x_weather], dim=-1)
        x = F.relu(self.input_proj(x))
        x = x.view(batch_size * seq_len, num_nodes, self.hidden_dim)
        for layer in self.mha_layers:
            x = layer(x, adj_spatial, adj_semantic)
        x = x.view(batch_size, seq_len, num_nodes, self.hidden_dim).transpose(1, 2)
        return self.decoder(x)

def cc_stmt_loss(pi, mu, theta, v_hat, target_order, target_speed, is_warmup=False):
    eps = 1e-8
    speed_mask = (target_speed > 0).float()
    speed_loss = F.smooth_l1_loss(v_hat * speed_mask, target_speed * speed_mask, reduction="sum") / (speed_mask.sum() + eps)
    if is_warmup:
        return F.mse_loss(mu, target_order) + speed_loss
    pi = torch.clamp(pi, eps, 1.0 - eps)
    mu = torch.clamp(mu, eps, 1e5)
    theta = torch.clamp(theta, eps, 1e5)
    target_order = torch.clamp(torch.round(target_order), min=0.0)
    t1 = torch.lgamma(target_order + theta) - torch.lgamma(target_order + 1.0) - torch.lgamma(theta)
    t2 = theta * torch.log(theta / (theta + mu) + eps)
    t3 = target_order * torch.log(mu / (theta + mu) + eps)
    log_nb_pdf = torch.clamp(t1 + t2 + t3, max=0.0)
    nb_pdf = torch.exp(log_nb_pdf)
    zero_mask = (target_order == 0).float()
    prob_zero = pi + (1.0 - pi) * nb_pdf
    prob_non_zero = (1.0 - pi) * nb_pdf
    prob_total = zero_mask * prob_zero + (1.0 - zero_mask) * prob_non_zero
    nll = -torch.log(prob_total + eps).mean()
    expected_value = (1.0 - pi) * mu
    mse_penalty = F.mse_loss(expected_value, target_order)
    return nll + 0.1 * mse_penalty + 0.1 * speed_loss
