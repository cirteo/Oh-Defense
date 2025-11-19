import torch

def normalize_columns(u):
    norms = torch.norm(u, dim=0)
    norms = torch.clamp(norms, min=1e-8)
    U_normalized = u / norms
    return U_normalized
def normalize_rows(u):
    norms = torch.norm(u, dim=1, keepdim=True) 
    norms = torch.clamp(norms, min=1e-8)  
    U_normalized = u / norms  
    return U_normalized
def compute_consine_similarity(u1, u2):
    u1 = u1.to(torch.float32)
    u2 = u2.to(torch.float32)
    if u1.shape[0] == 1:
        u1_normalized = normalize_rows(u1)
        u2_normalized = normalize_rows(u2)
        S = torch.matmul(u1_normalized, u2_normalized.T)
    else:
        u1_normalized = normalize_columns(u1)
        u2_normalized = normalize_columns(u2)
        S = torch.matmul(u1_normalized[:, :1].T, u2_normalized[:, :1])
    
    principal_angles = torch.acos(torch.clamp(S, -1, 1))
    principal_angles_degrees = principal_angles * 180 / torch.pi
    return principal_angles_degrees.item()

