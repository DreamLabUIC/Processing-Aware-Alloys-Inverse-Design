import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"This code partially adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"

def gather_by_timestep(schedule_tensor, time_ids, ref_shape):
    "TThe function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"
    batch_count = time_ids.shape[0]
    gathered = schedule_tensor.gather(-1, time_ids.cpu())
    reshaped = gathered.reshape(batch_count, *((1,) * (len(ref_shape) - 1))).to(time_ids.device)
    return reshaped


# forward diffusion (using the nice property)

def forward_diffuse(x_clean, time_ids, sqrt_alpha_cum, sqrt_one_minus_alpha_cum, noise_tensor=None):
    "The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"
    if noise_tensor is None:
        noise_tensor = torch.randn_like(x_clean)

    sqrt_alpha_t = gather_by_timestep(sqrt_alpha_cum, time_ids, x_clean.shape)
    sqrt_one_minus_alpha_t = gather_by_timestep(sqrt_one_minus_alpha_cum, time_ids, x_clean.shape)

    return sqrt_alpha_t * x_clean + sqrt_one_minus_alpha_t * noise_tensor


# Position embeddings
class SinusoidalPositionEmbeddings(nn.Module):
    "The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings



    
@torch.no_grad()
def reverse_step(noise_model, x_t, time_ids, condition_vec, step_index, beta_schedule):
    "The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"
    # define alphas
    alpha_schedule = 1.0 - beta_schedule
    alpha_cumprod = torch.cumprod(alpha_schedule, axis=0)
    alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alpha = torch.sqrt(1.0 / alpha_schedule)

    # calculations for diffusion q(z_t | z_{t-1}) and others
    sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)

    # calculations for posterior q(z_{t-1} | z_t, z_0)
    posterior_var = beta_schedule * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

    beta_t = gather_by_timestep(beta_schedule, time_ids, x_t.shape)
    sqrt_one_minus_alpha_t = gather_by_timestep(sqrt_one_minus_alpha_cumprod, time_ids, x_t.shape)
    sqrt_recip_alpha_t = gather_by_timestep(sqrt_recip_alpha, time_ids, x_t.shape)

    # Equation 11: predict mean using the noise predictor
    predicted_mean = sqrt_recip_alpha_t * (
        x_t - beta_t * noise_model(x_t, time_ids, condition_vec) / sqrt_one_minus_alpha_t
    )

    if step_index == 0:
        return predicted_mean
    else:
        posterior_var_t = gather_by_timestep(posterior_var, time_ids, x_t.shape)
        gaussian_noise = torch.randn_like(x_t)
        # Algorithm 2 line 4
        return predicted_mean + torch.sqrt(posterior_var_t) * gaussian_noise
    

# Algorithm 2 (including returning all images)

@torch.no_grad()
def reverse_process(noise_model, condition_vec, num_steps, beta_schedule, output_shape):
    "The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator"
    run_device = next(noise_model.parameters()).device

    batch_size = output_shape[0]
    # start from pure noise (for each example in the batch)
    current_state = torch.randn(output_shape, device=run_device)
    trajectory = []

    for step in reversed(range(0, num_steps)):
        step_times = torch.full((batch_size,), step, device=run_device, dtype=torch.long)
        current_state = reverse_step(noise_model, current_state, step_times, condition_vec, step, beta_schedule)
        trajectory.append(current_state)

    return trajectory


@torch.no_grad()
def generate_samples(noise_model, condition_vec, latent_dim, num_steps, beta_schedule, batch_size):
    return reverse_process(noise_model, condition_vec, num_steps, beta_schedule, output_shape=(batch_size, latent_dim))


class FiLMBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

    def forward(self, x, gamma, beta, t_emb):
        h = self.norm(x)
        h = h * gamma + beta        # FiLM conditioning (forces usage)
        h = self.fc(h)
        h = h + t_emb               # time embedding injection
        h = self.act(h)
        return x + h                # residual
        

class DenoiseNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, n_cond, d_cond):
        super().__init__()
        self.n_layers = n_layers
        self.n_cond = n_cond
        self.hidden_dim = hidden_dim
        self.d_cond = d_cond

        # ---- Condition MLP → produces gamma & beta per layer ----
        # total output: n_layers * 2 * hidden_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(n_cond, 128),
            nn.ReLU(),
            nn.Linear(128, n_layers * 2 * hidden_dim),
        )

        # ---- Time embedding ----
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # ---- Input projection ----
        self.inp = nn.Linear(input_dim, hidden_dim)

        # ---- Residual blocks with FiLM ----
        self.blocks = nn.ModuleList([
            FiLMBlock(hidden_dim) for _ in range(n_layers)
        ])

        # ---- Output projection ----
        self.out = nn.Linear(hidden_dim, input_dim)


    def forward(self, x, t, cond):
        B = x.size(0)

        # --- Condition processing ---
        cond = torch.nan_to_num(cond.view(B, -1), nan=0.0)
        cond_params = self.cond_mlp(cond)               # (B, L*2*H)
        cond_params = cond_params.view(B, self.n_layers, 2, self.hidden_dim)

        gammas = cond_params[:, :, 0, :]                # (B, L, H)
        betas  = cond_params[:, :, 1, :]                # (B, L, H)

        # --- Time embedding ---
        t_emb = self.time_mlp(t)                        # (B, H)

        # --- Project input ---
        h = self.inp(x)

        # --- Residual FiLM blocks ---
        for i, block in enumerate(self.blocks):
            h = block(h, gammas[:, i, :], betas[:, i, :], t_emb)

        # --- Output ---
        return self.out(h)



def cosine_beta_schedule(timesteps, s=0.008):
    """
    The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    """
    The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    """
    The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """
    The function is adopted from: https://github.com/iakovosevdaimon/Neural-Graph-Generator
    """
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start




def compute_total_mse(predicted, target):
    """
    Compute the total mean square error (MSE) between predicted and target properties.
    
    Parameters:
    predicted (torch.Tensor): The predicted values with shape [batch_size, 3].
    target (torch.Tensor): The target values with shape [batch_size, 3].
    
    Returns:
    total_mse (torch.Tensor): The total mean square error (sum of all errors).
    """
    # Ensure the tensors are of the same shape
    assert predicted.shape == target.shape, "Shape mismatch between predicted and target tensors"

    # Compute the total mean square error (sum of all errors)
    total_mse = F.mse_loss(predicted, target, reduction='mean')
    
    return total_mse