import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from einops import rearrange
import math


class PatchEmbedding(nn.Module):
    """ViT-style patch embedding."""
    def __init__(self, img_size: int = 1024, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


class TransformerBlock(nn.Module):
    """Standard ViT transformer block."""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ViTBackbone(nn.Module):
    """Shared ViT backbone for DAE."""
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x


class DisentangledAnomalyEncoder(nn.Module):
    """
    Disentangled Anomaly Encoder (DAE) for CaDIU.
    Two-branch architecture: Primitive (shared) + Semantic (task-specific).
    """
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        latent_dim_prim: int = 512,
        latent_dim_sem: int = 256
    ):
        super().__init__()
        self.img_size = img_size
        self.latent_dim_prim = latent_dim_prim
        self.latent_dim_sem = latent_dim_sem

        # Shared backbone
        self.backbone = ViTBackbone(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # Primitive branch (shared across tasks)
        self.primitive_head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim_prim),
            nn.ReLU()
        )

        # Semantic branch (task-specific, initialized per task)
        self.semantic_head = nn.Sequential(
            nn.Linear(embed_dim, latent_dim_sem),
            nn.ReLU()
        )

        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_prim + latent_dim_sem, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, img_size * img_size)  # For simplicity; use U-Net in practice
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input image (B, C, H, W)
        Returns:
            z_prim: Primitive latent (B, latent_dim_prim)
            z_sem: Semantic latent (B, latent_dim_sem)
            recon: Reconstructed image (B, H, W)
        """
        features = self.backbone(x)  # (B, N+1, E)
        cls_token = features[:, 0]   # (B, E)

        z_prim = self.primitive_head(cls_token)  # (B, latent_dim_prim)
        z_sem = self.semantic_head(cls_token)    # (B, latent_dim_sem)

        # Reconstruction
        z_concat = torch.cat([z_prim, z_sem], dim=1)  # (B, latent_dim_prim + latent_dim_sem)
        recon_flat = self.decoder(z_concat)           # (B, H*W)
        recon = recon_flat.view(-1, self.img_size, self.img_size)  # (B, H, W)

        return z_prim, z_sem, recon

    def compute_loss(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        z_prim: torch.Tensor,
        z_sem: torch.Tensor,
        recon: torch.Tensor,
        lambda_mi: float = 1.8,
        gamma_smooth: float = 0.3
    ) -> Dict[str, torch.Tensor]:
        """
        Compute causal disentanglement loss:
        L = L_recon + λ * I(z_sem; z_prim) + γ * ||∇_x z_sem||_F^2
        """
        # Reconstruction loss (pixel-wise MSE on anomaly mask)
        recon_loss = F.mse_loss(recon, mask.float())

        # Mutual information approximation via variational upper bound
        # Here we use a simple heuristic: ||z_sem^T z_prim||_F^2
        mi_loss = torch.mean(torch.sum(z_sem * z_prim, dim=1) ** 2)

        # Semantic smoothness (gradient penalty)
        x.requires_grad_(True)
        z_sem_grad = torch.autograd.grad(
            outputs=z_sem.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True
        )[0]
        smooth_loss = torch.mean(z_sem_grad.pow(2).sum(dim=[1, 2, 3]))

        total_loss = recon_loss + lambda_mi * mi_loss + gamma_smooth * smooth_loss

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "mi_loss": mi_loss,
            "smooth_loss": smooth_loss
        }

    def initialize_semantic_branch_from_primitive(self):
        """Initialize semantic head from primitive head for new tasks."""
        with torch.no_grad():
            self.semantic_head[0].weight.copy_(self.primitive_head[0].weight[:self.latent_dim_sem, :])
            self.semantic_head[0].bias.copy_(self.primitive_head[0].bias[:self.latent_dim_sem])


# Example usage
if __name__ == "__main__":
    # Simulate high-res input (VisA-style)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DisentangledAnomalyEncoder(
        img_size=256,  # Reduced for demo
        patch_size=16,
        latent_dim_prim=256,
        latent_dim_sem=128
    ).to(device)

    x = torch.randn(2, 3, 256, 256).to(device)
    mask = torch.randint(0, 2, (2, 256, 256)).to(device)

    z_prim, z_sem, recon = model(x)
    losses = model.compute_loss(x, mask, z_prim, z_sem, recon)

    print(f"Reconstruction shape: {recon.shape}")
    print(f"Primitive latent shape: {z_prim.shape}")
    print(f"Semantic latent shape: {z_sem.shape}")
    print(f"Total loss: {losses['total_loss'].item():.4f}")