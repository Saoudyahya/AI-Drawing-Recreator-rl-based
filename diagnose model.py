"""
Diagnostic Script for Saved Models
Tests models from saved_models/ directory with timestamp filenames
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import morphology
import glob
import os

# Copy exact architecture from training code
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=7, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, n_patches_side=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_patches_side = n_patches_side
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches_side * n_patches_side, embed_dim) * 0.02)

    def forward(self, x):
        return x + self.pos_embed


class PhaseAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.phase_weight = nn.Parameter(torch.ones(num_heads))

    def forward(self, x, phase_signal=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if phase_signal is not None:
            phase_mod = self.phase_weight.view(1, -1, 1, 1) * phase_signal.view(B, 1, 1, 1)
            attn = attn * (1 + 0.2 * phase_mod)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class ImprovedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PhaseAwareAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout)
        self.drop_path = nn.Dropout(0.05)

    def forward(self, x, phase_signal=None):
        x = x + self.drop_path(self.attn(self.norm1(x), phase_signal))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ImprovedVisionTransformer(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_channels=7, embed_dim=256,
                 depth=8, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.pos_embed = PositionalEncoding(embed_dim, img_size // patch_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            ImprovedTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.phase_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    def forward(self, x):
        B = x.shape[0]
        phase_signal = x[:, 6, 0, 0]
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        phase_tokens = self.phase_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, phase_tokens, x], dim=1)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, phase_signal)
        x = self.norm(x)
        return x


class ImprovedTransformerPolicy(nn.Module):
    def __init__(self, canvas_size=64, grid_size=20, patch_size=8, embed_dim=256,
                 depth=8, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size + 1
        self.embed_dim = embed_dim

        self.encoder = ImprovedVisionTransformer(
            img_size=canvas_size, patch_size=patch_size, in_channels=7,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout
        )

        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, self.num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)
        state = state.permute(0, 3, 1, 2)
        features = self.encoder(state)
        cls_token = features[:, 0]
        phase_token = features[:, 1]
        combined = torch.cat([cls_token, phase_token], dim=1)
        action_logits = self.action_head(combined)
        value = self.value_head(combined)
        return action_logits, value


class CoverageDrawingEnv:
    def __init__(self, canvas_size=64, grid_size=20, target_color=[1.0, 0.0, 0.0]):
        self.canvas_size = canvas_size
        self.grid_size = grid_size
        self.cell_size = canvas_size // grid_size
        self.target_color = np.array(target_color)
        self.target = None
        self.target_outline = None
        self.target_interior = None
        self.reset()

    def set_target(self, target):
        self.target = target
        target_mask = (target[:, :, 0] > 0.5) & (target[:, :, 1] < 0.5) & (target[:, :, 2] < 0.5)
        distance = ndimage.distance_transform_edt(target_mask)
        self.target_outline = (distance > 0) & (distance <= 4)
        self.target_outline = morphology.remove_small_objects(self.target_outline, min_size=2)
        eroded_interior = ndimage.binary_erosion(target_mask, iterations=4)
        self.target_interior = eroded_interior & ~self.target_outline

    def reset(self):
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32)
        self.cursor_grid_x = self.grid_size // 2
        self.cursor_grid_y = self.grid_size // 2
        self.step_count = 0
        self.trajectory = []
        return self.get_state()

    def compute_coverage(self):
        if self.target is None or self.target_outline is None or self.target_interior is None:
            return 0.0, 0.0, 0.0
        canvas_mask = (self.canvas[:, :, 0] > 0.5) & (self.canvas[:, :, 1] < 0.5) & (self.canvas[:, :, 2] < 0.5)
        outline_coverage = np.sum(self.target_outline & canvas_mask) / max(1, np.sum(self.target_outline))
        interior_coverage = np.sum(self.target_interior & canvas_mask) / max(1, np.sum(self.target_interior))
        total_target = np.sum(self.target_outline) + np.sum(self.target_interior)
        total_coverage = np.sum((self.target_outline | self.target_interior) & canvas_mask) / max(1, total_target)
        return outline_coverage, interior_coverage, total_coverage

    def get_state(self):
        state = np.zeros((self.canvas_size, self.canvas_size, 7), dtype=np.float32)
        state[:, :, :3] = self.canvas
        cx = self.cursor_grid_x * self.cell_size + self.cell_size // 2
        cy = self.cursor_grid_y * self.cell_size + self.cell_size // 2
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                    state[ny, nx, 3] = 1.0
        outline_cov, interior_cov, _ = self.compute_coverage()
        state[:, :, 4] = outline_cov
        state[:, :, 5] = interior_cov
        phase = 1.0 if outline_cov > 0.70 else 0.0
        state[:, :, 6] = phase
        return state

    def grid_to_canvas(self, grid_x, grid_y):
        canvas_x = grid_x * self.cell_size + self.cell_size // 2
        canvas_y = grid_y * self.cell_size + self.cell_size // 2
        return canvas_x, canvas_y

    def draw_stroke(self, x1, y1, x2, y2):
        x1 = int(np.clip(x1, 0, self.canvas_size - 1))
        y1 = int(np.clip(y1, 0, self.canvas_size - 1))
        x2 = int(np.clip(x2, 0, self.canvas_size - 1))
        y2 = int(np.clip(y2, 0, self.canvas_size - 1))
        img = Image.fromarray((self.canvas * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        color = tuple((self.target_color * 255).astype(int))
        draw.line([x1, y1, x2, y2], fill=color, width=12)
        self.canvas = np.array(img).astype(np.float32) / 255.0

    def step(self, action):
        prev_x, prev_y = self.grid_to_canvas(self.cursor_grid_x, self.cursor_grid_y)
        if action >= self.grid_size * self.grid_size:
            return self.get_state(), True
        target_grid_x = action % self.grid_size
        target_grid_y = action // self.grid_size
        new_x, new_y = self.grid_to_canvas(target_grid_x, target_grid_y)
        self.draw_stroke(prev_x, prev_y, new_x, new_y)
        self.cursor_grid_x = target_grid_x
        self.cursor_grid_y = target_grid_y
        self.trajectory.append((target_grid_x, target_grid_y))
        self.step_count += 1
        done = self.step_count >= 250
        return self.get_state(), done


def create_circle(canvas_size=64):
    target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
    y, x = np.ogrid[:canvas_size, :canvas_size]
    mask = (x - canvas_size//2)**2 + (y - canvas_size//2)**2 <= 20**2
    target[mask] = [1.0, 0.0, 0.0]
    return target


def create_square(canvas_size=64):
    target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
    size = 36
    start = (canvas_size - size) // 2
    end = start + size
    target[start:end, start:end] = [1.0, 0.0, 0.0]
    return target


def create_triangle(canvas_size=64):
    target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
    img = Image.fromarray((target * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    center = canvas_size // 2
    size = 28
    height = int(size * np.sqrt(3) / 2)
    points = [(center, center - height * 2 // 3), (center - size, center + height // 3),
              (center + size, center + height // 3)]
    draw.polygon(points, fill=(255, 0, 0))
    return np.array(img).astype(np.float32) / 255.0


def diagnose_model(model_path, shape_name='circle'):
    print("="*70)
    print(f"🔍 DIAGNOSTIC REPORT: {shape_name.upper()}")
    print("="*70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n📱 Device: {device}")
    print(f"📂 Model: {os.path.basename(model_path)}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create policy
    policy = ImprovedTransformerPolicy(
        canvas_size=64, grid_size=20, patch_size=8,
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0, dropout=0.1
    ).to(device)

    policy.load_state_dict(checkpoint['model_state_dict'])
    print(f"🎓 Model mode BEFORE eval(): {policy.training}")
    policy.eval()
    print(f"🎓 Model mode AFTER eval(): {policy.training}")

    # Show metadata
    print(f"\n📊 Checkpoint Metadata:")
    print(f"   Best MSE: {checkpoint.get('best_mse', 'N/A')}")
    print(f"   Best Outline: {checkpoint.get('best_outline_cov', 'N/A')}")
    print(f"   Best Interior: {checkpoint.get('best_interior_cov', 'N/A')}")
    print(f"   Best Score: {checkpoint.get('best_total_score', 'N/A')}")

    # Create target
    if shape_name == 'circle':
        target = create_circle(64)
    elif shape_name == 'square':
        target = create_square(64)
    elif shape_name == 'triangle':
        target = create_triangle(64)
    else:
        target = create_circle(64)

    # Create environment
    env = CoverageDrawingEnv(64, 20)
    env.set_target(target)

    # Edge detection info
    print(f"\n🔍 Edge Detection:")
    print(f"   Outline pixels: {np.sum(env.target_outline)}")
    print(f"   Interior pixels: {np.sum(env.target_interior)}")

    # Run inference
    print(f"\n🎨 Running Full Inference...")
    state = env.reset()
    done = False
    step = 0

    coverages_log = []
    actions_log = []

    with torch.no_grad():
        while not done and step < 250:
            out_cov, int_cov, tot_cov = env.compute_coverage()
            coverages_log.append((out_cov, int_cov, tot_cov))

            state_tensor = torch.FloatTensor(state).to(device)
            action_logits, value = policy(state_tensor)
            action = torch.argmax(action_logits).item()
            actions_log.append(action)

            if step < 5 or step % 50 == 0:
                print(f"   Step {step:3d}: Out={out_cov:.3f}, Int={int_cov:.3f}, Tot={tot_cov:.3f}, Action={action}, Value={value.item():.3f}")

            state, done = env.step(action)
            step += 1

    # Final results
    final_out, final_int, final_tot = env.compute_coverage()
    final_mse = np.mean((env.canvas - target) ** 2)

    print(f"\n✅ FINAL RESULTS (after {step} steps):")
    print(f"   Outline Coverage: {final_out:.1%}")
    print(f"   Interior Coverage: {final_int:.1%}")
    print(f"   Total Coverage: {final_tot:.1%}")
    print(f"   MSE: {final_mse:.4f}")

    # Check if stuck
    unique_actions = len(set(actions_log[-100:]))
    if unique_actions < 5:
        print(f"\n⚠️  WARNING: Model may be stuck! Only {unique_actions} unique actions in last 100 steps")
    else:
        print(f"\n✅ Action diversity: {unique_actions} unique actions in last 100 steps")

    # Visualize
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    axes[0].imshow(target)
    axes[0].set_title('Target', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(env.canvas)
    axes[1].set_title(f'Output\nOut:{final_out:.1%} Int:{final_int:.1%}', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    edge_viz = np.ones((64, 64, 3))
    edge_viz[env.target_outline] = [1, 0, 0]
    edge_viz[env.target_interior] = [0, 1, 0]
    axes[2].imshow(edge_viz)
    axes[2].set_title('Edge Detection\nRed=Outline Green=Interior', fontsize=10, fontweight='bold')
    axes[2].axis('off')

    diff = np.abs(target - env.canvas)
    axes[3].imshow(diff, cmap='hot')
    axes[3].set_title(f'Error\nMSE={final_mse:.4f}', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    # Coverage plot
    out_covs = [c[0] for c in coverages_log]
    int_covs = [c[1] for c in coverages_log]
    tot_covs = [c[2] for c in coverages_log]

    axes[4].plot(out_covs, label='Outline', color='red', linewidth=2)
    axes[4].plot(int_covs, label='Interior', color='green', linewidth=2)
    axes[4].plot(tot_covs, label='Total', color='blue', linewidth=2)
    axes[4].set_xlabel('Step', fontsize=10)
    axes[4].set_ylabel('Coverage', fontsize=10)
    axes[4].set_title('Coverage Over Time', fontsize=12, fontweight='bold')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)
    axes[4].set_ylim([0, 1.05])

    plt.suptitle(f'{shape_name.upper()} - Model Diagnostic', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_name = f'diagnostic_{shape_name}_{os.path.basename(model_path)[:-3]}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n💾 Saved: {output_name}")
    plt.show()

    print("\n" + "="*70)
    return final_out, final_int, final_tot, final_mse


def find_and_test_all_models(models_dir='saved_models'):
    """Find all models in saved_models directory and test them"""
    print("="*70)
    print("🔍 FINDING ALL MODELS")
    print("="*70)

    if not os.path.exists(models_dir):
        print(f"❌ Directory not found: {models_dir}")
        return

    # Find all .pt files
    all_models = glob.glob(os.path.join(models_dir, '*.pt'))

    if not all_models:
        print(f"❌ No .pt files found in {models_dir}")
        return

    print(f"\n📁 Found {len(all_models)} model file(s) in {models_dir}:\n")

    # Group by shape (if filename starts with shape name)
    shapes_models = {}
    unclassified_models = []

    for path in all_models:
        filename = os.path.basename(path)
        size_mb = os.path.getsize(path) / (1024**2)

        classified = False
        for shape in ['circle', 'square', 'triangle', 'diamond', 'star', 'heart']:
            if filename.startswith(shape):
                if shape not in shapes_models:
                    shapes_models[shape] = []
                shapes_models[shape].append(path)
                classified = True
                break

        if not classified:
            unclassified_models.append(path)
            print(f"  ⚠️  {filename} ({size_mb:.1f} MB) - No shape prefix detected")

    # Show classified models
    if shapes_models:
        print(f"\n✅ Models with shape prefix:")
        for shape, paths in shapes_models.items():
            print(f"\n  {shape.upper()}: {len(paths)} model(s)")
            for p in paths:
                size_mb = os.path.getsize(p) / (1024**2)
                print(f"    - {os.path.basename(p)} ({size_mb:.1f} MB)")

    # Handle unclassified models
    if unclassified_models:
        print(f"\n❓ Found {len(unclassified_models)} model(s) without shape prefix.")
        print("   These need to be tested manually with a shape name.")
        print(f"\n   Example:")
        print(f"   python diagnose_saved_models.py {unclassified_models[0]} circle")
        print(f"\n   Or to test with shape detection from checkpoint:")

        for model_path in unclassified_models:
            # Try to load and detect shape from checkpoint
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                shape_name = checkpoint.get('shape_name', None)
                if shape_name:
                    print(f"\n   ✅ Detected shape from checkpoint: {os.path.basename(model_path)} → {shape_name}")
                    if shape_name not in shapes_models:
                        shapes_models[shape_name] = []
                    shapes_models[shape_name].append(model_path)
            except Exception as e:
                print(f"   ❌ Could not load {os.path.basename(model_path)}: {e}")

    # Test each shape's final model
    if shapes_models:
        print(f"\n{'='*70}")
        print("🧪 TESTING MODELS")
        print("="*70 + "\n")

        results = []
        for shape in sorted(shapes_models.keys()):
            # Find final model
            final_models = [p for p in shapes_models[shape] if 'final' in os.path.basename(p)]
            if not final_models:
                # Use most recent model
                final_models = shapes_models[shape]

            if final_models:
                model_path = max(final_models, key=os.path.getmtime)  # Most recent
                print(f"\n🎯 Testing {shape.upper()}: {os.path.basename(model_path)}")
                print("-"*70)

                out, inter, tot, mse = diagnose_model(model_path, shape)
                results.append({
                    'shape': shape,
                    'outline': out,
                    'interior': inter,
                    'total': tot,
                    'mse': mse,
                    'path': model_path
                })

        # Summary
        if results:
            print("\n" + "="*70)
            print("📊 SUMMARY OF ALL MODELS")
            print("="*70)
            print(f"{'Shape':<12} {'Outline':<10} {'Interior':<10} {'Total':<10} {'MSE':<10}")
            print("-"*70)
            for r in results:
                print(f"{r['shape'].upper():<12} {r['outline']:<10.1%} {r['interior']:<10.1%} {r['total']:<10.1%} {r['mse']:<10.4f}")
            print("="*70)
    else:
        print("\n⚠️  No models could be tested automatically.")
        print("   Please run manually: python diagnose_saved_models.py <model_path> <shape_name>")


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 1:
        # No arguments - find and test all models
        find_and_test_all_models('saved_models')
    else:
        # Test specific model
        model_path = sys.argv[1]
        shape = sys.argv[2] if len(sys.argv) > 2 else 'circle'
        diagnose_model(model_path, shape)