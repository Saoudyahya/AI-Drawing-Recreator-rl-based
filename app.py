import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import glob
from pathlib import Path
import io
import time
from scipy import ndimage
from skimage import morphology


# ============================================================================
# COPY OF CORE CLASSES FROM ORIGINAL CODE (needed for model loading)
# ============================================================================

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


class DrawingEnvironment:
    """Full drawing environment matching training code - REQUIRED for correct inference"""

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
        """Set target with edge detection (CRITICAL for model to work correctly)"""
        self.target = target
        target_mask = (target[:, :, 0] > 0.5) & (target[:, :, 1] < 0.5) & (target[:, :, 2] < 0.5)

        # Edge detection - same as training!
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
        """Compute outline and interior coverage separately - CRITICAL"""
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

        # Cursor indicator
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                    state[ny, nx, 3] = 1.0

        # CRITICAL: Separate outline and interior coverage (was broken before!)
        outline_cov, interior_cov, _ = self.compute_coverage()
        state[:, :, 4] = outline_cov  # Channel 4: outline coverage
        state[:, :, 5] = interior_cov  # Channel 5: interior coverage

        # Phase switching at 70% outline coverage
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


# ============================================================================
# SHAPE GENERATORS
# ============================================================================

class ShapeGenerator:
    @staticmethod
    def create_circle(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        y, x = np.ogrid[:canvas_size, :canvas_size]
        mask = (x - canvas_size // 2) ** 2 + (y - canvas_size // 2) ** 2 <= 20 ** 2
        target[mask] = [1.0, 0.0, 0.0]
        return target

    @staticmethod
    def create_square(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        size = 36
        start = (canvas_size - size) // 2
        end = start + size
        target[start:end, start:end] = [1.0, 0.0, 0.0]
        return target

    @staticmethod
    def create_triangle(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        img = Image.fromarray((target * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        center = canvas_size // 2
        size = 28
        height = int(size * np.sqrt(3) / 2)
        points = [
            (center, center - height * 2 // 3),
            (center - size, center + height // 3),
            (center + size, center + height // 3)
        ]
        draw.polygon(points, fill=(255, 0, 0))
        target = np.array(img).astype(np.float32) / 255.0
        return target

    @staticmethod
    def create_diamond(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        img = Image.fromarray((target * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        center = canvas_size // 2
        size = 24
        points = [
            (center, center - size),
            (center + size, center),
            (center, center + size),
            (center - size, center)
        ]
        draw.polygon(points, fill=(255, 0, 0))
        target = np.array(img).astype(np.float32) / 255.0
        return target

    @staticmethod
    def create_star(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        img = Image.fromarray((target * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        center = canvas_size // 2
        outer_radius = 22
        inner_radius = 9
        points = []
        for i in range(10):
            angle = i * np.pi / 5 - np.pi / 2
            r = outer_radius if i % 2 == 0 else inner_radius
            x = center + r * np.cos(angle)
            y = center + r * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=(255, 0, 0))
        target = np.array(img).astype(np.float32) / 255.0
        return target

    @staticmethod
    def create_heart(canvas_size=64):
        target = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
        y, x = np.ogrid[:canvas_size, :canvas_size]
        center_x, center_y = canvas_size // 2, canvas_size // 2 + 3
        scale = 15
        mask = np.zeros((canvas_size, canvas_size), dtype=bool)
        for i in range(canvas_size):
            for j in range(canvas_size):
                nx = (j - center_x) / scale
                ny = -(i - center_y) / scale
                val = (nx ** 2 + ny ** 2 - 1) ** 3 - nx ** 2 * ny ** 3
                if val <= 0.1:
                    mask[i, j] = True
        target[mask] = [1.0, 0.0, 0.0]
        return target


# ============================================================================
# STREAMLIT APP
# ============================================================================

def find_available_models(models_dir='saved_models'):
    """Find all available trained models"""
    models = {}
    if not os.path.exists(models_dir):
        return models

    import glob

    for shape_name in ['circle', 'square', 'triangle', 'diamond', 'star', 'heart']:
        # Look for final models with timestamp pattern
        pattern = os.path.join(models_dir, f'{shape_name}_final_*.pt')
        final_models = glob.glob(pattern)

        if final_models:
            # Get most recent final model
            models[shape_name] = max(final_models, key=os.path.getmtime)
        else:
            # Fall back to epoch models
            pattern = os.path.join(models_dir, f'{shape_name}_epoch*.pt')
            epoch_models = glob.glob(pattern)
            if epoch_models:
                models[shape_name] = max(epoch_models, key=os.path.getmtime)

    return models


def load_model(model_path, device='cpu'):
    """Load a trained model"""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    policy = ImprovedTransformerPolicy(
        canvas_size=64, grid_size=20, patch_size=8,
        embed_dim=256, depth=8, num_heads=8, mlp_ratio=4.0, dropout=0.1
    ).to(device)

    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    return policy, checkpoint


def run_inference(policy, target_image, device='cpu', max_steps=250):
    """Run inference with the trained model"""
    env = DrawingEnvironment(canvas_size=64, grid_size=20)
    env.set_target(target_image)

    state = env.reset()
    done = False
    step = 0

    frames = [env.canvas.copy()]
    outline_covs = []
    interior_covs = []
    total_covs = []

    # Get initial coverage
    out_cov, int_cov, tot_cov = env.compute_coverage()
    outline_covs.append(out_cov)
    interior_covs.append(int_cov)
    total_covs.append(tot_cov)

    with torch.no_grad():
        while not done and step < max_steps:
            state_tensor = torch.FloatTensor(state).to(device)
            action_logits, _ = policy(state_tensor)
            action = torch.argmax(action_logits).item()

            state, done = env.step(action)
            step += 1

            frames.append(env.canvas.copy())
            out_cov, int_cov, tot_cov = env.compute_coverage()
            outline_covs.append(out_cov)
            interior_covs.append(int_cov)
            total_covs.append(tot_cov)

    return frames, (outline_covs, interior_covs, total_covs), env.canvas


def create_drawing_canvas():
    """Create a simple drawing interface"""
    canvas_size = 64

    # Initialize canvas in session state
    if 'drawn_canvas' not in st.session_state:
        st.session_state.drawn_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)

    st.write("**Draw on the canvas below:**")
    st.write("_(Note: This is a simplified drawing interface. Click to toggle pixels)_")

    # Create a grid for simple pixel drawing
    cols = st.columns(8)
    scale_factor = canvas_size // 8

    for i in range(8):
        for j in range(8):
            with cols[j]:
                # Check if this region is mostly red
                region = st.session_state.drawn_canvas[i * scale_factor:(i + 1) * scale_factor,
                j * scale_factor:(j + 1) * scale_factor]
                is_red = np.mean(region[:, :, 0]) > 0.7 and np.mean(region[:, :, 1]) < 0.3

                if st.button("🔴" if is_red else "⚪", key=f"pixel_{i}_{j}"):
                    # Toggle this region
                    if is_red:
                        st.session_state.drawn_canvas[i * scale_factor:(i + 1) * scale_factor,
                        j * scale_factor:(j + 1) * scale_factor] = [1.0, 1.0, 1.0]
                    else:
                        st.session_state.drawn_canvas[i * scale_factor:(i + 1) * scale_factor,
                        j * scale_factor:(j + 1) * scale_factor] = [1.0, 0.0, 0.0]
                    st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Canvas"):
            st.session_state.drawn_canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.float32)
            st.rerun()

    return st.session_state.drawn_canvas


def main():
    st.set_page_config(page_title="🎨 AI Shape Painter", layout="wide")

    st.title("🎨 AI Transformer Shape Painter")
    st.markdown("### Watch an AI learn to draw shapes using Vision Transformers!")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Find available models
    models_dir = st.sidebar.text_input("Models Directory", "saved_models")
    available_models = find_available_models(models_dir)

    if not available_models:
        st.error(f"❌ No trained models found in '{models_dir}'!")
        st.info("📝 Please run the training script first to generate models.")
        st.code("python your_training_script.py", language="bash")
        return

    st.sidebar.success(f"✅ Found {len(available_models)} trained models")

    # Mode selection
    mode = st.sidebar.radio("Select Mode", ["Pre-trained Shape", "Custom Drawing"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"🖥️ Using: {device.upper()}")

    # Shape selection or custom drawing
    if mode == "Pre-trained Shape":
        shape_name = st.sidebar.selectbox(
            "Choose a shape",
            list(available_models.keys()),
            format_func=lambda x: x.capitalize()
        )

        # Generate target shape
        shape_generators = {
            'circle': ShapeGenerator.create_circle,
            'square': ShapeGenerator.create_square,
            'triangle': ShapeGenerator.create_triangle,
            'diamond': ShapeGenerator.create_diamond,
            'star': ShapeGenerator.create_star,
            'heart': ShapeGenerator.create_heart
        }

        target_image = shape_generators[shape_name](64)
        model_path = available_models[shape_name]

    else:  # Custom Drawing
        st.sidebar.info("⚠️ Drawing mode uses the 'circle' model as default")
        target_image = create_drawing_canvas()

        # Use circle model for custom drawings (or let user choose)
        shape_name = st.sidebar.selectbox(
            "Which model to use?",
            list(available_models.keys()),
            format_func=lambda x: x.capitalize()
        )
        model_path = available_models[shape_name]

    # Animation settings
    st.sidebar.header("🎬 Animation")
    show_animation = st.sidebar.checkbox("Show Step-by-Step Animation", True)
    animation_speed = st.sidebar.slider("Animation Speed (FPS)", 1, 30, 10)
    max_steps = st.sidebar.slider("Max Steps", 50, 250, 250)

    # Main content
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Target Shape")
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(target_image)
        ax.axis('off')
        ax.set_title(f"Target: {shape_name.capitalize()}", fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    # Run button
    if st.button("🚀 Start Drawing!", type="primary", use_container_width=True):
        with st.spinner("🤖 Loading model..."):
            policy, checkpoint = load_model(model_path, device)

        st.success(f"✅ Model loaded: {shape_name.capitalize()}")

        # Show model info
        with st.expander("📊 Model Information"):
            best_mse = checkpoint.get('best_mse', None)
            best_outline = checkpoint.get('best_outline_cov', None)
            best_interior = checkpoint.get('best_interior_cov', None)
            best_score = checkpoint.get('best_total_score', None)

            st.write(f"**Best MSE:** {best_mse:.4f}" if best_mse is not None else "**Best MSE:** N/A")
            st.write(
                f"**Best Outline Coverage:** {best_outline:.1%}" if best_outline is not None else "**Best Outline Coverage:** N/A")
            st.write(
                f"**Best Interior Coverage:** {best_interior:.1%}" if best_interior is not None else "**Best Interior Coverage:** N/A")
            st.write(
                f"**Best Total Score:** {best_score:.3f}" if best_score is not None else "**Best Total Score:** N/A")

        # Run inference
        with st.spinner("🎨 Drawing in progress..."):
            frames, coverages_tuple, final_canvas = run_inference(
                policy, target_image, device, max_steps
            )

        outline_covs, interior_covs, total_covs = coverages_tuple

        st.success(f"✅ Drawing complete! ({len(frames)} steps)")

        # Display results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("🎯 Target")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(target_image)
            ax.axis('off')
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("🎨 AI Output")
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(final_canvas)
            ax.axis('off')
            ax.set_title(
                f"Total Coverage: {total_covs[-1]:.1%}\nOutline: {outline_covs[-1]:.1%} | Interior: {interior_covs[-1]:.1%}",
                fontsize=10)
            st.pyplot(fig)
            plt.close()

        with col3:
            st.subheader("📊 Difference")
            diff = np.abs(target_image - final_canvas)
            mse = np.mean(diff ** 2)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(diff, cmap='hot')
            ax.axis('off')
            ax.set_title(f"MSE: {mse:.4f}", fontsize=12)
            st.pyplot(fig)
            plt.close()

        # Show animation
        if show_animation and len(frames) > 1:
            st.subheader("🎬 Drawing Animation")

            # Create placeholder for animation
            animation_placeholder = st.empty()
            progress_bar = st.progress(0)
            coverage_chart = st.empty()

            # Animate
            for i in range(len(frames)):
                frame = frames[i]
                tot_cov = total_covs[i]
                out_cov = outline_covs[i]
                int_cov = interior_covs[i]

                # Update canvas
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(frame)
                ax.axis('off')
                ax.set_title(
                    f"Step {i}/{len(frames) - 1} | Total: {tot_cov:.1%} (Out: {out_cov:.1%}, Int: {int_cov:.1%})",
                    fontsize=14, fontweight='bold')
                animation_placeholder.pyplot(fig)
                plt.close()

                # Update progress
                progress_bar.progress((i + 1) / len(frames))

                # Update coverage chart
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.plot(total_covs[:i + 1], linewidth=2, color='#3498db', label='Total')
                ax.plot(outline_covs[:i + 1], linewidth=2, color='#e74c3c', label='Outline', alpha=0.7)
                ax.plot(interior_covs[:i + 1], linewidth=2, color='#2ecc71', label='Interior', alpha=0.7)
                ax.set_xlim(0, len(frames))
                ax.set_ylim(0, 1)
                ax.set_xlabel('Step')
                ax.set_ylabel('Coverage')
                ax.set_title('Coverage Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                coverage_chart.pyplot(fig)
                plt.close()

                # Control speed
                time.sleep(1.0 / animation_speed)

            st.balloons()

        # Download option
        st.subheader("💾 Download Results")

        # Create comparison image
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(target_image)
        axes[0].set_title('Target')
        axes[0].axis('off')

        axes[1].imshow(final_canvas)
        axes[1].set_title(
            f'AI Output\nTotal: {total_covs[-1]:.1%} (Out: {outline_covs[-1]:.1%}, Int: {interior_covs[-1]:.1%})')
        axes[1].axis('off')

        diff = np.abs(target_image - final_canvas)
        axes[2].imshow(diff, cmap='hot')
        axes[2].set_title(f'Difference (MSE: {np.mean(diff ** 2):.4f})')
        axes[2].axis('off')

        plt.tight_layout()

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()

        st.download_button(
            label="📥 Download Comparison",
            data=buf,
            file_name=f"{shape_name}_comparison.png",
            mime="image/png"
        )


if __name__ == "__main__":
    main()