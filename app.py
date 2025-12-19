import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
import io
import base64
from streamlit_drawable_canvas import st_canvas

# Set page config
st.set_page_config(page_title="AI Drawing Recreator", layout="wide", page_icon="🎨")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDrawingNetwork(nn.Module):
    """Lightweight network for quick inference"""

    def __init__(self, canvas_size=64, grid_size=16):
        super().__init__()
        self.grid_size = grid_size
        self.num_actions = grid_size * grid_size + 1

        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)

        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, state):
        if len(state.shape) == 3:
            state = state.unsqueeze(0)

        state = state.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)
        return self.fc(x)


class SimpleDrawingEnv:
    """Simplified drawing environment"""

    def __init__(self, canvas_size=64, grid_size=16):
        self.canvas_size = canvas_size
        self.grid_size = grid_size
        self.cell_size = canvas_size // grid_size
        self.target = None
        self.reset()

    def set_target(self, target):
        """Set the target image to recreate"""
        self.target = target
        # Convert to grayscale mask
        if len(target.shape) == 3:
            self.target_mask = np.mean(target, axis=2) < 0.5
        else:
            self.target_mask = target < 0.5

    def reset(self):
        self.canvas = np.ones((self.canvas_size, self.canvas_size, 3), dtype=np.float32)
        self.cursor_grid_x = self.grid_size // 2
        self.cursor_grid_y = self.grid_size // 2
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        state = np.zeros((self.canvas_size, self.canvas_size, 4), dtype=np.float32)
        state[:, :, :3] = self.canvas

        # Add cursor position
        cx = self.cursor_grid_x * self.cell_size + self.cell_size // 2
        cy = self.cursor_grid_y * self.cell_size + self.cell_size // 2
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.canvas_size and 0 <= ny < self.canvas_size:
                    state[ny, nx, 3] = 1.0

        return state

    def grid_to_canvas(self, grid_x, grid_y):
        return grid_x * self.cell_size + self.cell_size // 2, grid_y * self.cell_size + self.cell_size // 2

    def draw_stroke(self, x1, y1, x2, y2, color=[0, 0, 0], width=3):
        img = Image.fromarray((self.canvas * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        draw.line([x1, y1, x2, y2], fill=tuple((np.array(color) * 255).astype(int)), width=width)
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
        self.step_count += 1

        done = self.step_count >= 80
        return self.get_state(), done


def simple_ai_recreate(target_image, canvas_size=64, grid_size=16, max_steps=80):
    """Simple AI recreation using contour following"""
    env = SimpleDrawingEnv(canvas_size, grid_size)
    env.set_target(target_image)

    # Get target mask
    if len(target_image.shape) == 3:
        target_gray = np.mean(target_image, axis=2)
    else:
        target_gray = target_image

    target_mask = target_gray < 0.5

    # Find edge pixels
    from scipy import ndimage
    edges = ndimage.sobel(target_mask.astype(float))
    edge_points = np.argwhere(edges > 0.5)

    if len(edge_points) == 0:
        # If no edges, try to fill the shape
        edge_points = np.argwhere(target_mask)

    # Sample points strategically
    if len(edge_points) > max_steps:
        indices = np.linspace(0, len(edge_points) - 1, max_steps, dtype=int)
        edge_points = edge_points[indices]

    # Convert to grid coordinates and draw
    state = env.reset()
    trajectory = []

    for point in edge_points:
        y, x = point
        grid_x = int(x / env.cell_size)
        grid_y = int(y / env.cell_size)
        grid_x = np.clip(grid_x, 0, grid_size - 1)
        grid_y = np.clip(grid_y, 0, grid_size - 1)

        action = grid_y * grid_size + grid_x
        trajectory.append((grid_x, grid_y))
        state, done = env.step(action)

        if done:
            break

    return env.canvas, trajectory


def process_canvas_image(canvas_data, target_size=64):
    """Process the drawable canvas output"""
    if canvas_data is None:
        return None

    # Convert to PIL Image
    img = Image.fromarray(canvas_data.astype('uint8'), 'RGBA')

    # Convert to grayscale on white background
    background = Image.new('RGB', img.size, (255, 255, 255))
    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask

    # Resize to target size
    img_resized = background.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Convert to numpy array
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    return img_array


def main():
    # Header
    st.title("🎨 AI Drawing Recreator")
    st.markdown("""
    **Draw something on the left canvas**, and watch the AI try to recreate it on the right! 
    The AI uses a simplified reinforcement learning approach to trace your drawing.
    """)

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    canvas_size = st.sidebar.slider("Canvas Size", 200, 500, 400, 50)
    stroke_width = st.sidebar.slider("Brush Width", 1, 20, 5)
    drawing_mode = st.sidebar.selectbox("Drawing Mode", ["freedraw", "line", "circle", "rect"])

    # Color picker
    stroke_color = st.sidebar.color_picker("Brush Color", "#000000")

    # Convert hex to RGB
    stroke_color_rgb = tuple(int(stroke_color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👤 Your Drawing")

        # Create canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=canvas_size,
            width=canvas_size,
            drawing_mode=drawing_mode,
            key="canvas",
        )

    with col2:
        st.subheader("🤖 AI Recreation")

        # Placeholder for AI output
        ai_placeholder = st.empty()

        # Process button
        if st.button("🎨 Let AI Recreate!", type="primary", use_container_width=True):
            if canvas_result.image_data is not None:
                with st.spinner("AI is analyzing your drawing..."):
                    # Process the canvas
                    processed_img = process_canvas_image(canvas_result.image_data, target_size=64)

                    if processed_img is not None:
                        # AI recreation
                        ai_canvas, trajectory = simple_ai_recreate(processed_img, canvas_size=64, grid_size=16,
                                                                   max_steps=80)

                        # Resize for display
                        ai_canvas_display = Image.fromarray((ai_canvas * 255).astype(np.uint8))
                        ai_canvas_display = ai_canvas_display.resize((canvas_size, canvas_size),
                                                                     Image.Resampling.NEAREST)

                        # Display
                        ai_placeholder.image(ai_canvas_display, use_container_width=True)

                        # Show stats
                        st.success(f"✓ AI completed drawing in {len(trajectory)} strokes!")

                        # Comparison metrics
                        processed_gray = np.mean(processed_img, axis=2) if len(
                            processed_img.shape) == 3 else processed_img
                        ai_gray = np.mean(ai_canvas, axis=2) if len(ai_canvas.shape) == 3 else ai_canvas

                        user_pixels = np.sum(processed_gray < 0.5)
                        ai_pixels = np.sum(ai_gray < 0.5)

                        if user_pixels > 0:
                            overlap = np.sum((processed_gray < 0.5) & (ai_gray < 0.5))
                            similarity = overlap / user_pixels * 100
                            st.info(f"📊 Similarity: {similarity:.1f}%")
                    else:
                        st.warning("Please draw something first!")
            else:
                st.warning("Canvas is empty! Draw something first.")

    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. **Draw** something on the left canvas using your mouse/touch
        2. **Click** the "Let AI Recreate!" button
        3. **Watch** as the AI attempts to recreate your drawing on the right

        **Tips:**
        - Simple shapes work best (circles, squares, stars, letters)
        - Clear, bold strokes are easier for the AI to follow
        - Try different drawing modes in the sidebar
        - The AI traces edges and contours of your drawing

        **How it works:**
        The AI uses computer vision to detect edges in your drawing, then plans a path
        through those edges to recreate the image. It's inspired by how artists trace
        outlines before filling in details!
        """)

    # Example drawings
    with st.expander("💡 Example Ideas"):
        st.markdown("""
        Try drawing these:
        - ⭐ A star
        - ❤️ A heart  
        - 😊 A smiley face
        - 🏠 A simple house
        - ✋ Your initials
        - 🌙 A crescent moon
        - 🔺 Geometric shapes
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Streamlit | AI powered by Computer Vision & RL techniques
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()