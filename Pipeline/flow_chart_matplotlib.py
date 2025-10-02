import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(17, 12))
ax.set_xlim(0, 17)
ax.set_ylim(0, 12)
ax.axis('off')

# Define colors for different stages
stage_colors = ['#E3F2FD', '#F3E5F5', '#E8F5E8', '#FFF3E0']
detail_colors = ['#BBDEFB', '#E1BEE7', '#C8E6C8', '#FFE0B2']

# Function to create a box
def create_box(ax, x, y, width, height, text, color, text_size=10):
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle="round,pad=0.1",
                        facecolor=color,
                        edgecolor='black',
                        linewidth=1.5)
    ax.add_patch(box)
    
    # Add text
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center',
            fontsize=text_size, fontweight='bold',
            wrap=True)

# Function to draw arrow
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))

# Main stages
create_box(ax, 1, 9, 3, 1.5, 'Giai đoạn 1:\nThu thập & Tiền xử lý\n(Keypoint)', stage_colors[0], 12)
create_box(ax, 5, 9, 3, 1.5, 'Giai đoạn 2:\nMô hình Hóa Chuỗi\n(Transformer)', stage_colors[1], 12)
create_box(ax, 9, 9, 3, 1.5, 'Giai đoạn 3:\nPhân loại & Loss\n(CTC)', stage_colors[2], 12)
create_box(ax, 13, 9, 3, 1.5, 'Giai đoạn 4:\nHậu xử lý & Giải mã\n(Decoding)', stage_colors[3], 12)

# Stage 1 details
create_box(ax, 1.5, 7, 2, 0.8, 'Thu thập video\n(cử chỉ tay)', detail_colors[0])
create_box(ax, 1.5, 5.8, 2, 0.8, 'Ước tính Keypoint\n(MediaPipe Hands)', detail_colors[0])
create_box(ax, 1.5, 4.6, 2, 0.8, 'Chuẩn hóa tọa độ\n(Dịch chuyển, Chia tỷ lệ)', detail_colors[0])
create_box(ax, 1.5, 3.4, 2, 0.8, 'Đầu ra:\nChuỗi vector keypoint', detail_colors[0])

# Stage 2 details
create_box(ax, 5.5, 7, 2, 0.8, 'Positional Encoding', detail_colors[1])
create_box(ax, 5.5, 5.8, 2, 0.8, 'Transformer Encoder\n(Self-Attention, Dropout)', detail_colors[1])
create_box(ax, 5.5, 4.6, 2, 0.8, 'Đầu ra:\nContextualized Features', detail_colors[1])

# Stage 3 details
create_box(ax, 9.5, 7, 2, 0.8, 'Linear Classifier', detail_colors[2])
create_box(ax, 9.5, 5.8, 2, 0.8, 'CTC Loss\n(không cần alignment)', detail_colors[2])
create_box(ax, 9.5, 4.6, 2, 0.8, 'Đầu ra:\nLogits cho mỗi ký hiệu', detail_colors[2])

# Stage 4 details
create_box(ax, 13.5, 7, 2, 0.8, 'CTC Decoding', detail_colors[3])
create_box(ax, 13.5, 5.8, 2, 0.8, 'Beam Search Decoding', detail_colors[3])
create_box(ax, 13.5, 4.6, 2, 0.8, 'Đầu ra:\nVăn bản hoàn chỉnh', detail_colors[3])

# Horizontal arrows between main stages
draw_arrow(ax, 4, 9.75, 5, 9.75)
draw_arrow(ax, 8, 9.75, 9, 9.75)
draw_arrow(ax, 12, 9.75, 13, 9.75)

# Vertical arrows from main stages to details
draw_arrow(ax, 2.5, 9, 2.5, 7.8)
draw_arrow(ax, 6.5, 9, 6.5, 7.8)
draw_arrow(ax, 10.5, 9, 10.5, 7.8)
draw_arrow(ax, 14.5, 9, 14.5, 7.8)

# Arrows within each stage
# Stage 1
draw_arrow(ax, 2.5, 7, 2.5, 6.6)
draw_arrow(ax, 2.5, 5.8, 2.5, 5.4)
draw_arrow(ax, 2.5, 4.6, 2.5, 4.2)

# Stage 2
draw_arrow(ax, 6.5, 7, 6.5, 6.6)
draw_arrow(ax, 6.5, 5.8, 6.5, 5.4)

# Stage 3
draw_arrow(ax, 10.5, 7, 10.5, 6.6)
draw_arrow(ax, 10.5, 5.8, 10.5, 5.4)

# Stage 4
draw_arrow(ax, 14.5, 7, 14.5, 6.6)
draw_arrow(ax, 14.5, 5.8, 14.5, 5.4)


# Add title
ax.text(8, 11.5, 'Pipeline Nhận diện và Chuyển đổi Ký hiệu Tay', 
        ha='center', va='center', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('pipeline_hand_sign_text_matplotlib.png', dpi=300, bbox_inches='tight')
plt.show()
