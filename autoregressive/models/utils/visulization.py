import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
import numpy as np

import torch
import torch.distributed as dist
from autoregressive.models.utils.tokens import  TokenMap, TokenMapTensors, \
    TokenType, ImageToken
    
from jaxtyping import Float, Int64
from dataclasses import dataclass

from typing import Optional

import matplotlib as mpl
import numpy as np
from typing import List, Union

def generate_high_contrast_colors(n_colors: int, 
                                 saturation: float = 0.85, 
                                 value: float = 0.75) -> List[str]:
    """
    根据指定数量生成高对比度、非黑白灰的 Hex 颜色代码。

    参数:
        n_colors (int): 想要生成的颜色数量。
        saturation (float): 饱和度 (0.0 到 1.0)。高值避免灰色。
        value (float): 亮度/明度 (0.0 到 1.0)。中等值避免黑色和白色。

    返回:
        List[str]: 包含 N 个 Hex 颜色代码的列表。
    """
    if n_colors <= 0:
        return []
    
    # 1. 在色相环上均匀取点 (Hue)
    # linspace(0, 1, N+1) 会产生 N+1 个点，范围是 [0, 1]。
    # [:-1] 排除末尾的 1.0 (与 0.0 重复)，确保只有 N 个不重复的点。
    hues = np.linspace(0, 1, n_colors + 1)[:-1] 

    # 2. 生成颜色列表
    high_contrast_colors_hex = []

    for h in hues:
        # HSV 坐标 (H, S, V)
        hsv_coords = [h, saturation, value]
        
        # 将 HSV 转换为 RGB
        rgb_coords = mpl.colors.hsv_to_rgb(hsv_coords)
        
        # 将 RGB 转换为 Hex 代码
        high_contrast_colors_hex.append(mpl.colors.to_hex(rgb_coords))

    return high_contrast_colors_hex



def visualize_passes(
        img_width: int, img_height: int, 
        token_map: TokenMap, 
        decoded_masked_coords: list[torch.Tensor], 
        experiment_dir: str
        ):
    # visualization of each pass in the image, visualization of the attention mask
    # --- 参数设置 ---
    ROWS = img_height
    COLS = img_width
    TOTAL_SQUARES = ROWS * COLS
    SQUARE_SIZE = 1  # 每个方块的边长，方便计算坐标

    row_colors = generate_high_contrast_colors(n_colors=len(decoded_masked_coords))

    fig, ax = plt.subplots(figsize=(COLS * SQUARE_SIZE, ROWS * SQUARE_SIZE)) 

    # 4. 设置坐标轴和边界
    # 设置 x 和 y 轴的范围，从 0 到 ROWS/COLS * SQUARE_SIZE
    ax.set_xlim(0, COLS * SQUARE_SIZE)
    ax.set_ylim(0, ROWS * SQUARE_SIZE)

    # 隐藏坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 隐藏坐标轴的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 确保图形紧密，去除白边
    plt.tight_layout(pad=0) 
    len_token_seq = 0

    for pass_id, coords_i_pass in enumerate(decoded_masked_coords):
        current_color = row_colors[pass_id]
        for coord in coords_i_pass.tolist():
            col, row = coord
            output_token_index = token_map.get_output_token_index((ImageToken(col, row)))
            input_token, output_token = token_map._data[output_token_index]
            assert (output_token.x_coord, output_token.y_coord) == (col, row), \
                f"Token map index mismatch at pass {pass_id}, coord ({col}, {row})"
            
            input_token_index = None
            if input_token.token_type() == TokenType.CONDITION:
                input_token_index = "C"
            elif input_token.token_type() == TokenType.IMAGE:
                # input_token_index是要显示在图中的编号
                input_token_index = token_map.get_output_token_index((ImageToken(input_token.x_coord, input_token.y_coord)))
                pass_id_of_input_token = None
                for tmp_pass_id, tmp_coords in enumerate(decoded_masked_coords):
                    if (input_token.x_coord, input_token.y_coord) in [tuple(c) for c in tmp_coords.tolist()]:
                        pass_id_of_input_token = tmp_pass_id
                        break
                assert pass_id_of_input_token is not None, \
                    f"Input token {input_token} not found in any pass"
            else:
                assert False, "Only CONDITION and IMAGE tokens are expected as input tokens"

            
            # 计算方块的左下角坐标 (x, y)
            x = col * SQUARE_SIZE
            # 注意 Matplotlib 的 y 轴方向：row 0 对应图形的顶部，所以 y = (ROWS - 1 - row) * SQUARE_SIZE
            y = (ROWS - 1 - row) * SQUARE_SIZE
            # 2. 绘制方块
            rect = Rectangle((x, y), SQUARE_SIZE, SQUARE_SIZE,
                            facecolor=current_color,
                            edgecolor='white', # 方块间用白色边框分隔，使其看起来“紧密”
                            linewidth=1.5)
            ax.add_patch(rect)

            # 3. 添加编号
            center_x = x + SQUARE_SIZE / 2
            center_y = y + SQUARE_SIZE / 2
            
            ax.text(center_x, center_y, str(output_token_index) + f"\n({col},{row})", 
                    color='black', 
                    fontsize=14, 
                    fontweight='bold',
                    ha='center', # 水平居中
                    va='center') # 垂直居中
        
        # 5. 保存图片
        file_name = f"{experiment_dir}/colored_squares_grid_pass{pass_id}.png"
        plt.savefig(file_name, dpi=300) 

        print(f"Pass {pass_id} 的块网格图已生成并保存为：{file_name}")



def visualize_input_seq(
        img_width: int, img_height: int, 
        token_map: TokenMap,
        token_map_tensors: TokenMapTensors, 
        decoded_masked_coords: list[torch.Tensor],
        experiment_dir: str
        ):
    # visualization of each pass in the image, visualization of the attention mask
    # --- 参数设置 ---
    ROWS = 1
    COLS = img_width * img_height
    TOTAL_SQUARES = ROWS * COLS
    SQUARE_SIZE = 0.8  # 每个方块的边长，方便计算坐标

    row_colors = generate_high_contrast_colors(n_colors=len(decoded_masked_coords))

    con_colors = '#8D33FF' # 紫罗兰色

    fig, ax = plt.subplots(figsize=(COLS * SQUARE_SIZE, ROWS * SQUARE_SIZE)) 

    # 4. 设置坐标轴和边界
    # 设置 x 和 y 轴的范围，从 0 到 ROWS/COLS * SQUARE_SIZE
    ax.set_xlim(0, COLS * SQUARE_SIZE)
    ax.set_ylim(0, ROWS * SQUARE_SIZE)

    # 隐藏坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 隐藏坐标轴的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 确保图形紧密，去除白边
    plt.tight_layout(pad=0) 
    len_token_seq = 0

    image_mask = token_map_tensors.in_token_types == TokenType.IMAGE.value
    cond_mask = token_map_tensors.in_token_types == TokenType.CONDITION.value

    image_indices = token_map_tensors.in_token_indices[image_mask]
    cond_indices = token_map_tensors.in_token_indices[cond_mask]
    print(f"Input sequence contains {len(image_indices)} image tokens and {len(cond_indices)} condition tokens.")
            
    for cond_idx in cond_indices.tolist():
        # 计算方块的左下角坐标 (x, y)
        x = len_token_seq * SQUARE_SIZE
        # 注意 Matplotlib 的 y 轴方向：row 0 对应图形的顶部，所以 y = (ROWS - 1 - row) * SQUARE_SIZE
        y = (ROWS - 1) * SQUARE_SIZE
        # 2. 绘制方块
        rect = Rectangle((x, y), SQUARE_SIZE, SQUARE_SIZE,
                        facecolor=con_colors,
                        edgecolor='white', # 方块间用白色边框分隔，使其看起来“紧密”
                        linewidth=1.5)
        ax.add_patch(rect)

        # 3. 添加编号
        center_x = x + SQUARE_SIZE / 2
        center_y = y + SQUARE_SIZE / 2
        
        ax.text(center_x, center_y, f"C_{cond_idx}", 
                color='black', 
                fontsize=14, 
                fontweight='bold',
                ha='center', # 水平居中
                va='center') # 垂直居中
        
        len_token_seq += 1

    for img_idx in image_indices.tolist():
        current_color = None
        col, row = None, None
        output_token_index = None
        # determine which pass this image token belongs to
        for pass_id, coords_i_pass in enumerate(decoded_masked_coords):
            for coord in coords_i_pass.tolist():
                col, row = coord
                output_token_index = token_map.get_output_token_index((ImageToken(col, row)))
                if row * img_width + col == img_idx:
                    current_color = row_colors[pass_id]
                    break
            if current_color is not None:
                break
        
        # 计算方块的左下角坐标 (x, y)
        x = len_token_seq * SQUARE_SIZE
        # 注意 Matplotlib 的 y 轴方向：row 0 对应图形的顶部，所以 y = (ROWS - 1 - row) * SQUARE_SIZE
        y = (ROWS - 1) * SQUARE_SIZE
        # 2. 绘制方块
        assert current_color is not None, f"image token index {img_idx} not found in any pass"
        rect = Rectangle((x, y), SQUARE_SIZE, SQUARE_SIZE,
                        facecolor=current_color,
                        edgecolor='white', # 方块间用白色边框分隔，使其看起来“紧密”
                        linewidth=1.5)
        ax.add_patch(rect)

        # 3. 添加编号
        center_x = x + SQUARE_SIZE / 2
        center_y = y + SQUARE_SIZE / 2
        
        ax.text(center_x, center_y, str(output_token_index) + f"\n({col},{row})", 
                color='black', 
                fontsize=14, 
                fontweight='bold',
                ha='center', # 水平居中
                va='center') # 垂直居中
        
        len_token_seq += 1
            
        
    # 5. 保存图片
    file_name = f"{experiment_dir}/reordered_input_seq.png"
    plt.savefig(file_name, dpi=100) 

    print(f"重排序的输入序列已生成并保存为：{file_name}")



def visualize_target_seq(
        img_width: int, img_height: int, 
        token_map: TokenMap,
        token_map_tensors: TokenMapTensors, 
        decoded_masked_coords: list[torch.Tensor],
        experiment_dir: str
        ):
    # visualization of each pass in the image, visualization of the attention mask
    # --- 参数设置 ---
    ROWS = 1
    COLS = img_width * img_height
    TOTAL_SQUARES = ROWS * COLS
    SQUARE_SIZE = 0.8  # 每个方块的边长，方便计算坐标

    row_colors = generate_high_contrast_colors(n_colors=len(decoded_masked_coords))

    fig, ax = plt.subplots(figsize=(COLS * SQUARE_SIZE, ROWS * SQUARE_SIZE)) 

    # 4. 设置坐标轴和边界
    # 设置 x 和 y 轴的范围，从 0 到 ROWS/COLS * SQUARE_SIZE
    ax.set_xlim(0, COLS * SQUARE_SIZE)
    ax.set_ylim(0, ROWS * SQUARE_SIZE)

    # 隐藏坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 隐藏坐标轴的边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # 确保图形紧密，去除白边
    plt.tight_layout(pad=0) 
    len_token_seq = 0

    image_mask = token_map_tensors.out_token_types == TokenType.IMAGE.value
    image_indices = token_map_tensors.out_token_indices[image_mask]
    assert len(image_indices) == len(token_map), (
        f"Output sequence should contain all image tokens, but got {len(image_indices)} out of {len(token_map)}"
    )
    
    for img_idx in image_indices.tolist():
        current_color = None
        col, row = None, None
        output_token_index = None
        # determine which pass this image token belongs to
        for pass_id, coords_i_pass in enumerate(decoded_masked_coords):
            for coord in coords_i_pass.tolist():
                col, row = coord
                output_token_index = token_map.get_output_token_index((ImageToken(col, row)))
                if row * img_width + col == img_idx:
                    current_color = row_colors[pass_id]
                    break
            if current_color is not None:
                break
        # 计算方块的左下角坐标 (x, y)
        x = len_token_seq * SQUARE_SIZE
        # 注意 Matplotlib 的 y 轴方向：row 0 对应图形的顶部，所以 y = (ROWS - 1 - row) * SQUARE_SIZE
        y = (ROWS - 1) * SQUARE_SIZE
        # 2. 绘制方块
        assert current_color is not None, f"image token index {img_idx} not found in any pass"
        rect = Rectangle((x, y), SQUARE_SIZE, SQUARE_SIZE,
                        facecolor=current_color,
                        edgecolor='white', # 方块间用白色边框分隔，使其看起来“紧密”
                        linewidth=1.5)
        ax.add_patch(rect)

        # 3. 添加编号
        center_x = x + SQUARE_SIZE / 2
        center_y = y + SQUARE_SIZE / 2
        
        ax.text(center_x, center_y, str(output_token_index) + f"\n({col},{row})", 
                color='black', 
                fontsize=14, 
                fontweight='bold',
                ha='center', # 水平居中
                va='center') # 垂直居中
        
        len_token_seq += 1
            
        
    # 5. 保存图片
    file_name = f"{experiment_dir}/target_seq.png"
    plt.savefig(file_name, dpi=100) 

    print(f"target序列已生成并保存为：{file_name}")

def visualize_attention_mask(
        attention_mask: torch.Tensor,
        experiment_dir: str,
        mask_i: torch.Tensor = None,
        ):
    
    ROWS = attention_mask.shape[0]
    COLS = attention_mask.shape[1]

    COLOR_FOR_ONE = '#008B8B' 
    COLOR_FOR_ZERO = 'white'

    DPI = 250 
    
    if mask_i is not None:
        file_name = f"{experiment_dir}/attention_mask_for_pass_{mask_i}.png"
    else:
        file_name = f"{experiment_dir}/attention_mask.png"

    attention_mask_np = attention_mask.clone().detach().cpu().numpy()

    cmap_colors = [COLOR_FOR_ZERO, COLOR_FOR_ONE] 
    custom_cmap = ListedColormap(cmap_colors)

    fig_size_inches = 256 / DPI 

    fig, ax = plt.subplots(figsize=(fig_size_inches * 10, fig_size_inches * 10)) 

    im = ax.imshow(attention_mask_np, 
                cmap=custom_cmap,           
                aspect='equal',             
                interpolation='none',       
                vmin=0, vmax=1)             

    ax.set_xticks(np.arange(-0.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, ROWS, 1), minor=True)

    ax.grid(which='minor', 
            color='black',       
            linestyle='-',       
            linewidth=0.5)       


    ax.tick_params(which='both', length=0) 

    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    plt.tight_layout(pad=0)

    plt.savefig(file_name, dpi=DPI) 
    plt.close(fig)

    print(f"256x256 二值化 Attention Mask (带网格线) 已生成并保存为：{file_name}")
    

def visualize_sequences(
        img_width: int, img_height: int,
        token_map: TokenMap,
        token_map_tensors: TokenMapTensors,
        decoded_masked_coords: list[torch.Tensor],
        experiment_dir: str
        ):
    # --- 参数设置 ---
    # 现在有两行：第一行是输入序列 (ROWS=0)，第二行是目标序列 (ROWS=1)
    ROWS = 2 
    COLS = img_width * img_height + len(token_map_tensors.in_token_types[token_map_tensors.in_token_types == TokenType.CONDITION.value])
    # 最大的列数以适应最长序列（通常是输入序列，因为它包含条件token）
    
    SQUARE_SIZE = 0.8  # 每个方块的边长

    # 颜色设置
    row_colors = generate_high_contrast_colors(n_colors=len(decoded_masked_coords)) # 用于不同pass的图像token
    con_colors = '#8D33FF' # 紫罗兰色，用于条件token

    # 创建子图，现在的高度是 SQUARE_SIZE * ROWS
    fig, ax = plt.subplots(figsize=(COLS * SQUARE_SIZE, ROWS * SQUARE_SIZE))

    # 4. 设置坐标轴和边界
    ax.set_xlim(0, COLS * SQUARE_SIZE)
    ax.set_ylim(0, ROWS * SQUARE_SIZE) # y轴范围从0到 ROWS * SQUARE_SIZE

    # 隐藏坐标轴的刻度和标签
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # 隐藏坐标轴的边框
    for spine in ax.spines.values():
        spine.set_visible(False)

    # 确保图形紧密，去除白边
    plt.tight_layout(pad=0)
    
    # ----------------------------------------
    # 绘制第一行：输入序列 (Input Sequence)
    # ----------------------------------------
    # 假设输入序列在第一行 (row_to_draw = 1，从底部算起，或根据 Matplotlib 坐标系 y=(ROWS-1-row)*SQUARE_SIZE)
    # 绘图的 y 坐标：对于第一行 (顶部一行)，Matplotlib 的 y 坐标是 (ROWS - 1) * SQUARE_SIZE
    input_seq_y = (ROWS - 1) * SQUARE_SIZE
    
    len_token_seq_in = 0

    image_mask_in = token_map_tensors.in_token_types == TokenType.IMAGE.value
    cond_mask_in = token_map_tensors.in_token_types == TokenType.CONDITION.value

    cond_indices = token_map_tensors.in_token_indices[cond_mask_in]
    image_indices_in = token_map_tensors.in_token_indices[image_mask_in]
    
    print(f"Input sequence contains {len(image_indices_in)} image tokens and {len(cond_indices)} condition tokens.")
            
    # 1. 绘制条件 token
    for cond_idx in cond_indices.tolist():
        x = len_token_seq_in * SQUARE_SIZE
        draw_token(ax, x, input_seq_y, SQUARE_SIZE, con_colors, f"C_{cond_idx}")
        len_token_seq_in += 1

    # 2. 绘制图像 token
    for img_idx in image_indices_in.tolist():
        current_color, col, row, output_token_index = get_token_info(
            img_idx, img_width, token_map, decoded_masked_coords, row_colors
        )
        
        x = len_token_seq_in * SQUARE_SIZE
        label = str(output_token_index) + f"\n({col},{row})"
        draw_token(ax, x, input_seq_y, SQUARE_SIZE, current_color, label)
        len_token_seq_in += 1
        
    # 添加序列标签
    ax.text(-0.5 * SQUARE_SIZE, input_seq_y + SQUARE_SIZE / 2, "Input\nSeq", 
            color='black', fontsize=16, fontweight='bold', ha='right', va='center')


    # ----------------------------------------
    # 绘制第二行：目标序列 (Target Sequence)
    # ----------------------------------------
    # 绘图的 y 坐标：对于第二行 (底部一行)，Matplotlib 的 y 坐标是 0 * SQUARE_SIZE
    target_seq_y = 0 * SQUARE_SIZE 
    
    len_token_seq_out = 0

    image_mask_out = token_map_tensors.out_token_types == TokenType.IMAGE.value
    image_indices_out = token_map_tensors.out_token_indices[image_mask_out]

    assert len(image_indices_out) == img_width * img_height, (
        f"Output sequence should contain all image tokens, but got {len(image_indices_out)} out of {img_width * img_height}"
    )
    
    for img_idx in image_indices_out.tolist():
        current_color, col, row, output_token_index = get_token_info(
            img_idx, img_width, token_map, decoded_masked_coords, row_colors
        )
        
        x = len_token_seq_out * SQUARE_SIZE
        label = str(output_token_index) + f"\n({col},{row})"
        draw_token(ax, x, target_seq_y, SQUARE_SIZE, current_color, label)
        len_token_seq_out += 1
            
    # 添加序列标签
    ax.text(-0.5 * SQUARE_SIZE, target_seq_y + SQUARE_SIZE / 2, "Target\nSeq", 
            color='black', fontsize=16, fontweight='bold', ha='right', va='center')

        
    # 5. 保存图片
    file_name = f"{experiment_dir}/reordered_and_target_sequences.png"
    plt.savefig(file_name, dpi=100)

    print(f"输入序列和目标序列已生成并保存为：{file_name}")

# --- 辅助函数 (需要确保你的环境中有这些定义，例如 matplotlib.patches.Rectangle) ---

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import torch # 假设 torch, TokenMap, TokenMapTensors, ImageToken, TokenType 已在环境中定义

# 绘制单个 token 的辅助函数
def draw_token(ax, x, y, size, color, text_label):
    rect = Rectangle((x, y), size, size,
                     facecolor=color,
                     edgecolor='white',
                     linewidth=1.5)
    ax.add_patch(rect)

    center_x = x + size / 2
    center_y = y + size / 2
    ax.text(center_x, center_y, text_label,
            color='black',
            fontsize=14,
            fontweight='bold',
            ha='center',
            va='center')

# 获取图像 token 信息的辅助函数
def get_token_info(img_idx, img_width, token_map, decoded_masked_coords, row_colors):
    current_color = None
    col, row = None, None
    output_token_index = None
    
    # 确定图像 token 属于哪个 pass
    for pass_id, coords_i_pass in enumerate(decoded_masked_coords):
        for coord in coords_i_pass.tolist():
            c, r = coord
            # 这里的逻辑是将 (c, r) 映射到 output_token_index
            out_idx = token_map.get_output_token_index((ImageToken(c, r)))
            
            # 检查这个图像 token 是否就是 img_idx 对应的 token
            if r * img_width + c == img_idx:
                current_color = row_colors[pass_id]
                col, row = c, r
                output_token_index = out_idx
                break
        if current_color is not None:
            break
            
    assert current_color is not None, f"image token index {img_idx} not found in any pass"
    return current_color, col, row, output_token_index