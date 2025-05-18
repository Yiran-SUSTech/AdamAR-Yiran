import token
from tokenize import Token
from typing import OrderedDict

from autoregressive.models.utils import TokenMap, TokenMapTensors_v2, TokenType
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors

def to_xy(index: int, width: int):
    y = index // width
    x = index % width
    return x, y

def generate_colors(n):
    cmap = plt.get_cmap('hsv')
    return [cmap(i / n) for i in range(n)]

def visualize_token_map(token_map: TokenMap | TokenMapTensors_v2,
                        width: int,
                        height: int,
                        cond_len: int,
                        decoding_schedule: list[list[int]] | None = None):
    

    if isinstance(token_map, TokenMap):
        token_map = TokenMapTensors_v2(token_map, width, height)
    
    if decoding_schedule is not None:
        num_decoding_step = len(decoding_schedule)
        colors = generate_colors(num_decoding_step)
        
    image_len = width * height
    output_image_mask = (token_map.out_token_types == TokenType.IMAGE.value)
    output_indices = token_map.out_token_indices[output_image_mask]
    
    vis_map_indices = torch.zeros(image_len, dtype=torch.int)
    vis_map_type = torch.zeros(image_len, dtype=torch.int)
    
    vis_map_indices[output_indices] = token_map.in_token_indices[output_image_mask]    
    vis_map_type[output_indices] = token_map.in_token_types[output_image_mask] + torch.iinfo(torch.int).min

    if decoding_schedule is not None:
        for step, decoded_indices in enumerate(decoding_schedule):
            plt.figure()
            plt.imshow(vis_map_indices.reshape((height, width)), cmap='gray')
            plt.title(f"Token map at step {step}")
            plt.colorbar()

            for i in decoded_indices:
                if not output_image_mask[i]:
                    continue

                out_idx = token_map.out_token_indices[i].item()
                in_idx = token_map.in_token_indices[i].item()

                out_x, out_y = to_xy(out_idx, width)
                in_x, in_y = to_xy(in_idx, width)

                # Draw arrow from input to output
                dx = out_x - in_x
                dy = out_y - in_y

                if step == 4:
                    print(f"Step {step}: Drawing arrow from ({in_x}, {in_y}) to ({out_x}, {out_y})")
                plt.arrow(in_x, in_y, dx, dy, color='red', head_width=0.5, length_includes_head=True, alpha=0.7)

            plt.savefig(f"step_{step}.jpg")
            plt.close()


    cmap = mcolors.ListedColormap(['lightgray', 'steelblue', 'salmon'])
    bounds = [-0.5, 0.5, 1.5, 3.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Map 3 to index 2 in cmap, so vis_map_type needs no remapping
    plt.figure()
    plt.imshow(vis_map_type.reshape((height, width)), cmap=cmap, norm=norm)

    # Set ticks at the correct values
    cbar = plt.colorbar(ticks=[0, 1, 3])
    cbar.ax.set_yticklabels(['image', 'learned', 'condition'])  # Human-readable labels

    plt.title("Token Map Types")
    plt.savefig("token_map_types.png")
    plt.close()

def visualize_attention_mask(attention_mask: torch.Tensor,
                             token_map: TokenMap | TokenMapTensors_v2,
                             decoding_schedule: list[list[int]] | None,
                             height: int,
                             width: int):

    if isinstance(token_map, TokenMap):
        token_map = TokenMapTensors_v2(token_map, width, height)
        
    image_len = width * height
    
    output_image_mask = (token_map.out_token_types == TokenType.IMAGE.value)
    output_indices = token_map.out_token_indices[output_image_mask]

    input_image_mask = (token_map.in_token_types == TokenType.IMAGE.value)
    input_indices = token_map.in_token_indices[input_image_mask]
    h, w = torch.meshgrid(output_indices, output_indices)
    # attention_mask_no_cond = attention_mask[h, w]
    
    vis_map_indices = torch.zeros(image_len, dtype=torch.int)
    vis_map_type = torch.zeros(image_len, dtype=torch.int)
    
    vis_map_indices[output_indices] = token_map.in_token_indices[output_image_mask]    
    vis_map_type[output_indices] = token_map.in_token_types[output_image_mask] + torch.iinfo(torch.int).min

    if decoding_schedule is not None:
        for step, decoded_indices in enumerate(decoding_schedule):
            plt.figure()
            plt.title(f"Attention map {step}")

            attended_mask = attention_mask[decoded_indices, :]
            attended_image_mask = torch.any(attended_mask[:, input_image_mask], dim=0).int()
            
            vis_mask =  torch.zeros(image_len, dtype=torch.int32)
            for i in range(len(attended_image_mask)):
                if attended_image_mask[i]:
                    vis_mask[input_indices[i]] =1
 
            plt.imshow(vis_mask.reshape((height, width)), cmap='gray')
            plt.colorbar()

            plt.savefig(f"attention_step_{step}.jpg")
            plt.close()

