import token
from tokenize import Token
from typing import OrderedDict

from sympy import O
from autoregressive.models.utils import TokenMap, TokenMapTensors_v2, TokenType
import matplotlib.pyplot as plt
import torch
import matplotlib.colors as mcolors

def visualize_token_map(token_map: TokenMap | TokenMapTensors_v2,
                        width: int,
                        height: int,
                        cond_len: int):
    
    if isinstance(token_map, TokenMap):
        token_map = TokenMapTensors_v2(token_map, width, height)
    
    image_len = width * height
    output_image_mask = (token_map.out_token_types == TokenType.IMAGE.value)
    output_indices = token_map.out_token_indices[output_image_mask]
    
    vis_map_indices = torch.zeros(image_len, dtype=torch.int)
    vis_map_type = torch.zeros(image_len, dtype=torch.int)
    
    vis_map_indices[output_indices] = token_map.in_token_indices[output_image_mask]
    vis_map_type[output_indices] = token_map.in_token_types[output_image_mask] + torch.iinfo(torch.int).min


    plt.figure()
    plt.imshow(vis_map_indices.reshape((height, width)), cmap='gray')
    plt.title("Token Map Indices")
    plt.colorbar()
    plt.savefig("token_map_indices.png")
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