from collections import OrderedDict
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from sympy import Order, flatten
import torch
import math
import pathlib
import dataclasses

AXIS_ALIGNED_KEY = "axis_aligned"
NONAXIS_ALIGNED_KEY = "nonaxis_aligned"
    
@dataclasses.dataclass
class ShiftPattern:
    x_shift: int
    y_shift: int
    axis_aligned: bool=True
    
    def __post_init__(self):
        if self.axis_aligned:
            assert self.x_shift == 0 or self.y_shift == 0, "Axis-aligned shifts must have one of the shifts as zero"
        else:
            assert self.x_shift != 0 and self.y_shift != 0, "Non-axis-aligned shifts must have both shifts non-zero"

def get_adam_pattern(base_block_size: int) -> tuple[list[tuple[int, int, int, int]], list[dict[str, ShiftPattern]]]:
    assert (base_block_size & (base_block_size - 1)) == 0, "base_block_size must be a power of 2"
    levels = int(math.log2(base_block_size))
    num_passes = 2 * levels + 1
        
    patterns = []
    shift_patterns: list[dict[str, ShiftPattern]] = []
    for i in range(num_passes):
        if i == 0:
            patterns.append((0, 0, base_block_size, base_block_size))
            shift_patterns.append({AXIS_ALIGNED_KEY: ShiftPattern(base_block_size // 2, 0)})
            continue
        else:
            if i % 2 == 0:
                x_start = 0 
                y_start = int(base_block_size / (2**(i // 2)))
                x_stride = y_start
                y_stride = x_stride * 2
                if i != num_passes - 1:
                    shift_patterns.append({AXIS_ALIGNED_KEY: ShiftPattern(x_stride // 2, 0),
                                           NONAXIS_ALIGNED_KEY: ShiftPattern(x_stride // 2, y_stride // 2, axis_aligned=False)})
                    
            else:
                x_start = int(base_block_size / (2**((i+1) // 2)))
                y_start = 0
                x_stride = x_start * 2
                y_stride = x_stride
                shift_patterns.append({AXIS_ALIGNED_KEY: ShiftPattern(x_shift=0, y_shift=y_stride // 2), 
                                       NONAXIS_ALIGNED_KEY: ShiftPattern(x_shift=x_stride // 2, y_shift=y_stride // 2, axis_aligned=False)})

            patterns.append((x_start, y_start, x_stride, y_stride))

    assert len(shift_patterns) == num_passes - 1, "shift_patterns must have length num_passes - 1"
    return patterns, shift_patterns

def _generalized_adam_interlacing(width: int, height: int, base_block_size: int):

    assert width  >= base_block_size
    assert height >= base_block_size

    patterns, shift_patterns = get_adam_pattern(base_block_size)
    adam_masks: list[torch.Tensor] = []
    adam_coords: list[torch.Tensor] = []
    
    filled = torch.zeros(height, width, dtype=torch.bool)
    for x_start, y_start, x_step, y_step in patterns:
        mask = torch.zeros(height, width, dtype=torch.bool)
        coords = []
        for y in range(y_start, height, y_step):
            for x in range(x_start, width, x_step):
                if not filled[y, x]:
                    mask[y, x] = True
                    filled[y, x] = True
                    coords.append((x, y))
            
        coords = torch.tensor(coords, dtype=torch.int)
        adam_masks.append(mask)
        adam_coords.append(coords)

    return adam_masks, adam_coords, shift_patterns

def _get_attention_mask_from_adam(cond_len:int, adam_masks: list[torch.Tensor]) -> torch.Tensor:
    assert len(adam_masks) > 0, "adam_masks must not be empty"
    height, width = adam_masks[0].shape
    image_token_len = height * width
    total_len = cond_len + image_token_len
    attention_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))
    attention_mask[cond_len:, cond_len:] = 0

    img_attention_mask = torch.zeros((image_token_len, image_token_len), dtype=torch.bool)
    previous_indices = []
    for mask in adam_masks:
        mask = mask.view(-1)
        curr_indices = mask.nonzero()[:,0].tolist()
        previous_indices.extend(curr_indices)
        curr_tensor = torch.tensor(curr_indices)
        prev_tensor = torch.tensor(previous_indices)
        rows, cols = torch.meshgrid(curr_tensor, prev_tensor, indexing='ij')
        img_attention_mask[rows, cols] = 1

    assert len(previous_indices) == image_token_len, "adam_masks must exactly cover all image tokens"
    attention_mask[cond_len:, cond_len:] = img_attention_mask
    return attention_mask


def get_adam_attention_mask(width: int, height: int, base_block_size: int, cond_len: int) -> torch.Tensor:
    masks, _, _ = _generalized_adam_interlacing(width, height, base_block_size)
    attention_mask = _get_attention_mask_from_adam(cond_len, masks)
    return attention_mask


def get_output_pred_index_map(width: int, height: int, base_block_size: int, cond_len: int, include_cond:bool=False, learned_token: str='l') -> OrderedDict[int, list[int]]:
    # which input token indices are used to predict which output token indices
    _, masked_coords, shift_patterns = _generalized_adam_interlacing(width, height, base_block_size)
    map = OrderedDict()
    # autoregressive first pass  
    first_pass_indices = masked_coords[0].tolist()
    num_trans_tokens = len(first_pass_indices) - 1
    for ind in range(len(first_pass_indices)-1):
        prev_ind_x, pred_ind_y = first_pass_indices[ind]
        next_ind_x, next_ind_y = first_pass_indices[ind+1]
        pred_ind = pred_ind_y * width + prev_ind_x
        next_ind = next_ind_y * width + next_ind_x
        map[pred_ind] = [next_ind]

    last_ind_x, last_ind_y = first_pass_indices[-1]
    last_ind = last_ind_y * width + last_ind_x
    
    for i in range(1, len(masked_coords)-1):
        # second pass with learnable tokens
        indices = masked_coords[i].tolist()    
        # next_indices = masked_coords[i+1].tolist()
        if i == 1:
            curr_first_ind_x, curr_first_ind_y = indices[0]
            curr_first_ind =  curr_first_ind_y * width + curr_first_ind_x
            map[last_ind] = [curr_first_ind]               
            for j in range(num_trans_tokens):
                pred_ind_x, pred_ind_y = indices[j + 1]
                pred_ind = pred_ind_y * width + pred_ind_x
                map[learned_token + str(j)] = [pred_ind]

        for indice in indices:
            ind_x, ind_y = indice
            curr_ind = ind_y * width + ind_x
            shift_pattern = shift_patterns[i]
            
            mapped_to = []
            tmp = shift_pattern[AXIS_ALIGNED_KEY]
            x_shift = tmp.x_shift
            y_shift = tmp.y_shift
            next_ind_x = (ind_x + x_shift) % width
            next_ind_y = (ind_y + y_shift) % height
            next_ind = next_ind_y * width + next_ind_x
            mapped_to.append(next_ind)
                
            if len(shift_pattern) == 2:
                tmp = shift_pattern[NONAXIS_ALIGNED_KEY]
                x_shift = tmp.x_shift
                y_shift = tmp.y_shift
                next_ind_x = (ind_x + x_shift) % width
                next_ind_y = (ind_y + y_shift) % height
                next_ind = next_ind_y * width + next_ind_x
                mapped_to.append(next_ind)
            
            map[curr_ind] = mapped_to

    if include_cond:
        map_with_cond = OrderedDict()
        for cond_ind in range(cond_len):
            map_with_cond[cond_ind] = None
            if cond_ind == cond_len-1:
                map_with_cond[cond_ind] = [cond_len]
        
        for in_token_ind, output_token_indices in map.items():
            match in_token_ind:
                case int():
                    map_with_cond[in_token_ind + cond_len] = [i + cond_len for i in output_token_indices]
                case str():
                    map_with_cond[in_token_ind] = [i + cond_len for i in output_token_indices]
                case _:
                    raise ValueError(f"Unexpected token index type: {type(in_token_ind)}")
        return map_with_cond
    
    return map  

def test_attention_mask(cond_len: int, attention_mask: torch.Tensor, adam_masks: list[torch.Tensor]):
    assert (attention_mask[:cond_len, :cond_len] == torch.tril(torch.ones(cond_len, cond_len, dtype=torch.bool))).all()
    assert attention_mask[cond_len:, :cond_len].all()
    img_attention_mask = attention_mask[cond_len:, cond_len:]
    prev_indices = []
    for mask in adam_masks:
        curr_indices = mask.view(-1).nonzero()[:, 0].tolist()
        prev_indices.extend(curr_indices)
        assert img_attention_mask[torch.tensor(curr_indices)[:, None], torch.tensor(prev_indices)[None, :]].all()


def test_index_map():
    width = 16
    height = 16
    index_map = get_output_pred_index_map(16,16,8,2, include_cond=True)
    output_token_indices = [item for sublist in list(index_map.values()) if sublist is not None for item in sublist]
    assert len(output_token_indices) == len(set(output_token_indices))
    assert len(output_token_indices) == 16 * 16


    index_map = get_output_pred_index_map(16,16,8,2, include_cond=False)
    output_token_indices = [item for sublist in list(index_map.values()) if sublist is not None for item in sublist]
    assert len(output_token_indices) == 16 * 16 - 1
    image = torch.zeros((height, width), dtype=torch.int)
    gt_image = torch.ones_like(image)
    gt_image[0,0] = 0 
    for i in output_token_indices:
        x = i % width
        y = i // width
        image[y, x] = 1
    
    assert (image == gt_image).all()
    

def visualize_adam_masks(masks: list[torch.Tensor], filename: str | pathlib.Path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(masks), figsize=(3 * len(masks), 3))
    for i, mask in enumerate(masks):
        axes[i].imshow(mask.cpu(), cmap='gray', interpolation='none')
        axes[i].set_title(f'Pass {i+1}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    # width, height = 16, 16
    # base_block_size = 8
    # cond_len = 2
    # masks, _, _ = _generalized_adam_interlacing(width, height, base_block_size)
    # attention_mask = _get_attention_mask_from_adam(cond_len, masks)
    # test_attention_mask(cond_len, attention_mask, masks)
    # get_output_pred_index_map(width, height, base_block_size, cond_len)
    test_index_map()