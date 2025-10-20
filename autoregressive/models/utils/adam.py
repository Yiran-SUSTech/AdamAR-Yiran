import torch
import math
from dataclasses import dataclass
from autoregressive.models.utils.tokens import *
from autoregressive.models.utils.autoregr import AutoRegressiveStructure
import torch.distributed as dist


AXIS_ALIGNED_KEY = "axis_aligned"
NONAXIS_ALIGNED_KEY = "nonaxis_aligned"

Pre_TOKEN_FUNCTION = {
    'close_min': find_close_min_token,
    'close_max': find_close_max_token,
    'close_unattach_min': find_close_unattach_min_token,
    'close_unattach_max': find_close_unattach_max_token,
    'close_left_up': find_close_left_up_token,
    'close_center': find_close_center_token,
}

@dataclass
class ShiftPattern:
    x_shift: int
    y_shift: int
    axis_aligned: bool = True

    def __post_init__(self):
        if self.axis_aligned:
            assert self.x_shift == 0 or self.y_shift == 0, (
                "Axis-aligned shifts must have one of the shifts as zero"
            )
        else:
            assert self.x_shift != 0 and self.y_shift != 0, (
                "Non-axis-aligned shifts must have both shifts non-zero"
            )
            
def get_adam_pattern(
    base_block_size: int,
) -> tuple[list[tuple[int, int, int, int]], list[dict[str, ShiftPattern]]]:
    assert (base_block_size & (base_block_size - 1)) == 0, (
        "base_block_size must be a power of 2"
    )
    levels = int(math.log2(base_block_size))
    num_passes = 2 * levels + 1

    patterns = []
    shift_patterns: list[dict[str, ShiftPattern]] = []
    for i in range(num_passes):
        if i == 0:
            patterns.append((0, 0, base_block_size, base_block_size))
            shift_patterns.append(
                {AXIS_ALIGNED_KEY: ShiftPattern(base_block_size // 2, 0)}
            )
            continue
        else:
            if i % 2 == 0:
                x_start = 0
                y_start = int(base_block_size / (2 ** (i // 2)))
                x_stride = y_start
                y_stride = x_stride * 2
                if i != num_passes - 1:
                    shift_patterns.append(
                        {
                            AXIS_ALIGNED_KEY: ShiftPattern(x_stride // 2, 0),
                            NONAXIS_ALIGNED_KEY: ShiftPattern(
                                x_stride // 2, y_stride // 2, axis_aligned=False
                            ),
                        }
                    )

            else:
                x_start = int(base_block_size / (2 ** ((i + 1) // 2)))
                y_start = 0
                x_stride = x_start * 2
                y_stride = x_stride
                shift_patterns.append(
                    {
                        AXIS_ALIGNED_KEY: ShiftPattern(
                            x_shift=0, y_shift=y_stride // 2
                        ),
                        NONAXIS_ALIGNED_KEY: ShiftPattern(
                            x_shift=x_stride // 2,
                            y_shift=y_stride // 2,
                            axis_aligned=False,
                        ),
                    }
                )

            patterns.append((x_start, y_start, x_stride, y_stride))

    return patterns, shift_patterns

def get_quadrants(col:int, row:int, image_width: int, image_height: int) -> int:
    if col < image_width / 2 and row < image_height / 2:
        return 0  # Top-left
    elif col >= image_width / 2 and row < image_height / 2:
        return 1  # Top-right
    elif col < image_width / 2 and row >= image_height / 2:
        return 2  # Bottom-left
    else:
        return 3  # Bottom-right

def generalized_adam_interlacing(width: int, height: int, base_block_size: int, interlacing_type: str = "adam"):
    assert width >= base_block_size
    assert height >= base_block_size
    assert width % base_block_size == 0 and height % base_block_size == 0, (
        "width and height must be divisible by block size in this implementation"
    )

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
                    if interlacing_type == "adam" or "corner_adam":
                        coords.append((x, y))
                    elif interlacing_type == "spin_adam":
                        quadrant = get_quadrants(x, y, width, height)
                        if quadrant == 0: # Top-left
                            x_new = int(width / 2 - 1 - y)
                            y_new = int(height / 2 - 1 - x)
                        elif quadrant == 1: # Top-right
                            x_new = x
                            y_new = int(height / 2 - 1 - y)
                        elif quadrant == 2: # Bottom-left
                            x_new = int(width / 2 - 1 - x)
                            y_new = y
                        elif quadrant == 3: # Bottom-right
                            x_new = y
                            y_new = x
                        else:
                            raise ValueError(f"Invalid quadrant: {quadrant}")
                        coords.append((x_new, y_new))

        coords = torch.tensor(coords, dtype=torch.int)
        adam_masks.append(mask)
        adam_coords.append(coords)

    return adam_masks, adam_coords, shift_patterns

def convert_coner_adam_interlacing(width: int, height: int, adam_coords: list[torch.Tensor]) -> list[torch.Tensor]:
    filled = torch.zeros(height, width, dtype=torch.bool)
    first_pass_coords = adam_coords[0]

    new_adam_coords: list[torch.Tensor] = []
    converted_coords = []
    for first_pass_coord in first_pass_coords.tolist():
        x, y = first_pass_coord
        filled[y,x] = True
        if dist.get_rank() == 0:
            print(f"First pass coord: ({x}, {y})")
        converted_coords.append((x,y))
    new_adam_coords.append(torch.tensor(converted_coords, dtype=torch.int))
    
    pass_idx = 1
    while torch.sum(filled) < width * height:
        if dist.get_rank() == 0:
            print(f"torch.sum(filled): {torch.sum(filled)}, width*height: {width*height}")
        tmp_coords = []
        expand_dir = pass_idx % 2 # 1: to right, 0: to down
        expand_len = 2 ** ((pass_idx+1) // 2 - 1)
        if dist.get_rank() == 0:
            print(f"Pass {pass_idx}, len(converted_coords): {len(converted_coords)}, expand_dir: {expand_dir}, expand_len: {expand_len}")
        num_coords_current_pass = len(converted_coords)
        for coord_idx in range(num_coords_current_pass):
            x, y = converted_coords[coord_idx]
            if expand_dir == 1: # to right
                xnew = x + expand_len
                ynew = y
            else: # to down
                xnew = x
                ynew = y + expand_len
            if filled[ynew, xnew]:
                if dist.get_rank() == 0:
                    print(f"Current coord: ({x}, {y}), New coord: ({xnew}, {ynew})")

            filled[ynew, xnew] = True
            tmp_coords.append((xnew, ynew))
            converted_coords.append((xnew, ynew))
        new_adam_coords.append(torch.tensor(tmp_coords, dtype=torch.int))
        pass_idx += 1
        if dist.get_rank() == 0:
            print(f"pass_coords: ({tmp_coords})")

    return new_adam_coords





def autoregressive_first_step(adam_coords: list[torch.Tensor]) -> list[torch.Tensor]:
    ar_first_step = adam_coords[0].split(1, dim=0)
    new_adam_coords = list(ar_first_step) + adam_coords[1:]
    
    return new_adam_coords

# set subpass by length of each subpass
def set_subpass_by_len(adam_coords: list[torch.Tensor], subpass_len: int=1) -> list[torch.Tensor]:
    new_adam_coords = []
    assert subpass_len > 0, "subpass_len must be greater than 0"
    for pass_idx, adam_coord in enumerate(adam_coords):
        if pass_idx == 0: # do not mess up with the first pass
            new_adam_coords.append(adam_coord)
            continue
        # if subpass length is 0, then, it is serial generation within each pass
        assert len(adam_coord) % subpass_len == 0, "Each pass length must be divisible by subpass_len"
        tmp = adam_coord.reshape(-1, subpass_len, 2)
        tmp = tmp.transpose(0,1).reshape(-1, 2)
        autoregressive_n_step = tmp.split(len(adam_coord) // subpass_len, dim=0)
        new_adam_coords += list(autoregressive_n_step)
    
    return new_adam_coords

# set subpass by number of subpasses within each pass
def set_subpass_by_num(adam_coords: list[torch.Tensor], subpass_num: int=4) -> list[torch.Tensor]:
    new_adam_coords = []
    assert subpass_num >= 1, "subpass_num must be equal or greater than 1"
    for pass_idx, adam_coord in enumerate(adam_coords):
        if pass_idx == 0: # do not mess up with the first pass
            new_adam_coords.append(adam_coord)
            continue
        subpass_len = math.ceil(len(adam_coord) / subpass_num)
        autoregressive_n_step = adam_coord.split(subpass_len, dim=0)
        new_adam_coords += list(autoregressive_n_step)
    
    return new_adam_coords

def get_autoregressive_structure(
    logger, 
    width: int,
    height: int,
    base_block_size: int,
    cond_len: int,
    subpass_len: int=None, # subpass_len==1 means serial generation within each pass
    subpass_num: int=None,
    pre_token_choose: str="close_min",
    freqs_cis_reorder_shceme: str = None,
    interlacing_type: str = "adam",
) -> AutoRegressiveStructure:
    _, masked_coords, _ = generalized_adam_interlacing(width, height, base_block_size, interlacing_type)

    if interlacing_type == "corner_adam":
        masked_coords = convert_coner_adam_interlacing(width, height, masked_coords)
    total_len = width * height + cond_len

    #########################################
    assert (subpass_len is None) or (subpass_num is None), "Only one of subpass_len and subpass_num can be set"
    if subpass_len is not None:
        masked_coords = set_subpass_by_len(masked_coords, subpass_len)
    if subpass_num is not None:
        masked_coords = set_subpass_by_num(masked_coords, subpass_num)
    #########################################

    if dist.get_rank() == 0:
        num_output_image_tokens = 0
        for i_pass, coords_i_pass in enumerate(masked_coords):
            print(f"pass {i_pass}:")
            coords_n_indics = []
            for coord in coords_i_pass.tolist():
                x, y = coord
                coords_n_indics.append([x,y])
            print(coords_n_indics)
            num_output_image_tokens += len(coords_i_pass.tolist())
        print(f"num_output_image_tokens: {num_output_image_tokens}")

    token_map: TokenMap = TokenMap()
    first_pass_coords = masked_coords[0].tolist()
    first_image_token = ImageToken(
        x_coord=first_pass_coords[0][0], y_coord=first_pass_coords[0][1]
    )

    # bi_attention_size, num_generated_tokens_per_pass seem not important 
    ################################
    bi_attention_size = total_len - cond_len - len(first_pass_coords)
    num_generated_tokens_per_pass = [len(coords) for coords in masked_coords[1:]]
    assert sum(num_generated_tokens_per_pass) == bi_attention_size
    bi_attention_mask = torch.zeros(
        (bi_attention_size, bi_attention_size), dtype=torch.bool
    )

    num_prev_tokens = 0
    for num in num_generated_tokens_per_pass:
        num_prev_tokens += num
        bi_attention_mask[num_prev_tokens - num : num_prev_tokens, :num_prev_tokens] = 1
    ################################

    # add condition tokens
    # when GPT gets condition token, it should predict the first image token
    for i in range(cond_len):
        cond_token = ConditionToken(cond_index=i)
        if i == cond_len - 1:
            token_map[cond_token] = first_image_token
        else:
            token_map[cond_token] = EmptyToken()
    
    # autoregressive first pass
    for idx in range(1, len(first_pass_coords)):
        prev_coords = first_pass_coords[idx - 1]
        curr_coords = first_pass_coords[idx]

        prev_x, prev_y = prev_coords
        curr_x, curr_y = curr_coords

        prev_img_token = ImageToken(x_coord=prev_x, y_coord=prev_y)
        curr_img_token = ImageToken(x_coord=curr_x, y_coord=curr_y)
        token_map[prev_img_token] = curr_img_token


    # passes with learnable tokens
    for i_pass in range(1, len(masked_coords)):
        curr_coords = masked_coords[i_pass].tolist()

        generated_tokens = list(t for t in token_map.output_tokens() if t.token_type() == TokenType.IMAGE)
        for curr_coord in curr_coords:
            x, y = curr_coord
            curr_img_token = ImageToken(x_coord=x, y_coord=y)

            closest_token = Pre_TOKEN_FUNCTION[pre_token_choose](
                token_map, curr_img_token, generated_tokens, width
            )
            if dist.get_rank() == 0:
                print(f"closest previous token: ({closest_token.x_coord}, {closest_token.y_coord}), Current token: ({curr_img_token.x_coord}, {curr_img_token.y_coord})")
            token_map[closest_token] = curr_img_token  # 所以好几个token的前序token可能是相同的，这个相同的token在token_map中的_input_index会是一个列表，记录其被作为前序token的所有时刻

    decoded_masked_coords = autoregressive_first_step(masked_coords)
    # print("decoded_masked_coords:", decoded_masked_coords) ##############################################
    
    return AutoRegressiveStructure(
        logger=logger, ##############################################
        image_height=height,
        image_width=width,
        token_map=token_map,
        decoded_masked_coords=decoded_masked_coords,
        freqs_cis_reorder_shceme=freqs_cis_reorder_shceme,
    )
