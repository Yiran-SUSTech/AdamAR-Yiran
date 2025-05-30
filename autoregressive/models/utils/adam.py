import torch
import math
from dataclasses import dataclass
from autoregressive.models.utils.tokens import *
from autoregressive.models.utils.autoregr import AutoRegressiveStructure

AXIS_ALIGNED_KEY = "axis_aligned"
NONAXIS_ALIGNED_KEY = "nonaxis_aligned"

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

    assert len(shift_patterns) == num_passes - 1, (
        "shift_patterns must have length num_passes - 1"
    )
    return patterns, shift_patterns


def generalized_adam_interlacing(width: int, height: int, base_block_size: int):
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
                    coords.append((x, y))

        coords = torch.tensor(coords, dtype=torch.int)
        adam_masks.append(mask)
        adam_coords.append(coords)

    return adam_masks, adam_coords, shift_patterns


def autoregressive_first_step(adam_coords: list[torch.Tensor]) -> list[torch.Tensor]:
    autoregressive_first_step = adam_coords[0].split(1, dim=0)
    new_adam_coords = list(autoregressive_first_step) + adam_coords[1:]
    return new_adam_coords

def get_autoregressive_structure(
    width: int,
    height: int,
    base_block_size: int,
    cond_len: int,
) -> AutoRegressiveStructure:
    _, masked_coords, _ = generalized_adam_interlacing(width, height, base_block_size)
    total_len = width * height + cond_len

    token_map: TokenMap = TokenMap()
    first_pass_coords = masked_coords[0].tolist()
    first_image_token = ImageToken(
        x_coord=first_pass_coords[0][0], y_coord=first_pass_coords[0][1]
    )

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

    # add condition tokens
    for i in range(cond_len):
        cond_token = ConditionToken(cond_index=i)
        if i == cond_len - 1:
            token_map[cond_token] = first_image_token
        else:
            token_map[cond_token] = EmptyToken()

    # autoregressive first pass
    for idx in range(len(first_pass_coords) - 1):
        curr_coords = first_pass_coords[idx]
        next_coords = first_pass_coords[idx + 1]

        curr_x, curr_y = curr_coords
        next_x, next_y = next_coords

        curr_img_token = ImageToken(x_coord=curr_x, y_coord=curr_y)
        next_img_token = ImageToken(x_coord=next_x, y_coord=next_y)
        token_map[curr_img_token] = next_img_token

        if idx == len(first_pass_coords) - 2:
            last_img_token = next_img_token
            token_map[last_img_token] = EmptyToken()

    # passes with learnable tokens
    for i_pass in range(1, len(masked_coords)):
        curr_coords = masked_coords[i_pass].tolist()

        generated_tokens = list(t for t in token_map.output_tokens() if t.token_type() == TokenType.IMAGE)
        for curr_coord in curr_coords:
            x, y = curr_coord
            curr_img_token = ImageToken(x_coord=x, y_coord=y)
            closest_token = find_closest_token(
                curr_img_token, generated_tokens, width
            )
            token_map[closest_token] = curr_img_token
            
    decoded_masked_coords = autoregressive_first_step(masked_coords)
    
    return AutoRegressiveStructure(
        image_height=height,
        image_width=width,
        token_map=token_map,
        decoded_masked_coords=decoded_masked_coords,
    )
