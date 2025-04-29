from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple
import torch
import math
import pathlib

AXIS_ALIGNED_KEY = "axis_aligned"
NONAXIS_ALIGNED_KEY = "nonaxis_aligned"
TOKEN_MAP_KEY_TYPE = str | int
INVALID_TOKEN = -100   


class TokenType(Enum):
    IMAGE = auto()
    LEARNED = auto()
    CONDITION = auto()
    EMPTY = auto()
    
@dataclass(frozen=True)
class Token(ABC):

    @abstractmethod
    def token_type(self) -> TokenType:
        pass
    
@dataclass(frozen=True)
class EmptyToken(Token):
    def token_type(self) -> TokenType:
        return TokenType.EMPTY

@dataclass(frozen=True)
class SpatialToken(Token):
    x_coord: int 
    y_coord: int

    def image_index(self, image_width: int) -> int:
        return self.y_coord * image_width + self.x_coord
    
@dataclass(frozen=True)
class LearnedToken(SpatialToken):
    def token_type(self):
        return TokenType.LEARNED

@dataclass(frozen=True)
class ImageToken(SpatialToken):
    def token_type(self):
        return TokenType.IMAGE

@dataclass(frozen=True)
class ConditionToken(Token):
    cond_index: int

    def token_type(self):
        return TokenType.CONDITION

TokenMap = OrderedDict[Token, Token]

@dataclass(frozen=True)
class AutoRegressiveStructure:
    token_map: TokenMap
    attention_mask: torch.Tensor
    decoding_groups: list[list[int]] | None = None
    
    def __post_init__(self):
        assert self.attention_mask.shape[0] == self.attention_mask.shape[1] == len(self.token_map)
        

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


class TokenMapTensors(NamedTuple):
    axis_token_indices: torch.Tensor
    non_axis_token_indices: torch.Tensor

    image_token_indices: torch.Tensor
    learned_mask: torch.Tensor


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


def get_adam_attention_and_token_map(
    width: int, height: int, base_block_size: int, cond_len: int
) -> tuple[torch.Tensor, TokenMapTensors]:
    adam_masks, masked_coords, shift_patterns = _generalized_adam_interlacing(
        width, height, base_block_size
    )
    token_map, input_token_groups = _get_image_token_index_map(
        width, height, masked_coords, shift_patterns
    )
    attention_mask = _get_adam_attention_mask(
        adam_masks[0], cond_len, token_map, input_token_groups
    )
    return attention_mask, _token_map_to_tensors(token_map)


def _token_map_to_tensors(token_map: OrderedDict[TOKEN_MAP_KEY_TYPE, list[int]]):
    len_token_map = len(token_map)
    axis_token_indices = torch.full((len_token_map,), INVALID_TOKEN, dtype=torch.int)
    non_axis_token_indices = torch.full(
        (len_token_map,), INVALID_TOKEN, dtype=torch.int
    )

    image_token_indices = torch.full((len_token_map,), INVALID_TOKEN, dtype=torch.int)
    learned_mask = torch.zeros(len_token_map, dtype=torch.bool)

    for idx, (key, value) in enumerate(token_map.items()):
        match key:
            case int():
                image_token_indices[idx] = key
            case str():
                learned_mask[idx] = True
            case _:
                pass

        axis_token_indices[idx] = value[0]
        if len(value) == 2:
            non_axis_token_indices[idx] = value[1]

    return TokenMapTensors(
        axis_token_indices, non_axis_token_indices, image_token_indices, learned_mask
    )


def _generalized_adam_interlacing(width: int, height: int, base_block_size: int):
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


def _get_adam_attention_mask(
    first_adam_mask: torch.Tensor, cond_len: int, index_map, input_token_groups
) -> torch.Tensor:
    num_first_pass_tokens = first_adam_mask.int().sum()
    num_input_token_wo_cond = len(index_map)  # without condition
    total_len = num_input_token_wo_cond + cond_len

    attention_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))

    attention_mask_wo_cond = torch.tril(
        torch.ones(num_input_token_wo_cond, num_input_token_wo_cond, dtype=torch.bool)
    )
    attention_mask_wo_cond[num_first_pass_tokens - 1 :, num_first_pass_tokens - 1 :] = 0
    previous_indices = []
    for input_token_group in input_token_groups:
        previous_indices.extend(input_token_group)
        curr_tensor = torch.tensor(input_token_group)
        prev_tensor = torch.tensor(previous_indices)
        rows, cols = torch.meshgrid(curr_tensor, prev_tensor, indexing="ij")
        attention_mask_wo_cond[rows, cols] = 1

    attention_mask[cond_len:, cond_len:] = attention_mask_wo_cond
    return attention_mask


def _get_image_token_index_map(
    width: int,
    height: int,
    masked_coords: list[torch.Tensor],
    shift_patterns: list[dict[str, ShiftPattern]],
    learned_token: str = "l",
) -> tuple[OrderedDict[TOKEN_MAP_KEY_TYPE, list[int]], list[list[int]]]:
    map: OrderedDict[TOKEN_MAP_KEY_TYPE, list[int]] = OrderedDict()
    input_token_groups = []

    # first_pass_index_shift = cond_len if include_cond else 0
    # autoregressive first pass
    first_pass_indices = masked_coords[0].tolist()
    num_trans_tokens = len(first_pass_indices) - 1
    for ind in range(len(first_pass_indices) - 1):
        prev_ind_x, pred_ind_y = first_pass_indices[ind]
        next_ind_x, next_ind_y = first_pass_indices[ind + 1]
        pred_ind = pred_ind_y * width + prev_ind_x
        next_ind = next_ind_y * width + next_ind_x
        map[pred_ind] = [next_ind]

    last_ind_x, last_ind_y = first_pass_indices[-1]
    last_ind = last_ind_y * width + last_ind_x

    for i in range(1, len(masked_coords) - 1):
        # second pass with learnable tokens
        indices = masked_coords[i].tolist()
        # next_indices = masked_coords[i+1].tolist()
        if i == 1:
            curr_first_ind_x, curr_first_ind_y = indices[0]
            curr_first_ind = curr_first_ind_y * width + curr_first_ind_x

            input_token_group = []
            map[last_ind] = [curr_first_ind]
            input_token_group.append(len(map) - 1)
            for j in range(num_trans_tokens):
                pred_ind_x, pred_ind_y = indices[j + 1]
                pred_ind = pred_ind_y * width + pred_ind_x
                map[learned_token + str(j)] = [pred_ind]
                input_token_group.append(len(map) - 1)

            input_token_groups.append(input_token_group)

        input_token_group = []
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
            input_token_group.append(len(map) - 1)

        input_token_groups.append(input_token_group)

    return map, input_token_groups


def _get_image_token_index_map_v2(
    width: int,
    height: int,
    cond_len: int,
    masked_coords: list[torch.Tensor],
    ) -> AutoRegressiveStructure:

    total_len = width * height + cond_len
    attention_mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool))

    token_map: TokenMap = TokenMap()    
    first_pass_coords = masked_coords[0].tolist()
    first_image_token = ImageToken(x_coord=first_pass_coords[0][0], y_coord=first_pass_coords[0][1])
    attention_mask[cond_len+len(first_pass_coords):, cond_len+len(first_pass_coords):] = 0
    
    bi_attention_size = total_len - cond_len - len(first_pass_coords)
    num_generated_tokens_per_pass = [len(coords) for coords in masked_coords[1:]]
    assert sum(num_generated_tokens_per_pass) == bi_attention_size
    bi_attention_mask = torch.zeros((bi_attention_size , bi_attention_size), dtype=torch.bool)
    
    num_prev_tokens = 0
    for num in num_generated_tokens_per_pass:
        num_prev_tokens += num
        bi_attention_mask[num_prev_tokens-num:num_prev_tokens, :num_prev_tokens] = 1
    
    #prepare attention mask
    attention_mask[cond_len+len(first_pass_coords):, cond_len+len(first_pass_coords):] = bi_attention_mask

    # add condition tokens
    for i in range(cond_len):
        cond_token = ConditionToken(cond_index=i)
        if i == cond_len - 1:
            token_map[cond_token] = first_image_token
        else:
            token_map[cond_token] =  EmptyToken()

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
        if i_pass == 1:
            # learnable token
            for idx, coord in enumerate(curr_coords):
                x, y = coord
                learnable_token = LearnedToken(x_coord=x, y_coord=y)
                pred_img_token = ImageToken(x_coord=x, y_coord=y)
                token_map[learnable_token] = pred_img_token
                # input_token_group.append(curr_img_token)
        else:
            previous_coords = masked_coords[i_pass - 1].tolist()
            prev_coord_len = len(previous_coords)
            assert prev_coord_len == len(curr_coords) // 2
            for prev_coord, curr_coord in zip(previous_coords, curr_coords[:prev_coord_len]):
                prev_x, prev_y = prev_coord
                curr_x, curr_y = curr_coord
                prev_img_token = ImageToken(x_coord=prev_x, y_coord=prev_y)
                curr_img_token = ImageToken(x_coord=curr_x, y_coord=curr_y)
                
                token_map[prev_img_token] = curr_img_token
        
            for curr_coord in curr_coords[prev_coord_len:]:
                x, y = curr_coord
                learnable_token = LearnedToken(x_coord=x, y_coord=y)
                token_map[learnable_token] = ImageToken(x_coord=x, y_coord=y)
    
    return AutoRegressiveStructure(
        token_map=token_map,
        attention_mask=attention_mask,
    )
    

def _test_attention_mask(attention_mask, input_token_groups, first_adam_mask, cond_len):
    num_first_pass_tokens = first_adam_mask.int().sum()
    for input_ind in range(attention_mask.shape[0]):
        if input_ind < cond_len + num_first_pass_tokens - 1:
            assert attention_mask[input_ind][input_ind]
            assert not attention_mask[input_ind][input_ind + 1]
        else:
            assert attention_mask[input_ind].int().sum()

    expansion_steps = [1 for _ in range(cond_len + num_first_pass_tokens - 2)] + [
        len(g) for g in input_token_groups
    ]
    step_index = 0
    # check if the attention mask expands from left to right
    prev = attention_mask[0]
    prev_ones = sum(prev)
    for curr in attention_mask[1:]:
        for p, c in zip(prev, curr):
            if p == 1 and c == 0:
                assert False

        curr_ones = sum(curr)
        step = curr_ones - prev_ones
        if step == 0:
            prev = curr
            continue

        if step_index >= len(expansion_steps) or step != expansion_steps[step_index]:
            assert False

        # Move to next expected step
        step_index += 1
        prev = curr
        prev_ones = curr_ones


def _test_index_map(
    index_map: OrderedDict[TOKEN_MAP_KEY_TYPE, list[int]],
    adam_masks: list[torch.Tensor],
    height: int,
    width: int,
):
    # check expected num input tokens
    num_input_tokens = 0
    num_masks = len(adam_masks)
    for i in range(num_masks - 1):
        num_gen_tokens = adam_masks[i].int().sum()
        if i == 0:
            num_input_tokens += 2 * num_gen_tokens - 1
        else:
            num_input_tokens += num_gen_tokens
    assert len(index_map) == num_input_tokens

    # check expeceted num output tokens
    output_token_indices = [
        item
        for sublist in list(index_map.values())
        if sublist is not None
        for item in sublist
    ]
    assert len(output_token_indices) == height * width - 1
    assert len(set(output_token_indices)) == len(output_token_indices)

    image = torch.zeros((height, width), dtype=torch.int)
    gt_image = torch.ones_like(image)
    gt_image[0, 0] = 0
    for i in output_token_indices:
        x = (i) % width
        y = (i) // width
        image[y, x] = 1
    assert (image == gt_image).all()


def _test_input_token_groups(
    input_token_groups, width: int, height: int, base_block_size: int
):
    flattened = []
    num_base_blocks = (width * height) // (base_block_size**2)
    for idx, input_token_group in enumerate(input_token_groups):
        flattened.extend(input_token_group)
        if idx == 0 or idx == 1:
            assert len(input_token_group) == num_base_blocks
        else:
            assert len(input_token_group) == num_base_blocks * (2 ** (idx - 1))

    assert all(flattened[i] + 1 == flattened[i + 1] for i in range(len(flattened) - 1))
    assert flattened[0] == num_base_blocks - 1


def test_adam_utils_consistency():
    width = 32
    height = 32
    base_block_size = 16
    cond_len = 1

    adam_masks, masked_coords, shift_patterns = _generalized_adam_interlacing(
        width, height, base_block_size
    )
    index_map, input_token_groups = _get_image_token_index_map(
        width, height, masked_coords, shift_patterns
    )
    _get_image_token_index_map_v2(
        width,height,cond_len, masked_coords,
    )
    attention_mask = _get_adam_attention_mask(
        adam_masks[0], cond_len, index_map, input_token_groups
    )

    _test_index_map(index_map, adam_masks, height, width)
    _test_input_token_groups(input_token_groups, width, height, base_block_size)
    _test_attention_mask(attention_mask, input_token_groups, adam_masks[0], cond_len)
    visualize_adam_masks(adam_masks, f"adam_mask_block_size_{base_block_size}.png")


def visualize_adam_masks(masks: list[torch.Tensor], filename: str | pathlib.Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(masks), figsize=(3 * len(masks), 3))
    for i, mask in enumerate(masks):
        axes[i].imshow(mask.cpu(), cmap="gray", interpolation="none")
        axes[i].set_title(f"Pass {i + 1}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    test_adam_utils_consistency()