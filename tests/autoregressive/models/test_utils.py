import pathlib
from collections import OrderedDict

import torch

from autoregressive.models.gpt import precompute_freqs_cis_2d
from autoregressive.models.utils import (
    TOKEN_MAP_KEY_TYPE,
    _generalized_adam_interlacing,
    _get_adam_attention_mask,
    _get_image_token_index_map,
    _get_image_token_index_map_v2,
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
    ar_structure = _get_image_token_index_map_v2(
        width,
        height,
        cond_len,
        masked_coords,
    )

    bs = 16
    dim = 128
    rope_base = 10000
    per_head_dim = 32
    image_tokens = torch.randn(bs, width * height, dim)
    cond_tokens = torch.randn(bs, cond_len, dim)
    learnable_token = torch.randn(dim)

    image_token_idx = torch.arange(width * height).reshape(-1)
    image_token_idx = torch.stack(
        [image_token_idx[torch.randperm(image_token_idx.shape[0])] for _ in range(bs)]
    )

    freqs_cis = precompute_freqs_cis_2d(width, per_head_dim, rope_base, cond_len)
    input_tokens = ar_structure.assemble_input_tokens(
        image_tokens, cond_tokens, learnable_token, freqs_cis
    )
    target_token_idx, target_mask = ar_structure.assemble_target_tokens(image_token_idx)

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
