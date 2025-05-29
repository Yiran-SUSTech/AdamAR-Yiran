import pathlib
from collections import OrderedDict

import torch

from autoregressive.models.gpt import precompute_freqs_cis_2d
from autoregressive.models.utils.tokens import (
    TOKEN_MAP_KEY_TYPE,
)

from autoregressive.models.utils.adam import generalized_adam_interlacing, autoregressive_first_step, get_autoregressive_structure
from debug.vis_utils import visualize_attention_mask, visualize_token_map, visualize_adam_masks
from pathlib import Path

def _test_attention_mask(attention_mask, decoding_schedule):  
    expansion_steps = [len(decoded_group) for decoded_group in decoding_schedule]
    # check if the attention mask expands from left to right
    prev_ones = 0
    step_index = 0 
    for curr in attention_mask:
        curr_ones = sum(curr)
        step = curr_ones - prev_ones
        if step == 0:
            continue
        if step_index >= len(expansion_steps) or step != expansion_steps[step_index]:
            assert False
        step_index += 1
        prev_ones = curr_ones
class TestAutoregressiveStructure:

    def setup_method(self):
        self.width = 32
        self.height = 32
        self.base_block_size = 16
        self.cond_len = 1
        self.output_dir = Path('tests/test_outputs')  
        self.output_dir.mkdir(exist_ok=True)      
        
    def tests(self):
        ar_structure = get_autoregressive_structure(self.width, self.height, self.base_block_size, self.cond_len)
        adam_masks, masked_coords, _ = generalized_adam_interlacing(
            self.width, self.height, self.base_block_size
        )
        autoregres_first_masked_coords = autoregressive_first_step(masked_coords)
        decoding_schedule = ar_structure.get_decoding_schedule(autoregres_first_masked_coords)
        attention_mask = ar_structure.get_training_attention_mask(decoding_schedule)

        _test_attention_mask(attention_mask, decoding_schedule)

        visualize_token_map(ar_structure.token_map, width=self.width, height=self.height, decoding_schedule=decoding_schedule,vis_folder=self.output_dir)
        visualize_attention_mask(ar_structure.training_attention_mask, ar_structure.token_map, decoding_schedule, self.width, self.height, vis_folder=self.output_dir)
        visualize_adam_masks(adam_masks, self.output_dir / f"adam_mask_block_size_{self.base_block_size}.png")