import torch
import torch.distributed as dist
from autoregressive.models.utils.tokens import  TokenMap, TokenMapTensors, \
    TokenType, ImageToken
    
from jaxtyping import Float, Int64
from dataclasses import dataclass

from typing import Optional

@dataclass
class AutoRegressiveStructure:
    token_map: TokenMap
    token_map_tensors: TokenMapTensors
    decoding_schedule: list[list[int]]
    training_attention_mask: torch.Tensor

    def __init__(self,
                 logger, ##############################################
                 image_width: int,
                 image_height: int,
                 token_map: TokenMap,
                 decoded_masked_coords: list[torch.Tensor]):
        self.logger = logger ##############################################
      
        self.token_map = token_map
        self.token_map_tensors = TokenMapTensors(token_map, image_width, image_height)   
        self._cond_len = self.token_map.cond_len
        
        self.decoding_schedule = self.get_decoding_schedule(decoded_masked_coords) # decoding_schedule其实是output token的index的list，也就是从0到total_len-1
        self.training_attention_mask = self.get_training_attention_mask(self.decoding_schedule)
        
        if dist.get_rank() == 0:
            print(f"len(token_map): {len(token_map)}") ##############################################
            num_output_image_tokens = 0
            for i_pass, coords_i_pass in enumerate(decoded_masked_coords):
                print(f"pass {i_pass}:")
                coords_n_indics = []
                for coord in coords_i_pass.tolist():
                    x, y = coord
                    output_token_index = self.token_map.get_output_token_index((ImageToken(x, y)))
                    coords_n_indics.append([x,y,output_token_index])
                print(coords_n_indics)
                num_output_image_tokens += len(coords_i_pass.tolist())
            print("^"*50)
            print("input token and corresponding output token indices:")
            print(f"total num input tokens: {len(self.token_map._input_index.keys())}")
            total_input_image_tokens = 0
            for inp_token in self.token_map._input_index.keys():
                if inp_token.token_type() != TokenType.IMAGE:
                    continue
                output_token_indices = self.token_map._input_index[inp_token]
                total_input_image_tokens += len(output_token_indices)
                print(f"input token pos: [{inp_token.x_coord}, {inp_token.y_coord}] appeared {len(output_token_indices)} times in input sequence: {output_token_indices}")
            print(f"total input image tokens: {total_input_image_tokens}")
            print("^"*50)
            print(f"num_output_image_tokens: {num_output_image_tokens}") ##############################################
            print(f"Decoding schedule: {self.decoding_schedule}") ##############################################
            print(f"training attention mask: {self.training_attention_mask}") ##############################################
            print(f"shape of training attention mask: {self.training_attention_mask.shape}") ##############################################
        
    # @jaxtyped(typechecker=typechecker) (jaxtyped is not supported by torch.compile mode)
    def assemble_input_tokens(
        self,
        image_tokens: Float[torch.Tensor, "batch_size image_len embed_dim"],
        cond_tokens: Float[torch.Tensor, "batch_size cond_len embed_dim"],
        learnable_token: Float[torch.Tensor, "embed_dim"],
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
        device: Optional[torch.device] = None,
        is_random: bool = False, ##############################################
    ):
        if device is None:
            device = image_tokens.device

        batch_size, _, _ = image_tokens.shape
        num_total_tokens = self.token_map_tensors.out_token_indices.shape[0]
        embed_dim = image_tokens.shape[-1]

        input_tokens = torch.zeros(
            batch_size, num_total_tokens, embed_dim, device=device
        )

        image_mask = self.token_map_tensors.in_token_types == TokenType.IMAGE.value
        learned_mask = self.token_map_tensors.in_token_types == TokenType.LEARNED.value
        cond_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value

        image_indices = self.token_map_tensors.in_token_indices[image_mask]
        reordered_image_tokens = image_tokens[:, image_indices, :]

        input_tokens[:, image_mask, :] = reordered_image_tokens
        input_tokens[:, learned_mask, :] = learnable_token
        input_tokens[:, cond_mask, :] = cond_tokens

        freqs_cis = self.assemble_positional_embedding(freqs_cis)
        return input_tokens, freqs_cis

    def assemble_positional_embedding(
        self,
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
    ):
        in_img_mask = self.token_map_tensors.in_token_types == TokenType.IMAGE.value
        in_con_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value

        new_freqs_cis = torch.empty(in_img_mask.shape[0], freqs_cis.shape[1], freqs_cis.shape[2], device=freqs_cis.device)
        cond_lenn = self.token_map.cond_len

        new_freqs_cis[in_img_mask] = freqs_cis[
            self.token_map_tensors.in_token_indices[in_img_mask]+cond_lenn 
            # +cond_lenn is because the freqs_cis is of shape (cls_token_num+grid_size**2, head_dim // 2, 2)
            # the first frequency for image token is freqs_cis[cond_lenn]
        ]
        new_freqs_cis[in_con_mask] = freqs_cis[
            self.token_map_tensors.in_token_indices[in_con_mask]
        ]
        
        return new_freqs_cis
    
    def assemble_target_tokens(
        self,
        image_token_idx: Int64[torch.Tensor, "batch_size image_len"],
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = image_token_idx.device

        batch_size, _ = image_token_idx.shape
        num_total_tokens = self.token_map_tensors.out_token_indices.shape[0]

        target_tokens = torch.zeros(
            batch_size, num_total_tokens, dtype=torch.int64, device=device
        )
        target_mask = torch.zeros(
            batch_size, num_total_tokens, dtype=torch.bool, device=device
        )

        image_mask = self.token_map_tensors.out_token_types == TokenType.IMAGE.value
        empty_mask = self.token_map_tensors.out_token_types == TokenType.EMPTY.value
        assert (image_mask.int() + empty_mask.int() == 1).all(), (
            "Image and empty mask should be mutually exclusive"
        )
        reordered_image_token_idx = image_token_idx[
            :, self.token_map_tensors.out_token_indices[image_mask]
        ]

        target_tokens[:, image_mask] = reordered_image_token_idx
        target_mask[:, image_mask] = True
        
        return target_tokens, target_mask

    def assemble_input_tokens_for_decoding(
        self,
        image_tokens: Float[torch.Tensor, "batch_size image_len embed_dim"],
        cond_tokens: Float[torch.Tensor, "batch_size cond_len embed_dim"],
        learnable_token: Float[torch.Tensor, "embed_dim"],
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = image_tokens.device

        batch_size, _, _ = image_tokens.shape
        num_total_tokens = self.token_map_tensors.out_token_indices.shape[0]
        embed_dim = image_tokens.shape[-1]

        input_tokens = torch.zeros(
            batch_size, num_total_tokens, embed_dim, device=device
        )
        image_mask = self.token_map_tensors.in_token_types == TokenType.IMAGE.value
        learned_mask = self.token_map_tensors.in_token_types == TokenType.LEARNED.value
        cond_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value

        image_indices = self.token_map_tensors.in_token_indices[image_mask]
        reordered_image_tokens = image_tokens[:, image_indices, :]

        input_tokens[:, image_mask, :] = reordered_image_tokens
        input_tokens[:, learned_mask, :] = learnable_token
        input_tokens[:, cond_mask, :] = cond_tokens

        freqs_cis[~cond_mask] = freqs_cis[
            self.token_map_tensors.in_token_indices[~cond_mask]
        ]

        return input_tokens, freqs_cis
    
    def get_decoding_schedule(self, decoded_coords:list[torch.Tensor])-> list[list[int]]:
        num_pass = len(decoded_coords)
        decoding_schedule: list[list[int]] = []
        curr_max_decoded_idx = -1
        
        for i_pass in range(num_pass):
            tmp_coords = decoded_coords[i_pass]
            decoded_group = []
            for i, tmp_coord in enumerate(tmp_coords):
                x, y = tuple(tmp_coord)
                x, y = x.item(), y.item()
                output_token_index = self.token_map.get_output_token_index((ImageToken(x, y)))                    
                if curr_max_decoded_idx < output_token_index:
                    curr_max_decoded_idx = output_token_index
                else:
                    raise AssertionError(
                        f"Output token index {output_token_index} should be greater than previous max {curr_max_decoded_idx} in the autoregressive decoding schedule"
                    )
                
                decoded_group.append(output_token_index)
            
            
            # detect potential unused generated token between the generated image token groups
            assert len(decoded_group) != 0
            if len(decoding_schedule)!=0 and decoded_group[0] > decoding_schedule[-1][-1] + 1:
                unused_start_idx = decoding_schedule[-1][-1] + 1
                unused_end_idx = decoded_group[0]
                decoding_schedule.append(list(range(unused_start_idx, unused_end_idx)) + decoded_group)
            else:
                decoding_schedule.append(decoded_group)
            
        return decoding_schedule
    
    def get_training_attention_mask(self, decoding_schedule: list[list[int]]):
        total_len = len(self.token_map)  
        
        attention_mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        attention_mask[:self._cond_len, :self._cond_len] = torch.tril(torch.ones(self._cond_len, self._cond_len, dtype=torch.bool))
        prev_decoded_idx = list(range(0, self._cond_len-1))
        for idx in range(len(decoding_schedule)):
            prev_decoded_idx.extend(decoding_schedule[idx])
            rows, cols  = torch.meshgrid(
                torch.tensor(decoding_schedule[idx]), torch.tensor(prev_decoded_idx), indexing="ij"
            )
            attention_mask[rows, cols] = 1

        return attention_mask