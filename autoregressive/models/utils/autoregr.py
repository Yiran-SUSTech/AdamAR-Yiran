import torch
import torch.distributed as dist
from autoregressive.models.utils.tokens import  TokenMap, TokenMapTensors, \
    TokenType, ImageToken
    
from jaxtyping import Float, Int64
from dataclasses import dataclass

from typing import Optional
import copy

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
                 ori_masked_coords: list[torch.Tensor],
                 decoded_masked_coords: list[torch.Tensor],
                 freqs_cis_reorder_shceme: str = None,
                 ):
        self.logger = logger ##############################################
      
        self.token_map = token_map
        self.token_map_tensors = TokenMapTensors(token_map, image_width, image_height)   
        self._cond_len = self.token_map.cond_len
        self.decoded_masked_coords = copy.deepcopy(decoded_masked_coords)
        self.ori_masked_coords = copy.deepcopy(ori_masked_coords)
        
        self.decoding_schedule = self.get_decoding_schedule(decoded_masked_coords) # decoding_schedule其实是output token的index的list，也就是从0到total_len-1
        self.training_attention_mask = self.get_training_attention_mask(self.decoding_schedule)
        self.freqs_cis_reorder_shceme = freqs_cis_reorder_shceme
        
        # if dist.get_rank() == 0:
        #     print(f"len(token_map): {len(token_map)}") ##############################################
        #     self.logger.info(f"len(token_map): {len(token_map)}")
        #     num_output_image_tokens = 0
        #     for i_pass, coords_i_pass in enumerate(decoded_masked_coords):
        #         print(f"pass {i_pass}:")
        #         self.logger.info(f"pass {i_pass}:")
        #         coords_n_indics = []
        #         for coord in coords_i_pass.tolist():
        #             x, y = coord
        #             output_token_index = self.token_map.get_output_token_index((ImageToken(x, y)))
        #             coords_n_indics.append([x,y,output_token_index])
        #         print(coords_n_indics)
        #         self.logger.info(coords_n_indics)
        #         num_output_image_tokens += len(coords_i_pass.tolist())
        #     print("^"*50)
        #     self.logger.info("^"*50)
        #     print("input token and corresponding output token indices:")
        #     self.logger.info("input token and corresponding output token indices:")
        #     print(f"total num input tokens: {len(self.token_map._input_index.keys())}")
        #     self.logger.info(f"total num input tokens: {len(self.token_map._input_index.keys())}")
        #     total_input_image_tokens = 0
        #     for inp_token in self.token_map._input_index.keys():
        #         if inp_token.token_type() != TokenType.IMAGE:
        #             continue
        #         output_token_indices = self.token_map._input_index[inp_token]
        #         output_token_indices2 = self.token_map._input_index.get(inp_token, [])
        #         output_token_indices3 = self.token_map.get_input_token_index(inp_token)
        #         total_input_image_tokens += len(output_token_indices)
        #         print(f"input token pos: [{inp_token.x_coord}, {inp_token.y_coord}] appeared {len(output_token_indices)} times in input sequence: {output_token_indices}")
        #         self.logger.info(f"input token pos: [{inp_token.x_coord}, {inp_token.y_coord}] appeared {len(output_token_indices)} times in input sequence: {output_token_indices}")
        #         print(f"output_token_indices2 from get method: {output_token_indices2}")
        #         print(f"output_token_indices3 from get_input_token_index method: {output_token_indices3}")
        #     print(f"total input image tokens: {total_input_image_tokens}")
        #     self.logger.info(f"total input image tokens: {total_input_image_tokens}")
        #     print("^"*50)
        #     self.logger.info("^"*50)
        #     print(f"num_output_image_tokens: {num_output_image_tokens}") ##############################################
        #     self.logger.info(f"num_output_image_tokens: {num_output_image_tokens}")
        #     print(f"Decoding schedule: {self.decoding_schedule}") ##############################################
        #     self.logger.info(f"Decoding schedule: {self.decoding_schedule}")
        #     print(f"training attention mask: {self.training_attention_mask}") ##############################################
        #     self.logger.info(f"training attention mask: {self.training_attention_mask}")
        #     print(f"shape of training attention mask: {self.training_attention_mask.shape}") ##############################################
        #     self.logger.info(f"shape of training attention mask: {self.training_attention_mask.shape}")
        #     print(f"token_map._data: {self.token_map._data}")
        #     self.logger.info(f"token_map._data: {self.token_map._data}")
        
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
        assert len(image_indices) == num_total_tokens - cond_mask.sum(), (
            "the number of image tokens in the input sequence should be equal to total tokens - condition tokens"
        )

        reordered_image_tokens = image_tokens[:, image_indices, :]
        # if dist.get_rank() == 0:
        #     print(f"self.token_map_tensors.out_token_indices.shape[0]: {self.token_map_tensors.out_token_indices.shape[0]}")
        #     print(f"self.token_map_tensors.in_token_indices.shape[0]: {self.token_map_tensors.in_token_indices.shape[0]}")
        #     print(f"image_indices.shape: {image_indices.shape}") ##############################################
        #     print(f"image_indices: {image_indices}")
        #     print(f"length of image_indices: {len(image_indices)}, number of image tokens: {image_tokens.shape[1]}") ##############################################

        input_tokens[:, image_mask, :] = reordered_image_tokens
        input_tokens[:, learned_mask, :] = learnable_token
        input_tokens[:, cond_mask, :] = cond_tokens

        freqs_cis = self.assemble_positional_embedding(freqs_cis)
        return input_tokens, freqs_cis

    def assemble_positional_embedding(
        self,
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
    ):
        out_image_mask = self.token_map_tensors.out_token_types == TokenType.IMAGE.value
        inp_image_mask = self.token_map_tensors.in_token_types == TokenType.IMAGE.value
        inp_cond_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value
        new_freqs_cis = torch.empty(out_image_mask.shape[0], freqs_cis.shape[1], freqs_cis.shape[2], device=freqs_cis.device)
        cond_lenn = self.token_map.cond_len
        assert freqs_cis.shape[0] - cond_lenn == out_image_mask.shape[0], (
            "The length of freqs_cis - cond_lenn should match the total number of output tokens (total length of the output sequence)"
        )
        
        if self.freqs_cis_reorder_shceme == 'output_reorder':
            out_image_indices = self.token_map_tensors.out_token_indices[out_image_mask] + cond_lenn
            new_freqs_cis = freqs_cis[
                out_image_indices, :, :
            ]
        elif self.freqs_cis_reorder_shceme == 'input_reorder':
            inp_image_indices = self.token_map_tensors.in_token_indices[inp_image_mask] + cond_lenn
            inp_cond_indices = self.token_map_tensors.in_token_indices[inp_cond_mask]
            new_freqs_cis[inp_image_mask] = freqs_cis[inp_image_indices]
            new_freqs_cis[inp_cond_mask] = freqs_cis[inp_cond_indices]
        elif self.freqs_cis_reorder_shceme == 'None':
            new_freqs_cis = freqs_cis[:-1]
        else:
            assert False, f"Unknown freqs_cis_reorder_shceme: {self.freqs_cis_reorder_shceme}"
                
        # T_flat = new_freqs_cis.view(256, -1) 
        # # T_flat 的形状现在是 [256, 16]

        # # 步骤 2: 使用 unique(dim=0) 查找唯一的行
        # # unique(dim=0) 会返回所有在第 0 维度上不重复的元素（即不重复的 [8, 2] 向量）
        # T_unique = torch.unique(T_flat, dim=0)

        # # 步骤 3: 比较唯一元素的数量与原始数量
        # original_count = T_flat.shape[0]   # 256
        # unique_count = T_unique.shape[0] # 唯一的 [8, 2] 张量数量

        # # 最终判断
        # has_duplicates = unique_count < original_count
        # if dist.get_rank() == 0:
        #     print(f"original_count: {original_count}")
        #     print(f"unique_count: {unique_count}")
        #     print(f"has_duplicates: {has_duplicates}")
        
        return new_freqs_cis
    
    def assemble_sinusoidal_positional_embedding(
        self,
        SinusoidalPosEmb: Float[torch.Tensor, "total_len _ dim"],
    ):
        out_image_mask = self.token_map_tensors.out_token_types == TokenType.IMAGE.value
        inp_image_mask = self.token_map_tensors.in_token_types == TokenType.IMAGE.value
        inp_cond_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value

        new_SinusoidalPosEmb = torch.empty(out_image_mask.shape[0], SinusoidalPosEmb.shape[1], device=SinusoidalPosEmb.device)
        cond_lenn = self.token_map.cond_len
        assert SinusoidalPosEmb.shape[0] - cond_lenn == out_image_mask.shape[0], (
            "The length of SinusoidalPosEmb - cond_lenn should match the total number of output tokens (total length of the output sequence)"
        )
        
        out_image_indices = self.token_map_tensors.out_token_indices[out_image_mask] + cond_lenn
        if dist.get_rank() == 0:
            print(f"out_image_indices.shape: {out_image_indices.shape}") ##############################################
            print(f"out_image_indices: {out_image_indices}") ##############################################
        new_SinusoidalPosEmb = SinusoidalPosEmb[out_image_indices]
        
        return new_SinusoidalPosEmb
    
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
        assert image_mask.sum() == len(self.token_map), (
            "The number of image tokens in the output sequence should be equal to total tokens"
        )
        assert (image_mask.int() + empty_mask.int() == 1).all(), (
            "Image and empty mask should be mutually exclusive"
        )
        reordered_image_token_idx = image_token_idx[
            :, self.token_map_tensors.out_token_indices[image_mask]
        ]

        target_tokens[:, image_mask] = reordered_image_token_idx
        target_mask[:, image_mask] = True
        
        # if dist.get_rank() == 0:
        #     print(f"target_tokens.shape: {target_tokens.shape}") ##############################################
        #     print(f"target_mask.shape: {target_mask.shape}") ##############################################
        #     print(f"image_mask: {image_mask}") ##############################################
        #     print(f"empty_mask: {empty_mask}") ##############################################

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
    