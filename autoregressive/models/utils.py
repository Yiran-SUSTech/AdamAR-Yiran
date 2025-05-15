import math
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple, Sequence
from collections import defaultdict

from matplotlib.backend_bases import CloseEvent
import torch
from beartype import beartype as typechecker
from jaxtyping import Float, Int, Int64, jaxtyped

AXIS_ALIGNED_KEY = "axis_aligned"
NONAXIS_ALIGNED_KEY = "nonaxis_aligned"
TOKEN_MAP_KEY_TYPE = str | int
INVALID_TOKEN = -100


class TokenType(Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return torch.iinfo(torch.int).min + count

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


def spatial_token_distance(token: SpatialToken, other: SpatialToken, type: str="manhattan") -> float:
    match type.lower(): 
        case "manhattan":
            return abs(token.x_coord - other.x_coord) + abs(token.y_coord - other.y_coord)
        case "euclidean":
            return math.sqrt((token.x_coord - other.x_coord) ** 2 + (token.y_coord - other.y_coord) ** 2)
        case _:
            raise ValueError(f"Unknown distance type: {type}")
        
    
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

@dataclass
class TokenMap:
    _data: list[tuple[Token, Token]] = field(default_factory=list)
    _index: dict[Token, list[int]] = field(default_factory=lambda: defaultdict(list))
        
    def __len__(self):
        return len(self._data)

    def __setitem__(self, in_token: Token, out_token: Token):
        self._data.append((in_token, out_token))
        self._index[in_token].append(len(self._data) - 1)
        
    def __getitem__(self, key: int | tuple[int, int]):
        match key:
            case int():
                return self._data[key]
            case Token():
                # Return all matches for the given in_token
                indices = self._index.get(key, [])
                return [self._data[i] for i in indices]
            case (x, y):
                learned_indices = self._index.get(LearnedToken(x, y), [])
                image_indices = self._index.get(ImageToken(x, y), [])
                indices = learned_indices + image_indices
                return [self._data[i] for i in indices]
            case _:
                raise KeyError(f"Key {key} not found in TokenMap")

    def input_tokens(self):
        return (in_token for in_token, _ in self._data)
    
    def output_tokens(self):
        return (out_token for _, out_token in self._data)

    @property
    def data(self):
        return tuple(self._data)

@dataclass
class TokenMapTensors_v2:
    # input tokens
    in_token_indices: Int[torch.Tensor, "num_total_tokens"]
    in_token_types: Int[torch.Tensor, "num_total_tokens"]

    # output tokens
    out_token_indices: Int[torch.Tensor, "num_total_tokens"]
    out_token_types: Int[torch.Tensor, "num_total_tokens"]

    def __init__(self, token_map, width: int, height: int):
        num_total_tokens = len(token_map)
        self.out_token_indices = torch.full(
            (num_total_tokens,), TokenType.EMPTY.value, dtype=torch.int
        )
        self.out_token_types = torch.full(
            (num_total_tokens,), TokenType.EMPTY.value, dtype=torch.int
        )
        self.in_token_indices = torch.full(
            (num_total_tokens,), TokenType.EMPTY.value, dtype=torch.int
        )
        self.in_token_types = torch.full(
            (num_total_tokens,), TokenType.EMPTY.value, dtype=torch.int
        )

        for idx, (in_token, out_token) in enumerate(token_map):
            match out_token.token_type():
                case TokenType.IMAGE:
                    self.out_token_indices[idx] = out_token.image_index(width)
                    self.out_token_types[idx] = out_token.token_type().value
                case TokenType.EMPTY:
                    self.out_token_indices[idx] = TokenType.EMPTY.value
                    self.out_token_types[idx] = TokenType.EMPTY.value
                case TokenType.CONDITION | TokenType.LEARNED:
                    assert False, (
                        "Condition or learnable token should not be in output token map"
                    )

            in_token_type = in_token.token_type()
            match in_token_type:
                case TokenType.IMAGE | TokenType.LEARNED:
                    self.in_token_indices[idx] = in_token.image_index(width)
                    self.in_token_types[idx] = in_token_type.value
                case TokenType.EMPTY:
                    self.in_token_indices[idx] = TokenType.EMPTY.value
                    self.in_token_types[idx] = TokenType.EMPTY.value
                case TokenType.CONDITION:
                    self.in_token_indices[idx] = in_token.cond_index
                    self.in_token_types[idx] = in_token_type.value


@dataclass(frozen=True)
class AutoRegressiveStructure:
    token_map: TokenMap
    token_map_tensors: TokenMapTensors_v2
    
    cond_len: int
    total_len: int
    attention_mask: torch.Tensor | None = None


    # @jaxtyped(typechecker=typechecker) (jaxtyped is not supported by torch.compile mode)
    def assemble_input_tokens(
        self,
        image_tokens: Float[torch.Tensor, "batch_size image_len embed_dim"],
        cond_tokens: Float[torch.Tensor, "batch_size cond_len embed_dim"],
        learnable_token: Float[torch.Tensor, "embed_dim"],
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
        device: torch.device | None = None,
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

    # @jaxtyped(typechecker=typechecker) (jaxtyped is not supported by torch.compile mode)
    def assemble_positional_embedding(
        self,
        freqs_cis: Float[torch.Tensor, "total_len _ 2"],
    ):
        cond_mask = self.token_map_tensors.in_token_types == TokenType.CONDITION.value
        freqs_cis[~cond_mask] = freqs_cis[
            self.token_map_tensors.out_token_indices[~cond_mask]
        ]
        return freqs_cis
    
    # @jaxtyped(typechecker=typechecker) (jaxtyped is not supported by torch.compile mode)
    def assemble_target_tokens(
        self,
        image_token_idx: Int64[torch.Tensor, "batch_size image_len"],
        device: torch.device | None = None,
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
        device: torch.device | None = None,
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
    
    def fastest_decoding_schedule(self):
        # the decoding schedule dependent on the token map
        # which several output tokens can be generated at once
        # this will require the output token to not dependent on the input token that has not been generated yet

        list_input_tokens = list(self.token_map.input_tokens())
        list_output_tokens = list(self.token_map.output_tokens())
        
        total_len = len(list_input_tokens)
        
        prev_output_tokens: set[Token] = {EmptyToken()}
        parallel_decoding_groups= []
        
        start = self.cond_len - 1
        end = self.cond_len
        tmp_parallel_decoding_idx = [start]
        while end < len(self.token_map):
            if list_input_tokens[end] in prev_output_tokens or list_input_tokens[end].token_type() == TokenType.LEARNED:
                tmp_parallel_decoding_idx.append(end)
                end += 1
            else:
                parallel_decoding_groups.append(tmp_parallel_decoding_idx)
                prev_output_tokens.update({list_output_tokens[idx] for idx in range(start, end)})
                start = end
                end = start + 1
                tmp_parallel_decoding_idx = [start]
                
        parallel_decoding_groups.append(tmp_parallel_decoding_idx)
        return parallel_decoding_groups
    
    def get_training_attention_mask(self, parallel_decoding_groups: list[list[int]]):

        attention_mask = torch.zeros(self.total_len, self.total_len, dtype=torch.bool)
        attention_mask[:self.cond_len, :self.cond_len] = torch.tril(torch.ones(self.cond_len, self.cond_len, dtype=torch.bool))
        prev_decoded_idx = list(range(0, self.cond_len-1))
        for idx in range(len(parallel_decoding_groups)):
            prev_decoded_idx.extend(parallel_decoding_groups[idx])
            
            rows, cols  = torch.meshgrid(
                torch.tensor(parallel_decoding_groups[idx]), torch.tensor(prev_decoded_idx), indexing="ij"
            )
            attention_mask[rows, cols] = 1

        return attention_mask
    
    
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


def _find_closest_token(
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    min_dist = float("inf")
    closest_token = None
    closest_index = float("inf")

    for token in candidate_tokens:
        dist = spatial_token_distance(query_token, token)
        index = query_token.image_index(image_width)

        if dist < min_dist or (dist == min_dist and index < closest_index):
            min_dist = dist
            closest_token = token
            closest_index = index

    assert closest_token is not None, "No closest token found"
    return closest_token

    
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
    first_image_token = ImageToken(
        x_coord=first_pass_coords[0][0], y_coord=first_pass_coords[0][1]
    )
    attention_mask[
        cond_len + len(first_pass_coords) :, cond_len + len(first_pass_coords) :
    ] = 0

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

    # prepare attention mask
    attention_mask[
        cond_len + len(first_pass_coords) :, cond_len + len(first_pass_coords) :
    ] = bi_attention_mask

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
        if i_pass == 1:
            # learnable token
            for idx, coord in enumerate(curr_coords):
                x, y = coord
                learnable_token = LearnedToken(x_coord=x, y_coord=y)
                pred_img_token = ImageToken(x_coord=x, y_coord=y)
                token_map[learnable_token] = pred_img_token
                # input_token_group.append(curr_img_token)
        else:
            # previous_coords = masked_coords[i_pass - 1].tolist()
            # prev_coord_len = len(previous_coords)
            # assert prev_coord_len == len(curr_coords) // 2
            generated_tokens = list(t for t in token_map.output_tokens() if t.token_type() == TokenType.IMAGE)
            for curr_coord in curr_coords:
                x, y = curr_coord
                curr_img_token = ImageToken(x_coord=x, y_coord=y)
                closest_token = _find_closest_token(
                    curr_img_token, generated_tokens, width
                )
                token_map[closest_token] = curr_img_token
                
            '''
            for prev_coord, curr_coord in zip(
                previous_coords, curr_coords[:prev_coord_len]
            ):
                prev_x, prev_y = prev_coord
                curr_x, curr_y = curr_coord
                prev_img_token = ImageToken(x_coord=prev_x, y_coord=prev_y)
                curr_img_token = ImageToken(x_coord=curr_x, y_coord=curr_y)

                token_map[prev_img_token] = curr_img_token

            for curr_coord in curr_coords[prev_coord_len:]:
                x, y = curr_coord
                learnable_token = LearnedToken(x_coord=x, y_coord=y)
                token_map[learnable_token] = ImageToken(x_coord=x, y_coord=y)
            '''

    token_map_tensors = TokenMapTensors_v2(token_map, width, height)
    return AutoRegressiveStructure(
        token_map=token_map,
        token_map_tensors=token_map_tensors,
        cond_len=cond_len,
        total_len=total_len,
        attention_mask=attention_mask,
    )


def get_autoregressive_structure(
    width: int,
    height: int,
    base_block_size: int,
    cond_len: int,
) -> AutoRegressiveStructure:
    _, masked_coords, _ = _generalized_adam_interlacing(width, height, base_block_size)
    ar_structure = _get_image_token_index_map_v2(
        width,
        height,
        cond_len,
        masked_coords,
    )
    return ar_structure
