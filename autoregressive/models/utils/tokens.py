import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

import torch
from jaxtyping import Int

from typing import Union


TOKEN_MAP_KEY_TYPE = Union[str, int]
INVALID_TOKEN = -100


class TokenType(Enum):
    IMAGE = torch.iinfo(torch.int).min
    LEARNED = torch.iinfo(torch.int).min + 1
    CONDITION = torch.iinfo(torch.int).min + 2
    EMPTY = torch.iinfo(torch.int).min + 3


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
    if type.lower() == "manhattan":
        return abs(token.x_coord - other.x_coord) + abs(token.y_coord - other.y_coord)
    elif type.lower() == "euclidean":
        return math.sqrt((token.x_coord - other.x_coord) ** 2 + (token.y_coord - other.y_coord) ** 2)
    else:
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
    _input_index: dict[Token, list[int]] = field(default_factory=lambda: defaultdict(list))
    _output_index: dict[Token, int] = field(default_factory=lambda: defaultdict(int))
    _cond_len: int = 0
    
    def __len__(self):
        return len(self._data)
    
    def __setitem__(self, in_token: Token, out_token: Token):
        self._data.append((in_token, out_token))
        self._input_index[in_token].append(len(self._data) - 1)
        
        assert out_token not in self._output_index, (
            f"Token {out_token} already exists in output index"
        )
        self._output_index[out_token] = len(self._data) - 1
        if isinstance(in_token, ConditionToken):
            self._cond_len += 1
    
    def __getitem__(self, in_token: Token):
        # Return all matches for the given in_token
        indices = self._input_index.get(in_token, [])
        return [self._data[i][1] for i in indices]
        
    def get_input_token_index(self, input_token: Token) -> list[int]:
        return self._input_index.get(input_token, [])
    
    def get_output_token_index(self, output_token: Token) -> int:
        return self._output_index[output_token]
    
    def input_tokens(self):
        return (in_token for in_token, _ in self._data)
    
    def output_tokens(self):
        return (out_token for _, out_token in self._data)

    @property
    def cond_len(self):
        return self._cond_len
    
    @property
    def data(self):
        return tuple(self._data)

    def items(self) -> Iterable[tuple[Token, Token]]:
        return iter(self._data)

@dataclass
class TokenMapTensors:
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

        for idx, (in_token, out_token) in enumerate(token_map.items()):
            
            out_token_type = out_token.token_type()
            if out_token_type == TokenType.IMAGE:
                self.out_token_indices[idx] = out_token.image_index(width) # 第i个out_token在图片中的index
                self.out_token_types[idx] = out_token_type.value
            elif out_token_type == TokenType.EMPTY:
                self.out_token_indices[idx] = TokenType.EMPTY.value
                self.out_token_types[idx] = TokenType.EMPTY.value
            elif out_token_type == TokenType.CONDITION or out_token_type == TokenType.LEARNED:
                assert False, (
                    "Condition or learnable token should not be in output token map"
                )
            else:
                assert False, (
                    f"Unknown output token type: {in_token_type}"
                )
                
            in_token_type = in_token.token_type()
            if in_token_type == TokenType.IMAGE or in_token_type == TokenType.LEARNED:
                self.in_token_indices[idx] = in_token.image_index(width)
                self.in_token_types[idx] = in_token_type.value
            elif in_token_type == TokenType.EMPTY:
                self.in_token_indices[idx] = TokenType.EMPTY.value
                self.in_token_types[idx] = TokenType.EMPTY.value
            elif in_token_type == TokenType.CONDITION:
                self.in_token_indices[idx] = in_token.cond_index
                self.in_token_types[idx] = in_token_type.value
            else:
                assert False, (f"Unknown input token type: {in_token_type}")


def find_close_min_token(
    token_map: TokenMap,
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    min_dist = float("inf")
    closest_min_token = None
    closest_min_index = float("inf") 
    
    for token in candidate_tokens:
        dist = spatial_token_distance(query_token, token)
        index = token.image_index(image_width)

        if dist < min_dist or (dist == min_dist and index < closest_min_index):
            min_dist = dist
            closest_min_token = token
            closest_min_index = index

    assert closest_min_token is not None, "No closest token found"
    return closest_min_token

def find_close_max_token(
    token_map: TokenMap,
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    min_dist = float("inf")
    closest_max_token = None
    closest_max_index = -1000000 
    
    for token in candidate_tokens:
        dist = spatial_token_distance(query_token, token)
        index = token.image_index(image_width)

        if dist < min_dist or (dist == min_dist and index > closest_max_index): 
            min_dist = dist
            closest_max_token = token
            closest_max_index = index

    assert closest_max_token is not None, "No closest token found"
    return closest_max_token

def find_close_unattach_min_token(
    token_map: TokenMap,
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    close_unattach_min_token = None
    min_dist = float("inf")
    close_unattach_min_index = 10000000
    close_unattach_min_outseq_index = 10000000
    lest_attached_times = float("inf")
    
    for index_outseq, token in enumerate(candidate_tokens):
        dist = spatial_token_distance(query_token, token)
        index = token.image_index(image_width)
        if token not in token_map._input_index.keys():
            attached_times = 0
        else:
            attached_times = len(token_map._input_index[token])

        if dist < min_dist or \
                (dist == min_dist and attached_times < lest_attached_times) or \
                (dist == min_dist and attached_times == lest_attached_times and index_outseq < close_unattach_min_outseq_index): ###############
            lest_attached_times = attached_times
            close_unattach_min_token = token
            close_unattach_min_outseq_index = index_outseq
            min_dist = dist

    assert close_unattach_min_token is not None, "No closest token found"
    return close_unattach_min_token


def find_close_unattach_max_token(
    token_map: TokenMap,
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    close_unattach_max_token = None
    min_dist = float("inf")
    close_unattach_max_index = -10000000
    close_unattach_max_outseq_index = -10000000
    lest_attached_times = float("inf")
    
    for index_outseq, token in enumerate(candidate_tokens):
        dist = spatial_token_distance(query_token, token)
        index = token.image_index(image_width)
        if token not in token_map._input_index.keys():
            attached_times = 0
        else:
            attached_times = len(token_map._input_index[token])

        if dist < min_dist or \
                (dist == min_dist and attached_times < lest_attached_times) or \
                (dist == min_dist and attached_times == lest_attached_times and index_outseq > close_unattach_max_outseq_index): ###############
            lest_attached_times = attached_times
            close_unattach_max_token = token
            close_unattach_max_outseq_index = index_outseq
            min_dist = dist

    assert close_unattach_max_token is not None, "No closest token found"
    return close_unattach_max_token


def find_close_left_up_token(
    token_map: TokenMap,
    query_token: SpatialToken, 
    candidate_tokens: Sequence[SpatialToken],
    image_width: int
) -> SpatialToken:
    # If there are multiple closest tokens, return the one with the smallest index
    if len(candidate_tokens) == 0:
        raise ValueError("No candidate tokens provided")
    
    close_left_up_token = None
    min_dist = float("inf")
    close_left_up_index = -10000000
    close_left_up_outseq_index = -10000000
    
    for index_outseq, token in enumerate(candidate_tokens):
        dist = spatial_token_distance(query_token, token)
        index = token.image_index(image_width)
        if dist < min_dist or \
                (dist == min_dist and token.x_coord < query_token.x_coord) or \
                (dist == min_dist and token.x_coord == query_token.x_coord and token.y_coord < query_token.y_coord): ###############
            close_left_up_token = token
            close_left_up_outseq_index = index_outseq
            min_dist = dist

    assert close_left_up_token is not None, "No closest token found"
    return close_left_up_token