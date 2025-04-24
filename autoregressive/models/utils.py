import torch
import math
import pathlib

def get_adam_pattern(base_block_size: int):
    assert (base_block_size & (base_block_size - 1)) == 0, "base_block_size must be a power of 2"
    levels = int(math.log2(base_block_size))
    num_passes = 2 * levels + 1

    patterns = []
    for i in range(num_passes):
        if i == 0:
            patterns.append((0, 0, base_block_size, base_block_size))
            continue
        else:
            if i % 2 == 0:
                x_start = 0 
                y_start = int(base_block_size / (2**(i // 2)))
                x_stride = y_start
                y_stride = x_stride * 2
            else:
                x_start = int(base_block_size / (2**((i+1) // 2)))
                y_start = 0
                x_stride = x_start * 2
                y_stride = x_stride
            
            patterns.append((x_start, y_start, x_stride, y_stride))
    return patterns

def generalized_adam_interlacing(width: int, height: int, base_block_size: int) -> list[torch.Tensor]:

    assert width  >= base_block_size
    assert height >= base_block_size

    patterns = get_adam_pattern(base_block_size)
    adam_masks = []
    filled = torch.zeros(height, width, dtype=bool)

    for x_start, y_start, x_step, y_step in patterns:
        mask = torch.zeros(height, width, dtype=bool)
        for y in range(y_start, height, y_step):
            for x in range(x_start, width, x_step):
                if not filled[y, x]:
                    mask[y, x] = True
                    filled[y, x] = True
        adam_masks.append(mask)

    return adam_masks

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



def test_attention_mask(cond_len: int, attention_mask: torch.Tensor, adam_masks: list[torch.Tensor]):
    assert (attention_mask[:cond_len, :cond_len] == torch.tril(torch.ones(cond_len, cond_len, dtype=torch.bool))).all()
    assert attention_mask[cond_len:, :cond_len].all()
    img_attention_mask = attention_mask[cond_len:, cond_len:]
    prev_indices = []
    for mask in adam_masks:
        curr_indices = mask.view(-1).nonzero()[:, 0].tolist()
        prev_indices.extend(curr_indices)
        assert img_attention_mask[torch.tensor(curr_indices)[:, None], torch.tensor(prev_indices)[None, :]].all()

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
    width, height = 8, 8
    base_block_size = 4
    cond_len = 2
    masks = generalized_adam_interlacing(width, height, base_block_size)
    attention_mask = _get_attention_mask_from_adam(cond_len, masks)
    test_attention_mask(cond_len, attention_mask, masks)