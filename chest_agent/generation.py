"""Stop persistent token loops without altering medical words or source values."""

from transformers import StoppingCriteria


def repeated_suffix(tokens, max_period=32, repetitions=6, min_span=96):
    for period in range(1, max_period + 1):
        span = period * repetitions
        if span < min_span or span > len(tokens):
            continue
        block = tokens[-period:]
        if tokens[-span:] == block * repetitions:
            return True
    return False


class RepetitionStop(StoppingCriteria):
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length
        self.triggered = False

    def __call__(self, input_ids, scores, **kwargs):
        import torch

        generated = input_ids.shape[1] - self.prompt_length
        if generated >= 96 and generated % 16 == 0:
            tokens = input_ids[0, -min(generated, 192) :].tolist()
            self.triggered = repeated_suffix(tokens)
        return torch.full(
            (input_ids.shape[0],),
            self.triggered,
            device=input_ids.device,
            dtype=torch.bool,
        )
