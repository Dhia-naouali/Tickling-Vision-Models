channel_idx = ...
donor = ...

def ablation_hook(module, input, output):
    output[:, channel_idx].zero_()

def donor_hook(module, input, output):
    output[:, channel_idx].copy_(donor[:, channel_idx])
