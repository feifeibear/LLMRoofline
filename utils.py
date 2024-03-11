def get_ai(bs, in_len, kv_len, h, hi, gqa_grp = 1):
    qkv_flops = 6 * bs * in_len * h**2
    attn_flops =  4 * bs * in_len**2 * h
    o_flops = 2 * bs * in_len * h * h
    ffn_flops = 2 * 2 * bs * in_len * h * hi

    qkvo_param = 4 * h**2 + 4 * h
    ffn_params = (2 * h * hi + h + hi)
    kv_params = 2 * bs * kv_len * h / gqa_grp
    return (qkv_flops + attn_flops +  o_flops + ffn_flops) / ((qkvo_param + ffn_params + kv_params) * 2) 


def get_ai_moe(bs, in_len, kv_len, h, hi, gqa_grp, expert_count, activate_expert_count):
    qkv_flops = 6 * bs * in_len * h**2
    attn_flops =  4 * bs * in_len**2 * h
    o_flops = 2 * bs * in_len * h * h
    ffn_flops = 2 * 2 * bs * in_len * h * hi * activate_expert_count

    qkvo_param = 4 * h**2 + 4 * h
    ffn_params = (2 * h * hi + h + hi) * expert_count
    kv_params = 2 * bs * kv_len * h / gqa_grp
    return (qkv_flops + attn_flops +  o_flops + ffn_flops) / ((qkvo_param + ffn_params + kv_params) * 2) 