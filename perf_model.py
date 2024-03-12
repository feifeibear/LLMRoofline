import sys

sys.path.insert(0, './LLM-Viewer')
from model_analyzer import ModelAnalyzer


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

def get_naive_perf_model(peak_flop, bw, bs, in_len, kv_len, h, hi, gqa_grp = 1):
    ai = get_ai(bs, in_len, kv_len, h, hi, gqa_grp)
    return min(peak_flop, bw * ai)

def get_naive_perf_model_moe(peak_flop, bw, bs, in_len, kv_len, h, hi, gqa_grp, expert_count, activate_expert_count):
    ai = get_ai_moe(bs, in_len, kv_len, h, hi, gqa_grp, expert_count, activate_expert_count)
    return min(peak_flop, bw * ai)


def get_llm_viewer_model(model_id, hardware, bs, kv_len):
    analyzer=ModelAnalyzer(model_id, hardware)
    results=analyzer.analyze(batchsize=bs, seqlen=kv_len, use_flashattention=True)
    return 1/results["total_results"]["decode"]["inference_time"]