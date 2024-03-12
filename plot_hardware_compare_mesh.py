import matplotlib.pyplot as plt
import numpy as np
from perf_model import get_naive_perf_model_moe, get_naive_perf_model, get_llm_viewer_model


import argparse
import matplotlib.pyplot as plt
import numpy as np
from perf_model import get_naive_perf_model_moe, get_naive_perf_model, get_llm_viewer_model

# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Plot Speedup for LLM Models')

# 添加参数
parser.add_argument('--max_bs', type=int, default=257, help='Maximum Batch Size')
parser.add_argument('--max_seq_len', type=int, default=2049, help='Maximum Sequence Length')
parser.add_argument('--model_name', type=str, default="LLAMA2_70B", help='Model Name')
parser.add_argument('--hw1', type=str, default="nvidia_A100", help='First Hardware Name')
parser.add_argument('--hw2', type=str, default="nvidia_H20", help='Second Hardware Name')

# 解析参数
args = parser.parse_args()

max_bs = args.max_bs
max_seq_len = args.max_seq_len
model_name = args.model_name
hw1_name = args.hw1
hw2_name = args.hw2

bs_values = np.arange(1, max_bs, 8)
seq_len_values = np.arange(128, max_seq_len, 64)

fig, ax = plt.subplots()

color_grid = np.zeros((len(bs_values), len(seq_len_values)))


hardware_dict = {
    "GH200-new" : {"flops": 989.5, "bwd": 4.9},
    "GH200-old" : {"flops": 989.5, "bwd": 4.0},
    "H800" : {"flops": 989.5, "bwd": 3.35},
    "H200" : {"flops": 989.5, "bwd": 4.8},
    "nvidia_H20" : {"flops": 148, "bwd": 4.0},
    "nvidia_A100" : {"flops": 312, "bwd": 2.039},
    "L40S" : {"flops": 362, "bwd": 0.864},
    "L40" : {"flops": 181, "bwd": 0.864},
    "L20" : {"flops": 119.5, "bwd": 0.864},
    "A30" : {"flops": 165, "bwd": 0.933},
    "4090" : {"flops": 330, "bwd": 1.008},
    
}


model_dict = {
"LLAMA2_70B":
# https://huggingface.co/TheBloke/Llama-2-70B-fp16/blob/main/config.json
{
"h" : 8192,
"hi" : 28672,
"gqa_grp" : 64 / 8,
"type" : "dense",
"model_id" : "/Users/jiaruifang/Documents/plot/perfmodel/models/Llama-2-70b",
},

"LLAMA2_13B":
# https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
{
"h" : 5120,
"hi" : 13824,
"gqa_grp" : 1,
"type" : "dense",
"model_id" : "/Users/jiaruifang/Documents/plot/perfmodel/models/Llama-2-13b",
},
"LLAMA2_7B":
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
{
"h" : 4096,
"hi" : 11008,
"gqa_grp" : 1,
"type" : "dense",
"model_id" : "/Users/jiaruifang/Documents/plot/perfmodel/models/Llama-2-7b",
},

# mistral 7B v2
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
"mistral_7B_v2":
{
"h" : 4096,
"hi" : 14336,
"gqa_grp" : 32 / 8,
"type" : "dense",
"model_id" : "/Users/jiaruifang/Documents/plot/perfmodel/models/mistral_7B_v2",
},
"Mixtral_8x7B":
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json
{
"h" : 4096,
"hi" : 14336,
"gqa_grp" : 32 / 8,
"type" : "moe",
"model_id" : "/Users/jiaruifang/Documents/plot/perfmodel/models/Mixtral_8x7B",
},
}

hidden_size = model_dict[model_name]["h"]
intermediate_size = model_dict[model_name]["hi"]
gqa_grp = model_dict[model_name]["gqa_grp"]
type = model_dict[model_name]["type"]
model_id = model_dict[model_name]["model_id"]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for ax, USE_LLM_VIEWER in zip(axs, [True, False]):
    color_grid = np.zeros((len(bs_values), len(seq_len_values)))

    for i, bs in enumerate(bs_values):
        for j, seq_len in enumerate(seq_len_values):
            if USE_LLM_VIEWER:
                perf1 = get_llm_viewer_model(model_id, hw1_name, bs, seq_len)
                perf2 = get_llm_viewer_model(model_id, hw2_name, bs, seq_len)
            else:
                flops1 = hardware_dict[hw1_name]["flops"]
                bwd1 = hardware_dict[hw1_name]["bwd"]
                flops2 = hardware_dict[hw2_name]["flops"]
                bwd2 = hardware_dict[hw2_name]["bwd"]

                if type == "moe":
                    perf1 = get_naive_perf_model_moe(flops1, bwd1, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp, 8, 2)
                    perf2 = get_naive_perf_model_moe(flops2, bwd2, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp, 8, 2)
                else:
                    perf1 = get_naive_perf_model(flops1, bwd1, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp)
                    perf2 = get_naive_perf_model(flops2, bwd2, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp)

            condition = perf1 / perf2
            color_grid[i, j] = condition

    cax = ax.pcolormesh(seq_len_values, bs_values, color_grid, shading='auto', cmap='viridis')
    fig.colorbar(cax, ax=ax)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Batch Size')

    suffix = " LLM-Viewer" if USE_LLM_VIEWER else " Naive"
    ax.set_title(f"Speedup {hw1_name}/{hw2_name} ({model_name})" + suffix)

    ax.grid(True)

plt.tight_layout()
plt.show()