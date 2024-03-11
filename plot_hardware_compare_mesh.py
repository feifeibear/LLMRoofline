import matplotlib.pyplot as plt
import numpy as np
from utils import get_ai, get_ai_moe

bs_values = np.arange(1, 257, 8)
seq_len_values = np.arange(128, 2049, 64)

fig, ax = plt.subplots()

color_grid = np.zeros((len(bs_values), len(seq_len_values)))


hardware_dict = {
    "GH200-new" : {"flops": 989.5, "bwd": 4.9},
    "GH200-old" : {"flops": 989.5, "bwd": 4.0},
    "H800" : {"flops": 989.5, "bwd": 3.35},
    "H200" : {"flops": 989.5, "bwd": 4.8},
    "H20" : {"flops": 148, "bwd": 4.0},
    "A800" : {"flops": 312, "bwd": 2.039},
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
},

"LLAMA_13B":
# https://huggingface.co/meta-llama/Llama-2-13b-hf/blob/main/config.json
{
"h" : 5120,
"hi" : 13824,
"gqa_grp" : 1,
"type" : "dense",
},
"LLAMA_7B":
# https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/blob/main/config.json
{
"h" : 4096,
"hi" : 11008,
"gqa_grp" : 1,
"type" : "dense",
},

# mistral 7B v2
# https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/config.json
"mistral_7B_v2":
{
"h" : 4096,
"hi" : 14336,
"gqa_grp" : 32 / 8,
"type" : "dense",
},
"Mixtral-8x7B":
# https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1/blob/main/config.json
{
"h" : 4096,
"hi" : 14336,
"gqa_grp" : 32 / 8,
"type" : "moe",
},
}

model_name = "Mixtral-8x7B"

h1_name = "A800"
h2_name = "H20"

hidden_size = model_dict[model_name]["h"]
intermediate_size = model_dict[model_name]["hi"]
gqa_grp = model_dict[model_name]["gqa_grp"]
type = model_dict[model_name]["type"]

for i, bs in enumerate(bs_values):
    for j, seq_len in enumerate(seq_len_values):
        if type == "moe":
            ai = get_ai_moe(bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp, 8, 2)
        else:
            ai = get_ai(bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp)
        # 1 gaudi good, 0 h20 good
        h1 = hardware_dict[h1_name]["flops"]
        b1 = hardware_dict[h1_name]["bwd"]

        h2 = hardware_dict[h2_name]["flops"]
        b2 = hardware_dict[h2_name]["bwd"]

        condition = min(h1, ai * b1) / min(h2, ai * b2)
        color_grid[i, j] = condition

cax = ax.pcolormesh(seq_len_values, bs_values, color_grid, shading='auto', cmap='viridis')


fig.colorbar(cax, ax=ax)

ax.set_xlabel('Sequence Length')
ax.set_ylabel('Batch Size')
ax.set_title(f"Speedup {h1_name}/{h2_name} ({model_name})")

ax.grid(True)

plt.show()
