import argparse
import matplotlib.pyplot as plt
import numpy as np
from perf_model import get_naive_perf_model_moe, get_naive_perf_model, get_llm_viewer_model
from model_dict import MODEL_DICT
from hardware_dict import HARDWARE_DICT

import sys
import os


parser = argparse.ArgumentParser(description='Plot Speedup for LLM Models')

parser.add_argument('--max_bs', type=int, default=257, help='Maximum Batch Size')
parser.add_argument('--max_seq_len', type=int, default=2049, help='Maximum Sequence Length')
parser.add_argument('--model_name', type=str, default="Mixtral_8x7B", help='Model Name')
parser.add_argument('--hw1', type=str, default="nvidia_A100", help='First Hardware Name')
parser.add_argument('--hw2', type=str, default="nvidia_H20", help='Second Hardware Name')

args = parser.parse_args()

max_bs = args.max_bs
max_seq_len = args.max_seq_len
model_name = args.model_name
hw1_name = args.hw1
hw2_name = args.hw2

bs_values = np.arange(1, max_bs, 8)
seq_len_values = np.arange(128, max_seq_len, 64)

color_grid = np.zeros((len(bs_values), len(seq_len_values)))

hidden_size = MODEL_DICT[model_name]["h"]
intermediate_size = MODEL_DICT[model_name]["hi"]
gqa_grp = MODEL_DICT[model_name]["gqa_grp"]
type = MODEL_DICT[model_name]["type"]
model_id = MODEL_DICT[model_name]["model_id"]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

for ax, USE_LLM_VIEWER in zip(axs, [True, False]):
    color_grid = np.zeros((len(bs_values), len(seq_len_values)))

    is_error = False
    for i, bs in enumerate(bs_values):
        for j, seq_len in enumerate(seq_len_values):
            if USE_LLM_VIEWER:
                try:
                    perf1 = get_llm_viewer_model(model_id, hw1_name, bs, seq_len)
                    perf2 = get_llm_viewer_model(model_id, hw2_name, bs, seq_len)
                except:
                    is_error = True
                    continue
            else:
                flops1 = HARDWARE_DICT[hw1_name]["flops"]
                bwd1 = HARDWARE_DICT[hw1_name]["bwd"]
                flops2 = HARDWARE_DICT[hw2_name]["flops"]
                bwd2 = HARDWARE_DICT[hw2_name]["bwd"]

                if type == "moe":
                    perf1 = get_naive_perf_model_moe(flops1, bwd1, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp, 8, 2)
                    perf2 = get_naive_perf_model_moe(flops2, bwd2, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp, 8, 2)
                else:
                    perf1 = get_naive_perf_model(flops1, bwd1, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp)
                    perf2 = get_naive_perf_model(flops2, bwd2, bs, 1, seq_len, hidden_size, intermediate_size, gqa_grp)

            condition = perf1 / perf2
            color_grid[i, j] = condition

    if not is_error:
        cax = ax.pcolormesh(seq_len_values, bs_values, color_grid, shading='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax)

    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Batch Size')

    suffix = " LLM-Viewer" if USE_LLM_VIEWER else " Naive"
    ax.set_title(f"Speedup {hw1_name}/{hw2_name} ({model_name})" + suffix)

    ax.grid(True)

plt.tight_layout()
plt.show()