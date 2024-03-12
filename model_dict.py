MODEL_DICT = {
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
