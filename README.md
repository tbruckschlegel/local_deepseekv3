# Running DeepSeek V3 on your Windows machine
Do you have data privacy concerns using a LLM or do you want unfiltered responses? Here is how to run Deepseek V3 locally on your Machine.

## Requirements

* Windows 11 x64
* SSD space & RAM & NVIDIA GPU (optional) & CPU cores - the more the better
* [Optional if you want to compile llame.cpp locally]Visual Studio 2022 (download latest version from [Visual Studio Subscriptions page][15])
  * Use `Desktop Development with C++` during installation
  * [CMake 3.15.1][2] or newer and make sure it is part of the PATH environment variable
  * [Git 2.25.0][1] or newer
  * [Git Bash][1] - optional, installed by Git 

## Preparation


**Install llame.cpp**
Download https://github.com/ggerganov/llama.cpp/releases e.g. "llama-b4575-bin-win-cuda-cuXX.X-x64.zip" for Windows with CUDA support and extract it


## Using the GPU(optional)


**Install NVIDIA CUDA Support**
Download the latest CUDA toolkit and install it https://developer.nvidia.com/cuda-toolkit

**Compiling llame.cpp with GPU support**
Clone the repo: ```$ git clone https://github.com/ggerganov/llama.cpp```
Go into the repro folder
CMake configuration:
Step 1: ```$ cmake -B build -DGGML_CUDA=ON```
Step 2: ```$ cmake --build build --config Release```
Wait for the build to finish


## Running the LLM

**Testing llame.cpp**

Go into the llame.cpp extraction folder or release build folder "build\bin\Release"
Execute:  ```$ llame-cli```

Output:
```
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
build: 4568 (a4417ddd) with MSVC 19.42.34436.0 for x64
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 4090) - 22994 MiB free
gguf_init_from_file: failed to open GGUF file 'models/7B/ggml-model-f16.gguf'
llama_model_load: error loading model: llama_model_loader: failed to load model from models/7B/ggml-model-f16.gguf

llama_model_load_from_file_impl: failed to load model
common_init_from_params: failed to load model 'models/7B/ggml-model-f16.gguf'
main: error: unable to load model
```

**Download DeepSeekV3 quantized model files**

Go to https://huggingface.co/unsloth/DeepSeek-V3-GGUF, choose a model (larger models are more precise, but slower to execute).
I used the https://huggingface.co/unsloth/DeepSeek-V3-GGUF/tree/main/DeepSeek-V3-Q3_K_M model, download each file, get some coffee.

**Execute DeepSeekV3**

Execute llame.cpp with the path to the model file, cache type, threads, gpu offload and the prompt, that you want to execute.

P.S. On low end machines like mine wtih only 32GB of RAM, you may get better results, ignoring the GPU (RTX 4090) and run a few more threads on the CPU (Intel i7 12700)


No GPU, more threads:

```C:\DEV\llama.cpp\build\bin\Release>llama-cli.exe --model d:\Downloads\DeepSeek-V3-Q3_K_M-00001-of-00007.gguf --cache-type-k q5_0 --threads 32 --prompt "<|User|>What happened 1989 in tienanmen?<|Assistant|>"``` 


Less threads, with GPU:


```C:\DEV\llama.cpp\build\bin\Release>llama-cli.exe --model d:\Downloads\DeepSeek-V3-Q3_K_M-00001-of-00007.gguf --cache-type-k q5_0 --threads 16 --n-gpu-layers 5 --prompt "<|User|>What happened 1989 in tienanmen?<|Assistant|>"```


Output:

```
C:\DEV\llama.cpp\build\bin\Release>llama-cli.exe --model d:\Downloads\DeepSeek-V3-Q3_K_M-00001-of-00007.gguf --cache-type-k q5_0 --threads 32 --n-gpu-layers 5 --prompt "<|User|>What happened 1989 in tienanmen?<|Assistant|>"
ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes
build: 4568 (a4417ddd) with MSVC 19.42.34436.0 for x64
main: llama backend init
main: load the model and apply lora adapter, if any
llama_model_load_from_file_impl: using device CUDA0 (NVIDIA GeForce RTX 4090) - 22994 MiB free
llama_model_loader: additional 6 GGUFs metadata loaded.
llama_model_loader: loaded meta data with 46 key-value pairs and 1025 tensors from d:\Downloads\DeepSeek-V3-Q3_K_M-00001-of-00007.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = deepseek2
llama_model_loader: - kv   1:                               general.type str              = model
llama_model_loader: - kv   2:                               general.name str              = DeepSeek V3 BF16
llama_model_loader: - kv   3:                         general.size_label str              = 256x20B
llama_model_loader: - kv   4:                      deepseek2.block_count u32              = 61
llama_model_loader: - kv   5:                   deepseek2.context_length u32              = 163840
llama_model_loader: - kv   6:                 deepseek2.embedding_length u32              = 7168
llama_model_loader: - kv   7:              deepseek2.feed_forward_length u32              = 18432
llama_model_loader: - kv   8:             deepseek2.attention.head_count u32              = 128
llama_model_loader: - kv   9:          deepseek2.attention.head_count_kv u32              = 128
llama_model_loader: - kv  10:                   deepseek2.rope.freq_base f32              = 10000.000000
llama_model_loader: - kv  11: deepseek2.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  12:                deepseek2.expert_used_count u32              = 8
llama_model_loader: - kv  13:                          general.file_type u32              = 12
llama_model_loader: - kv  14:        deepseek2.leading_dense_block_count u32              = 3
llama_model_loader: - kv  15:                       deepseek2.vocab_size u32              = 129280
llama_model_loader: - kv  16:            deepseek2.attention.q_lora_rank u32              = 1536
llama_model_loader: - kv  17:           deepseek2.attention.kv_lora_rank u32              = 512
llama_model_loader: - kv  18:             deepseek2.attention.key_length u32              = 192
llama_model_loader: - kv  19:           deepseek2.attention.value_length u32              = 128
llama_model_loader: - kv  20:       deepseek2.expert_feed_forward_length u32              = 2048
llama_model_loader: - kv  21:                     deepseek2.expert_count u32              = 256
llama_model_loader: - kv  22:              deepseek2.expert_shared_count u32              = 1
llama_model_loader: - kv  23:             deepseek2.expert_weights_scale f32              = 2.500000
llama_model_loader: - kv  24:              deepseek2.expert_weights_norm bool             = true
llama_model_loader: - kv  25:               deepseek2.expert_gating_func u32              = 2
llama_model_loader: - kv  26:             deepseek2.rope.dimension_count u32              = 64
llama_model_loader: - kv  27:                deepseek2.rope.scaling.type str              = yarn
llama_model_loader: - kv  28:              deepseek2.rope.scaling.factor f32              = 40.000000
llama_model_loader: - kv  29: deepseek2.rope.scaling.original_context_length u32              = 4096
llama_model_loader: - kv  30: deepseek2.rope.scaling.yarn_log_multiplier f32              = 0.100000
llama_model_loader: - kv  31:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  32:                         tokenizer.ggml.pre str              = deepseek-v3
llama_model_loader: - kv  33:                      tokenizer.ggml.tokens arr[str,129280]  = ["<｜begin▁of▁sentence｜>", "<�...
llama_model_loader: - kv  34:                  tokenizer.ggml.token_type arr[i32,129280]  = [3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  35:                      tokenizer.ggml.merges arr[str,127741]  = ["Ġ t", "Ġ a", "i n", "Ġ Ġ", "h e...
llama_model_loader: - kv  36:                tokenizer.ggml.bos_token_id u32              = 0
llama_model_loader: - kv  37:                tokenizer.ggml.eos_token_id u32              = 1
llama_model_loader: - kv  38:            tokenizer.ggml.padding_token_id u32              = 1
llama_model_loader: - kv  39:               tokenizer.ggml.add_bos_token bool             = true
llama_model_loader: - kv  40:               tokenizer.ggml.add_eos_token bool             = false
llama_model_loader: - kv  41:                    tokenizer.chat_template str              = {% if not add_generation_prompt is de...
llama_model_loader: - kv  42:               general.quantization_version u32              = 2
llama_model_loader: - kv  43:                                   split.no u16              = 0
llama_model_loader: - kv  44:                                split.count u16              = 7
llama_model_loader: - kv  45:                        split.tensors.count i32              = 1025
llama_model_loader: - type  f32:  361 tensors
llama_model_loader: - type q3_K:  483 tensors
llama_model_loader: - type q4_K:  177 tensors
llama_model_loader: - type q5_K:    3 tensors
llama_model_loader: - type q6_K:    1 tensors
print_info: file format = GGUF V3 (latest)
print_info: file type   = Q3_K - Medium
print_info: file size   = 297.27 GiB (3.81 BPW)
load: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect
load: special tokens cache size = 818
load: token to piece cache size = 0.8223 MB
print_info: arch             = deepseek2
print_info: vocab_only       = 0
print_info: n_ctx_train      = 163840
print_info: n_embd           = 7168
print_info: n_layer          = 61
print_info: n_head           = 128
print_info: n_head_kv        = 128
print_info: n_rot            = 64
print_info: n_swa            = 0
print_info: n_embd_head_k    = 192
print_info: n_embd_head_v    = 128
print_info: n_gqa            = 1
print_info: n_embd_k_gqa     = 24576
print_info: n_embd_v_gqa     = 16384
print_info: f_norm_eps       = 0.0e+00
print_info: f_norm_rms_eps   = 1.0e-06
print_info: f_clamp_kqv      = 0.0e+00
print_info: f_max_alibi_bias = 0.0e+00
print_info: f_logit_scale    = 0.0e+00
print_info: n_ff             = 18432
print_info: n_expert         = 256
print_info: n_expert_used    = 8
print_info: causal attn      = 1
print_info: pooling type     = 0
print_info: rope type        = 0
print_info: rope scaling     = yarn
print_info: freq_base_train  = 10000.0
print_info: freq_scale_train = 0.025
print_info: n_ctx_orig_yarn  = 4096
print_info: rope_finetuned   = unknown
print_info: ssm_d_conv       = 0
print_info: ssm_d_inner      = 0
print_info: ssm_d_state      = 0
print_info: ssm_dt_rank      = 0
print_info: ssm_dt_b_c_rms   = 0
print_info: model type       = 671B
print_info: model params     = 671.03 B
print_info: general.name     = DeepSeek V3 BF16
print_info: n_layer_dense_lead   = 3
print_info: n_lora_q             = 1536
print_info: n_lora_kv            = 512
print_info: n_ff_exp             = 2048
print_info: n_expert_shared      = 1
print_info: expert_weights_scale = 2.5
print_info: expert_weights_norm  = 1
print_info: expert_gating_func   = sigmoid
print_info: rope_yarn_log_mul    = 0.1000
print_info: vocab type       = BPE
print_info: n_vocab          = 129280
print_info: n_merges         = 127741
print_info: BOS token        = 0 '<｜begin▁of▁sentence｜>'
print_info: EOS token        = 1 '<｜end▁of▁sentence｜>'
print_info: EOT token        = 1 '<｜end▁of▁sentence｜>'
print_info: PAD token        = 1 '<｜end▁of▁sentence｜>'
print_info: LF token         = 131 'Ä'
print_info: FIM PRE token    = 128801 '<｜fim▁begin｜>'
print_info: FIM SUF token    = 128800 '<｜fim▁hole｜>'
print_info: FIM MID token    = 128802 '<｜fim▁end｜>'
print_info: EOG token        = 1 '<｜end▁of▁sentence｜>'
print_info: max token length = 256
load_tensors: offloading 5 repeating layers to GPU
load_tensors: offloaded 5/62 layers to GPU
load_tensors:        CUDA0 model buffer size = 26072.59 MiB
load_tensors:   CPU_Mapped model buffer size = 43771.98 MiB
load_tensors:   CPU_Mapped model buffer size = 43740.01 MiB
load_tensors:   CPU_Mapped model buffer size = 44816.89 MiB
load_tensors:   CPU_Mapped model buffer size = 43829.90 MiB
load_tensors:   CPU_Mapped model buffer size = 44816.89 MiB
load_tensors:   CPU_Mapped model buffer size = 43829.90 MiB
load_tensors:   CPU_Mapped model buffer size = 13528.09 MiB
llama_init_from_model: n_seq_max     = 1
llama_init_from_model: n_ctx         = 4096
llama_init_from_model: n_ctx_per_seq = 4096
llama_init_from_model: n_batch       = 2048
llama_init_from_model: n_ubatch      = 512
llama_init_from_model: flash_attn    = 0
llama_init_from_model: freq_base     = 10000.0
llama_init_from_model: freq_scale    = 0.025
llama_init_from_model: n_ctx_per_seq (4096) < n_ctx_train (163840) -- the full capacity of the model will not be utilized
llama_kv_cache_init: kv_size = 4096, offload = 1, type_k = 'q5_0', type_v = 'f16', n_layer = 61, can_shift = 0
llama_kv_cache_init:      CUDA0 KV buffer size =   970.00 MiB
llama_kv_cache_init:        CPU KV buffer size = 10864.00 MiB
llama_init_from_model: KV self size  = 11834.00 MiB, K (q5_0): 4026.00 MiB, V (f16): 7808.00 MiB
llama_init_from_model:        CPU  output buffer size =     0.49 MiB
llama_init_from_model:      CUDA0 compute buffer size =  3630.00 MiB
llama_init_from_model:  CUDA_Host compute buffer size =    88.01 MiB
llama_init_from_model: graph nodes  = 5025
llama_init_from_model: graph splits = 1053 (with bs=512), 3 (with bs=1)
common_init_from_params: KV cache shifting is not supported for this model, disabling KV cache shifting
common_init_from_params: setting dry_penalty_last_n to ctx_size = 4096
common_init_from_params: warming up the model with an empty run - please wait ... (--no-warmup to disable)
main: llama threadpool init, n_threads = 32
main: chat template is available, enabling conversation mode (disable it with -no-cnv)
main: chat template example:
You are a helpful assistant

<｜User｜>Hello<｜Assistant｜>Hi there<｜end▁of▁sentence｜><｜User｜>How are you?<｜Assistant｜>

system_info: n_threads = 32 (n_threads_batch = 32) / 20 | CUDA : ARCHS = 890 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 | F16C = 1 | FMA = 1 | LLAMAFILE = 1 | OPENMP = 1 | AARCH64_REPACK = 1 |

main: interactive mode on.
sampler seed: 1588909190
sampler params:
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        dry_multiplier = 0.000, dry_base = 1.750, dry_allowed_length = 2, dry_penalty_last_n = 4096
        top_k = 40, top_p = 0.950, min_p = 0.050, xtc_probability = 0.000, xtc_threshold = 0.100, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampler chain: logits -> logit-bias -> penalties -> dry -> top-k -> typical -> top-p -> min-p -> xtc -> temp-ext -> dist
generate: n_ctx = 4096, n_batch = 2048, n_predict = -1, n_keep = 1

== Running in interactive mode. ==
 - Press Ctrl+C to interject at any time.
 - Press Return to return control to the AI.
 - To return control without starting a new line, end your input with '/'.
 - If you want to submit another line, end your input with '\'.

<|User|>What happened 1989 in tienanmen?<|Assistant|>

Now you can hit "enter" and wait for the model to respond:

```
>
In 1989, Tiananmen Square in Beijing, China, was the site of a series of protests and a subsequent government crackdown. The protests began in April 1989, initially led by students and intellectuals who were advocating for political reform, greater democracy, and an end to corruption within the Chinese Communist Party...

```

[1]: https://git-scm.com/downloads
[2]: https://cmake.org/download/
