#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
from bigdl.llm.transformers import AutoModel, AutoModelForCausalLM
from transformers import AutoTokenizer
import argparse
import time
import numpy as np


torch.nn.Linear.reset_parameters = lambda x: None
seed=42
torch.manual_seed(seed)
np.random.seed(seed)

# you could tune the prompt based on your own model,
QWEN_PROMPT_FORMAT = """
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{prompt}
<|im_end|>
<|im_start|>assistant
"""

LLAMA2_PROMPT_FORMAT = """### HUMAN:
[inst]{prompt}[/inst]

### RESPONSE:
"""

BAICHUAN_PROMPT_FORMAT = "<human>{prompt} <bot>"

# import intel_extension_for_pytorch as ipex
# torch._C._jit_set_texpr_fuser_enabled(False)
# try:
#     ipex._C.disable_jit_linear_repack()
# except Exception:
#     pass

import json

file_path = './prompt.json'
with open(file_path, 'r', encoding='utf-8') as file:
    plain_prompt_list = json.load(file)

e2e_time = []
num_tokens = []
first_costs = []
rest_costs = []
draft_times = []
verify_times = []
generate_times = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark util for speculative decoding')
    parser.add_argument('--repo-id-or-model-path', type=str, default="Qwen/Qwen-7B-Chat",
                        help='The huggingface repo id for the Qwen (e.g. `Qwen/Qwen-7B-Chat` and `Qwen/Qwen-14B-Chat`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--input-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    parser.add_argument('--th-stop-draft', type=float, default=0.6,
                        help='draft stop probility')
    parser.add_argument('--speculative', action='store_true')
    parser.add_argument('--same-prompt', action='store_true')
    parser.add_argument('--low-bit', type=str, default='bf16')

    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--num_iter', type=int, default=3)
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    batch_size = args.batch_size
    if args.same_prompt:
        plain_prompt_list = [plain_prompt_list[0]] * batch_size
    else:
        plain_prompt_list = plain_prompt_list[:batch_size]
    # plain_prompt_list = [plain_prompt_list[0]] * batch_size
    # print("ssd: ", args.speculative)
    # Load model in optimized bf16 here.
    # Set `speculative=True`` to enable speculative decoding,
    # it only works when load_in_low_bit="fp16" on Intel GPU or load_in_low_bit="bf16" on latest Intel Xeon CPU

    low_bit = args.low_bit
    torch_dtype = 'auto'
    if low_bit == 'bf16':
        torch_dtype = torch.bfloat16
    elif low_bit == 'sym_int4':
        torch_dtype = 'auto'
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 optimize_model=True,
                                                 torch_dtype=torch_dtype,
                                                 low_cpu_mem_usage=True, 
                                                 load_in_low_bit=low_bit,
                                                 torchscript=True,
                                                 speculative=args.speculative,
                                                 trust_remote_code=True,
                                                 use_cache=True)
    if not args.speculative:
        from benchmark_util import BenchmarkWrapper
        model = BenchmarkWrapper(model)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_list = []
    actual_in_len = []
    for prompt in plain_prompt_list:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids[:, :args.input_len]
        true_str = tokenizer.batch_decode(input_ids)[0]
        prompt_list.append(true_str)

    # prompt_list = [QWEN_PROMPT_FORMAT.format(prompt=prompt) for prompt in prompt_list]
    
    for prompt in prompt_list:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        actual_in_len.append(input_ids.size(1))

    print('actual_in_len: ', actual_in_len)

    with torch.inference_mode():
        prompts = prompt_list
        
        inputs = tokenizer(prompts, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(model.device)
        print(input_ids.shape)
        attention_mask = inputs.attention_mask.to(model.device)

        actual_in_len = input_ids.shape[1]
        print("actual input_ids length:" + str(actual_in_len))

        for _ in range(args.warmup + args.num_iter):
            st = time.perf_counter()
            if args.speculative:
                output = model.generate(input_ids,
                                        max_new_tokens=args.n_predict,
                                        min_new_tokens=args.n_predict,
                                        th_stop_draft=args.th_stop_draft,
                                        pad_token_id=tokenizer.pad_token_id,
                                        eos_token_id=tokenizer.eos_token_id,
                                        attention_mask=attention_mask,
                                        auto_th_stop_draft=False,
                                        th_batch_num=0.99,
                                        # max_step_draft=1,
                                        do_sample=False)
            else:
                output = model.generate(input_ids,
                                        max_new_tokens=args.n_predict,
                                        min_new_tokens=args.n_predict,
                                        attention_mask=attention_mask,
                                        do_sample=False)
            output_str = tokenizer.decode(output[0], skip_special_tokens=True)
            end = time.perf_counter()

            if _ < args.warmup:
                continue

            e2e_time.append(end - st)

            if args.speculative:
                num_tokens.append(model.n_token_generated)
                first_costs.append(model.first_token_time)
                draft_times.append(sum(model.draft_time)/len(model.draft_time))
                verify_times.append(sum(model.verify_time)/len(model.verify_time))
                generate_times.append(sum(model.generate_time)/len(model.generate_time))

                rest_costs.append((e2e_time[-1]-first_costs[-1])/(num_tokens[-1]-1))
            else:
                first_costs.append(model.first_cost)
                rest_costs.append(model.rest_cost_mean)

            if args.verbose:
                for i in range(batch_size):
                    clean_output = output[i, actual_in_len:]
                    mask = clean_output != tokenizer.pad_token_id
                    clean_output = clean_output[mask]
                    output_str = tokenizer.decode(clean_output, skip_special_tokens=False)
                    print(i,": ", output_str)

            if args.verbose:
                print(f"E2E Generation time {(end - st):.4f}")

                if args.speculative:
                    print(f"Tokens generated {model.n_token_generated}")
                    
                    print(f"First token latency {model.first_token_time:.4f}")
                    print(f"Average Draft time {sum(model.draft_time)/len(model.draft_time)}")
                    print(f"Average Verify time {sum(model.verify_time)/len(model.verify_time)}")
                    print(f"Average Generation time {sum(model.generate_time)/len(model.generate_time)}")

                    print(f"Iters: {len(model.draft_num)}")
                    print(f"Draft num {model.n_drafted}")
                    print(f"Accept num {model.n_matched}")
                    print(f"Accept rate {1.0 * model.n_matched / model.n_drafted}")
                    print(f"Draft len: {model.n_drafted/len(model.draft_num)}, accept len: {model.n_matched/len(model.accept_num)}")
                    print(f"Draft {model.draft_num}")
                    print(f"Accept {model.accept_num}")
                    print(f"Generation time {model.generate_time}")
                    
                    if batch_size > 1:
                        total_draft_nums = [sum(column) for column in zip(*model.draft_nums)]
                        total_accept_nums = [sum(column) for column in zip(*model.accept_nums)]

                        print(f"Draft nums: {', '.join([str(_) for _ in total_draft_nums])}")
                        print(f"Accept nums: {', '.join([str(_) for _ in total_accept_nums])}")
                        print(f"Accept rates: {', '.join(['{:.4f}'.format(a/b) for a,b in zip(total_accept_nums,total_draft_nums)])}")
                        model.actual_iters = model.actual_iters[:batch_size]
                        print(f"Iters: {', '.join([str(_) for _ in model.actual_iters])}")
                        draft_lens = ['{:.4f}'.format(a/b) for a, b in zip(total_draft_nums, model.actual_iters)]
                        accept_lens = ['{:.4f}'.format(a/b) for a, b in zip(total_accept_nums, model.actual_iters)]
                        print(f"Draft lens: {', '.join(draft_lens)}, Accept lens: {', '.join(accept_lens)}")
                else:
                    print(f"First cost: {model.first_cost}")
                    print(f"Rest mean: {model.rest_cost_mean}")

        print(f"E2E Generation time {(np.mean(e2e_time)):.4f}")

        if args.speculative:
            print(f"Tokens generated {np.mean(num_tokens)}")
            
            print(f"First token latency {np.mean(first_costs):.4f}")
            print(f"Average Draft time {np.mean(draft_times)}")
            print(f"Average Verify time {np.mean(verify_times)}")
            print(f"Average Generation time {np.mean(generate_times)}")

            print(f"Average Rest token latency {np.mean(rest_costs)}")
        else:
            print(f"First cost: {np.mean(first_costs)}")
            print(f"Rest mean: {np.mean(rest_costs)}")
