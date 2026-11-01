# Stand-in for vllm.transformers_utils.tokenizer.get_tokenizer so vLLM v0.6.6.post1's
# benchmark_serving.py runs without vLLM (its own fallback lacks the tokenizer_mode argument).
from transformers import AutoTokenizer


def get_tokenizer(name, tokenizer_mode="auto", trust_remote_code=False, **_):
    return AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code,
                                         use_fast=tokenizer_mode != "slow")
