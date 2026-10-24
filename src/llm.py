"""LLM backends. Every prompt technique talks to the same tiny interface:

    llm.generate(messages, max_new_tokens=..., temperature=..., n=..., prefill=...) -> list[str]

* HFChatLLM  - any instruction-tuned causal LM on the Hugging Face Hub that ships a chat template
               (Qwen2.5-*-Instruct, Llama-3.x-Instruct, Phi-3, Gemma-it, ...).
* MockLLM    - scripted answers, used by the unit tests so the prompt logic can be checked offline.

Swap the model without touching any other code:
    HF_LLM=Qwen/Qwen2.5-0.5B-Instruct streamlit run app.py
"""
import os

DEFAULT_MODEL = os.environ.get("HF_LLM", "Qwen/Qwen2.5-1.5B-Instruct")


class LLM:
    name = "llm"

    def generate(self, messages, max_new_tokens=256, temperature=0.0, n=1, prefill=""):
        raise NotImplementedError


class HFChatLLM(LLM):
    """Local Hugging Face causal LM driven through its chat template."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_name
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # bf16/fp16 halves memory on GPU; keep fp32 on CPU for speed and compatibility
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except TypeError:  # older transformers used `torch_dtype`
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.to(self.device).eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, messages, max_new_tokens=256, temperature=0.0, n=1, prefill=""):
        """Return `n` completions. temperature=0 -> greedy (deterministic); >0 -> sampling.

        `prefill` is "response priming": text forced to be the start of the assistant turn
        (e.g. "{" to push the model straight into a JSON object). It is included in the returned text.
        """
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt += prefill
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        sample = temperature > 0
        kwargs = dict(max_new_tokens=max_new_tokens, do_sample=sample, pad_token_id=self.tokenizer.pad_token_id)
        if sample:
            kwargs.update(temperature=temperature, top_p=0.95, num_return_sequences=n)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_tokens = out[:, inputs["input_ids"].shape[1]:]
        texts = [prefill + t for t in self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]
        # greedy decoding yields a single sequence; replicate if the caller asked for n
        return texts if sample else texts * n


class MockLLM(LLM):
    """Returns scripted answers. `responder` is either a list (consumed in order, last one repeats)
    or a callable(messages, temperature, n, prefill) -> list[str]. Records every call in `.calls`."""

    name = "mock"

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self._i = 0

    def generate(self, messages, max_new_tokens=256, temperature=0.0, n=1, prefill=""):
        self.calls.append({"messages": messages, "temperature": temperature, "n": n, "prefill": prefill})
        if callable(self.responder):
            return self.responder(messages, temperature, n, prefill)
        idx = min(self._i, len(self.responder) - 1)
        self._i += 1
        item = self.responder[idx]
        outs = item if isinstance(item, list) else [item] * n
        return [prefill + o for o in outs]
