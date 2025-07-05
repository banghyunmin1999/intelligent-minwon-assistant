from llama_cpp import Llama

llm = Llama(
    model_path="/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1  # GPU VRAM에 맞게 조정
)

output = llm("한국에서 상속받은 농지는 직접 농사를 짓지 않아도 되나요?", max_tokens=256)
print(output['choices'][0]['text'])
