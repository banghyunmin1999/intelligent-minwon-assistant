import io
import sys
import re
from langchain_community.llms import LlamaCpp

log_capture = io.StringIO()
sys_stdout = sys.stdout
sys.stdout = log_capture

try:
    llm = LlamaCpp(
        model_path="/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        n_gpu_layers=-1,
        n_batch=2048,
        n_ctx=4096,
        max_tokens=1024,
        temperature=0.7,
        verbose=True,
    )
finally:
    sys.stdout = sys_stdout

init_log = log_capture.getvalue().lower()  # 소문자로 통일

gpu_keywords = [
    "using device cuda",
    "assigned to device cuda",
    "offloading",
    "offloaded",
    "dev = cuda0",
    "cuda0 model buffer size",
    "cuda0 kv buffer size",
]

if any(keyword in init_log for keyword in gpu_keywords):
    print("✅ GPU에서 모델이 실행되고 있습니다.")
else:
    print("⚠️  GPU를 사용하지 않고 CPU에서 모델이 실행되고 있습니다.")

# (필요하면 로그도 출력)
#print(init_log)
