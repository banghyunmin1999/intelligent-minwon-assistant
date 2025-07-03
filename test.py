import io
import sys
from langchain_community.llms import LlamaCpp

# 로그 캡처용
log_capture = io.StringIO()
sys_stdout = sys.stdout
sys.stdout = log_capture

try:
    llm = LlamaCpp(
        model_path="/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf",
        n_gpu_layers=40,
        n_batch=2048,
        n_ctx=4096,
        max_tokens=1024,
        temperature=0.7,
        verbose=True,
    )
finally:
    sys.stdout = sys_stdout

# 로그 내용 확인
init_log = log_capture.getvalue()
if "using CUDA for GPU acceleration" in init_log or "offloading" in init_log:
    print("✅ GPU에서 모델이 실행되고 있습니다.")
else:
    print("⚠️  GPU를 사용하지 않고 CPU에서 모델이 실행되고 있습니다.")
print(init_log)