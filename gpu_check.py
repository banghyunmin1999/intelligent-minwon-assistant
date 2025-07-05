
import io
import sys
import re
from langchain_community.llms import LlamaCpp

MODEL_PATH = "/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
all_captured_logs = []
def llama_log_callback(level, text):
    all_captured_logs.append(text.strip())
def custom_print(*args, **kwargs):
    output = " ".join(map(str, args))
    all_captured_logs.append(output)
llm = None
try:
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_batch=2048,
        n_ctx=4096,
        max_tokens=1024,
        temperature=0.7,
        verbose=True,
        f_log_callback=llama_log_callback,
    )
    test_prompt = "대한민국의 수도는 어디인가요?"
    custom_print(f"테스트 프롬프트: \"{test_prompt}\"")
    response = llm.invoke(test_prompt)
    custom_print(f"모델 응답 (일부): {response[:200]}...")
except Exception as e:
    custom_print(f"[오류 발생]: {e}")
finally:
    pass
print("\n".join(all_captured_logs))
