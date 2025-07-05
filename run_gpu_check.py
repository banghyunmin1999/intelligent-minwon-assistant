import subprocess
import os
import re
import sys

# [필수 설정]
MODEL_PATH_IN_WSL = "/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-gguf-Q4_K_M/llama-3-Korean-Bllossom-8B-Q4_K_M.gguf"
PYTHON_INTERPRETER_PATH = "/home/bang/intelligent-minwon-assistant/home_venv/bin/python"
GPU_CHECK_SCRIPT_PATH = "/home/bang/intelligent-minwon-assistant/gpu_check.py"

# gpu_check.py 내용 (실행 시 모델 경로가 치환됨)
GPU_CHECK_FILE_CONTENT = """
import io
import sys
import re
from langchain_community.llms import LlamaCpp

MODEL_PATH = "REPLACE_WITH_ACTUAL_MODEL_PATH"
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
    custom_print(f"테스트 프롬프트: \\"{test_prompt}\\"")
    response = llm.invoke(test_prompt)
    custom_print(f"모델 응답 (일부): {response[:200]}...")
except Exception as e:
    custom_print(f"[오류 발생]: {e}")
finally:
    pass
print("\\n".join(all_captured_logs))
"""

# gpu_check.py 임시 생성 및 실행
temp_gpu_check_path = GPU_CHECK_SCRIPT_PATH
try:
    updated_gpu_check_content = GPU_CHECK_FILE_CONTENT.replace("REPLACE_WITH_ACTUAL_MODEL_PATH", MODEL_PATH_IN_WSL)
    with open(temp_gpu_check_path, "w", encoding='utf-8') as f:
        f.write(updated_gpu_check_content)

    full_captured_log_raw = ""
    try:
        result = subprocess.run(
            [PYTHON_INTERPRETER_PATH, temp_gpu_check_path],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            env=os.environ
        )
        full_captured_log_raw = result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        full_captured_log_raw = e.stdout + e.stderr
    except FileNotFoundError:
        print(f"[오류]: Python 인터프리터 또는 '{temp_gpu_check_path}' 파일을 찾을 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[알 수 없는 오류 발생]: {e}", file=sys.stderr)
        sys.exit(1)
finally:
    pass
    # 임시 파일 삭제 필요시 추가

# 로그 분석
full_captured_log_processed = re.sub(r'\s+', ' ', full_captured_log_raw.lower()).strip()

gpu_positive_keywords = [
    "found 1 cuda devices",
    "using device cuda",
    "assigned to device cuda",
    "offloading",
    "offloaded",
    "dev = cuda0",
    "cuda0 model buffer size",
    "cuda0 kv buffer size",
    "cuda compute buffer size",
    "cuda : archs =",
]
gpu_critical_error_keywords = [
    "cuda error",
    "rms_norm failed",
]
cpu_offload_warning_keywords = [
    "using cpu instead",
    "cannot be used with preferred buffer type cuda_host"
]

is_gpu_detected_and_allocated_log_present = any(keyword in full_captured_log_processed for keyword in gpu_positive_keywords)
is_critical_cuda_error_found = any(keyword in full_captured_log_processed for keyword in gpu_critical_error_keywords)
is_cpu_offload_warning_found = any(keyword in full_captured_log_processed for keyword in cpu_offload_warning_keywords)

prompt_tps_match = re.search(r"prompt eval time\s*=\s*\S+\s*ms\s*/\s*\S+\s*tokens\s*\(\s*(\d+\.\d+)\s*ms\s*per token,\s*(\d+\.\d+)\s*tokens per second\)", full_captured_log_processed, re.IGNORECASE | re.DOTALL)
eval_tps_match = re.search(r"eval time\s*=\s*\S+\s*ms\s*/\s*\S+\s*runs\s*\(\s*(\d+\.\d+)\s*ms\s*per token,\s*(\d+\.\d+)\s*tokens per second\)", full_captured_log_processed, re.IGNORECASE | re.DOTALL)

extracted_prompt_tps = float(prompt_tps_match.group(2)) if prompt_tps_match else 0.0
extracted_eval_tps = float(eval_tps_match.group(2)) if eval_tps_match else 0.0

has_sufficient_tps = (extracted_eval_tps > 20.0) or (extracted_prompt_tps > 20.0)

# ========== 1. 상세 로그 먼저 출력 ==========
print("="*60)
print("--- 상세 로그 (문제 진단 시 참고용) ---")
print("="*60)
print(full_captured_log_raw)
print("="*60)

# ========== 2. 진단 결과 요약 및 성능 정보는 맨 뒤에 출력 ==========
print("✨✨✨ GPU 사용 여부 최종 진단 결과 ✨✨✨")
print("="*60)

if is_critical_cuda_error_found:
    print("❌ **심각한 오류로 인해 GPU가 작동하지 않거나 모델이 실행되지 못했습니다.**")
    print("   -> 해결: `llama-cpp-python`이 CUDA 지원 빌드로 설치되었는지, NVIDIA 드라이버가 최신인지 확인.")
    print("   -> 전체 상세 로그를 확인하여 'CUDA error' 등의 메시지를 개발자에게 공유하세요.")
elif has_sufficient_tps and is_gpu_detected_and_allocated_log_present:
    if is_cpu_offload_warning_found:
        print("⚠️ **GPU 사용 확인 완료! 단, 일부 토큰 임베딩 레이어는 CPU에서 실행 중입니다.**")
        print(f"   -> 성능: 프롬프트 {extracted_prompt_tps:.2f} tokens/s, 생성 {extracted_eval_tps:.2f} tokens/s")
        print("   -> 이는 종종 정상적인 현상으로, 대부분의 연산은 GPU가 처리하여 성능 이점을 누리고 있습니다.")
    else:
        print("✅ **축하합니다! GPU가 완벽하게 작동 중입니다!**")
        print(f"   -> 성능: 프롬프트 {extracted_prompt_tps:.2f} tokens/s, 생성 {extracted_eval_tps:.2f} tokens/s")
        print("   -> 모델의 모든 주요 부분이 GPU에서 실행되고 있으며, 최적의 성능을 기대할 수 있습니다.")
    print("\n💡 이 속도는 GPU 가속이 활발히 이루어지고 있음을 강력하게 시사합니다.")
    print("   이제 모델을 빠르게 사용할 수 있습니다.")

elif is_gpu_detected_and_allocated_log_present:
    print("⚠️ **GPU는 감지되었으나, 성능 지표가 낮거나 확인되지 않아 GPU 가속이 충분히 활용되지 않을 수 있습니다.**")
    print(f"   -> 성능: 프롬프트 {extracted_prompt_tps:.2f} tokens/s, 생성 {extracted_eval_tps:.2f} tokens/s")
    print("   -> 전체 상세 로그를 확인하여 'using CPU instead' 메시지나 다른 병목 현상을 점검하세요.")
else:
    print("❌ **GPU가 감지되지 않았거나, 모델이 CPU에서만 실행되고 있습니다.**")
    print(f"   -> 성능: 프롬프트 {extracted_prompt_tps:.2f} tokens/s, 생성 {extracted_eval_tps:.2f} tokens/s")
    print("   -> 다음 사항들을 확인해주세요:")
    print("      1. Windows 호스트에 최신 NVIDIA GPU 드라이버가 설치되어 있나요?")
    print("      2. WSL(Ubuntu) 내부에 CUDA 툴킷이 제대로 설치되어 있나요? (`nvcc --version` 확인)")
    print("      3. `llama-cpp-python`이 CUDA 지원을 명시하여(`CMAKE_ARGS=\"-DGGML_CUDA=on\"` 등) 설치되었나요?")
    print("      4. 컴퓨터를 재부팅해 보셨나요?")

print("="*60)
print(f"[{__file__}] 테스트 완료.")
