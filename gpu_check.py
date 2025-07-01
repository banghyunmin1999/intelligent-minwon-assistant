import sys

print("--- GPU 환경 점검 시작 ---")

try:
    # 1. PyTorch를 이용한 CUDA 가용성 체크
    import torch
    print("\n[1/2] PyTorch GPU 확인 중...")
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  - 상태: 성공")
        print(f"  - PyTorch가 {gpu_count}개의 GPU를 감지했습니다.")
        print(f"  - GPU 모델: {gpu_name}")
        pytorch_ok = True
    else:
        print("  - 상태: 실패")
        print("  - PyTorch가 CUDA를 사용할 수 있는 GPU를 찾지 못했습니다.")
        pytorch_ok = False

except ImportError:
    print("\n[1/2] PyTorch GPU 확인 중...")
    print("  - 상태: 실패")
    print("  - PyTorch 라이브러리가 설치되어 있지 않습니다. (pip install torch)")
    pytorch_ok = False
    
try:
    # 2. FAISS를 이용한 GPU 가용성 체크
    import faiss
    print("\n[2/2] FAISS GPU 확인 중...")
    gpu_count_faiss = faiss.get_num_gpus()
    if gpu_count_faiss > 0:
        print(f"  - 상태: 성공")
        print(f"  - FAISS가 {gpu_count_faiss}개의 GPU를 감지했습니다.")
        faiss_ok = True
    else:
        print("  - 상태: 실패")
        print("  - FAISS가 GPU를 찾지 못했습니다. faiss-cpu 버전이 설치되었을 수 있습니다.")
        faiss_ok = False

except ImportError:
    print("\n[2/2] FAISS GPU 확인 중...")
    print("  - 상태: 실패")
    print("  - FAISS 라이브러리가 설치되어 있지 않습니다. (conda install -c pytorch faiss-gpu)")
    faiss_ok = False


# 최종 결과
print("\n--- 최종 점검 결과 ---")
if pytorch_ok and faiss_ok:
    print("✅ 완벽합니다! GPU 환경이 모든 AI 작업을 수행할 준비가 되었습니다.")
else:
    print("❌ 설정 오류! 위의 실패 메시지를 확인하여 문제를 해결해야 합니다.")
    sys.exit(1) # 오류가 있으면 프로그램 종료