import torch
import sys

def check_gpu_architecture():
    """
    현재 시스템에서 사용 가능한 NVIDIA GPU의 이름과 Compute Capability(아키텍처 버전)를 확인합니다.
    """
    print("="*50)
    print("GPU 아키텍처 버전 확인을 시작합니다...")
    print("="*50)

    if not torch.cuda.is_available():
        print("❌ CUDA를 사용할 수 없습니다. NVIDIA 드라이버 또는 PyTorch 설치를 확인해주세요.")
        sys.exit()

    try:
        device_count = torch.cuda.device_count()
        print(f"✅ {device_count}개의 NVIDIA GPU가 감지되었습니다.")
        
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            major, minor = torch.cuda.get_device_capability(i)
            arch_version = f"{major}{minor}"
            
            print(f"\n--- GPU {i} 정보 ---")
            print(f"  - 모델명: {gpu_name}")
            print(f"  - Compute Capability: {major}.{minor}")
            print(f"  - 아키텍처 버전 (CMAKE_ARGS용): {arch_version}")
            print("\n다음 단계에서는 이 '아키텍처 버전' 숫자를 사용하세요.")

    except Exception as e:
        print(f"❌ GPU 정보 확인 중 오류 발생: {e}")

if __name__ == "__main__":
    check_gpu_architecture()

