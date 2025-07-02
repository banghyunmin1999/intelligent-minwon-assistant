import psutil
import torch
import platform
import os
import sys
from datetime import datetime
import json

print("=== 시스템 리소스 체크 ===")
print(f"실행 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 시스템 정보
print("\n=== 시스템 정보 ===")
print(f"운영체제: {platform.system()} {platform.release()}")
print(f"프로세서: {platform.processor()}")
print(f"Python 버전: {sys.version}")
print(f"Python 경로: {sys.executable}")

# 메모리 정보
print("\n=== 메모리 정보 ===")
mem = psutil.virtual_memory()
print(f"총 메모리: {mem.total / (1024**3):.2f}GB")
print(f"사용 중 메모리: {mem.used / (1024**3):.2f}GB")
print(f"메모리 사용률: {mem.percent}%")

# 디스크 정보
print("\n=== 디스크 정보 ===")
disk = psutil.disk_usage('/')
print(f"총 디스크: {disk.total / (1024**3):.2f}GB")
print(f"사용 중 디스크: {disk.used / (1024**3):.2f}GB")
print(f"디스크 사용률: {disk.percent}%")

# CPU 정보
print("\n=== CPU 정보 ===")
print(f"CPU 코어 수: {psutil.cpu_count(logical=True)}")
print(f"CPU 사용률: {psutil.cpu_percent(interval=1)}%")

# GPU 정보
print("\n=== GPU 정보 ===")
if torch.cuda.is_available():
    print("GPU 사용 가능")
    print(f"GPU 개수: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        # PyTorch의 메모리 정보
        print(f"GPU {i} PyTorch 메모리: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f}GB")
        
        # nvidia-smi 명령어로도 메모리 확인
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"GPU {i} nvidia-smi 메모리: {float(result.stdout.strip()) / 1024:.2f}GB")
            else:
                print("nvidia-smi 명령어 실행 실패")
        except:
            print("nvidia-smi 명령어 실행 실패")
else:
    print("GPU 사용 불가능")

# 환경 변수
print("\n=== 환경 변수 ===")
print("CUDA_VISIBLE_DEVICES:", os.environ.get('CUDA_VISIBLE_DEVICES', '없음'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', '없음'))

# 네트워크
print("\n=== 네트워크 정보 ===")
if psutil.net_if_stats():
    print("활성화된 네트워크 인터페이스:")
    for interface, addrs in psutil.net_if_addrs().items():
        print(f"  {interface}:")
        for addr in addrs:
            print(f"    {addr.family}: {addr.address}")
else:
    print("네트워크 인터페이스가 없습니다.")
