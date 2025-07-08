#!/bin/bash

echo "AI Server 포트 포워딩 시작 (http://127.0.0.1:8000)"
kubectl port-forward service/ai-server-service 8000:8000 &
AI_PID=$! # 백그라운드 프로세스 ID 저장

echo "Egov Server 포트 포워딩 시작 (http://127.0.0.1:8080)"
kubectl port-forward service/egov-server-service 8080:8080 &
EGOV_PID=$! # 백그라운드 프로세스 ID 저장

echo "---------------------------------------------------------"
echo "서버 포트 포워딩이 백그라운드에서 실행 중입니다."
echo "AI Server: http://127.0.0.1:8000"
echo "Egov Server: http://127.0.0.1:8080"
echo "이 터미널을 닫으면 포트 포워딩이 중단됩니다."
echo "종료하려면 이 터미널에서 Ctrl+C를 누르세요."
echo "---------------------------------------------------------"

# 백그라운드 프로세스가 종료될 때까지 스크립트가 대기하도록 합니다.
# (Ctrl+C를 누르면 이 스크립트와 백그라운드 프로세스들이 함께 종료됩니다.)
wait $AI_PID $EGOV_PID