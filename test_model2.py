from llama_cpp import Llama

# 1. 테스트할 Q2_K 모델의 정확한 로컬 경로를 지정합니다.
model_path = "/home/bang/intelligent-minwon-assistant/models/llama-3-Korean-Bllossom-8B-Q2_K-GGUF/llama-3-korean-bllossom-8b-q2_k.gguf"

# 2. GPU를 최대한 사용하도록 설정합니다. (Q2_K 모델은 가벼우므로 -1로 시도)
n_gpu_layers = -1

print(">> 모델 로드를 시작합니다...")
print(f">> 모델 경로: {model_path}")
print(f">> GPU 레이어: {n_gpu_layers}")

try:
    # 3. Llama 객체를 생성하여 모델을 로드합니다.
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        n_ctx=2048, # 컨텍스트 크기
        verbose=True # 로딩 과정을 자세히 보기 위해 True로 설정
    )
    print("\n✅ 모델 로드 성공!")

    # 4. 간단한 프롬프트로 추론(답변 생성) 테스트를 진행합니다.
    prompt = "대한민국에서 가장 높은 산은?"
    print(f"\n>> 프롬프트: \"{prompt}\"")
    print(">> 답변 생성을 시작합니다...")

    output = llm(
        prompt,
        max_tokens=50, # 짧은 답변만 확인
        stop=["\n"],   # 줄바꿈 시 생성 중단
        echo=False     # 프롬프트를 다시 출력하지 않음
    )

    print("\n✅ 답변 생성 성공!")
    
    # 5. 생성된 답변을 출력합니다.
    answer = output['choices'][0]['text'].strip()
    print(f"\n>> 생성된 답변: {answer}")

except Exception as e:
    import traceback
    print(f"\n❌ 테스트 중 오류가 발생했습니다:")
    traceback.print_exc()