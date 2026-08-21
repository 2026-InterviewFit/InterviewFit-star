INFERENCE_SYSTEM_PROMPT = (
    "면접 질문과 사용자의 답변을 바탕으로 답변을 STAR 구조로 분석하고, "
    "답변에서 확인되는 강점과 개선점을 분석한다. "
    "분석 결과는 지정된 JSON 형식으로 출력한다."
)


def create_inference_prompt(question: str, answer: str) -> str:
    return (
        f"### 질문:\n{question}\n\n"
        f"### 답변:\n{answer}"
    )