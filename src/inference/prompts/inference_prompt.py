INFERENCE_SYSTEM_PROMPT = (
    "사용자의 질문과 답변을 바탕으로 답변을 STAR 구조"
    "(Situation, Task, Action, Result)로 분석한다. "
    "Situation은 상황과 배경, Task는 지원자의 역할과 목표, "
    "Action은 지원자가 수행한 행동과 문제 해결 과정, "
    "Result는 행동의 결과와 성과를 분석한다. "
    "또한 답변에서 확인되는 구체적인 강점과 개선점을 분석한다. "
    "분석 결과는 지정된 JSON 형식으로 출력한다."
)


def create_inference_prompt(question: str, answer: str) -> str:
    return (
        f"### 질문:\n{question}\n\n"
        f"### 답변:\n{answer}"
    )