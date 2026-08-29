import json

from openai import OpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from src.evaluation.prompts.judge_prompt import JUDGE_SYSTEM_PROMPT, create_judge_prompt
from src.schema.schema import EvaluationScore
from src.config.config import settings


# client = OpenAI(
#     api_key=settings.UPSTAGE_API_KEY,
#     base_url="https://api.upstage.ai/v1",
# )
#
#
# def judge(ground_truth: dict, prediction: dict):
#     user_prompt = create_judge_prompt(ground_truth, prediction)
#
#     response = client.chat.completions.create(
#         model=settings.UPSTAGE_MODEL,
#         temperature=0,
#         response_format={
#             "type": "json_object"
#         },
#         messages=[
#             {
#                 "role": "system",
#                 "content": JUDGE_SYSTEM_PROMPT,
#             },
#             {
#                 "role": "user",
#                 "content": user_prompt,
#             },
#         ],
#     )
#
#     content = response.choices[0].message.content
#
#     result = json.loads(content)
#
#     return EvaluationScore(
#         **result
#     ).model_dump()


client = OpenAI(api_key=settings.OPENAI_API_KEY)


def judge(question: str, answer: str, ground_truth: dict, prediction: dict):
    user_prompt = create_judge_prompt(question, answer, ground_truth, prediction)

    response = client.chat.completions.parse(
        model=settings.OPENAI_MODEL,
        temperature=0,
        response_format=EvaluationScore,
        messages=[
            ChatCompletionSystemMessageParam(
                role="system",
                content=JUDGE_SYSTEM_PROMPT,
            ),
            ChatCompletionUserMessageParam(
                role="user",
                content=user_prompt,
            ),
        ],
    )

    return response.choices[0].message.parsed.model_dump()


# if __name__ == "__main__":
#     gt = {
#         "star": {
#             "situation": "프로젝트 일정 지연 문제가 발생함",
#             "task": "팀원들과 문제를 해결해야 함",
#             "action": "업무를 재분배하고 일정 관리를 진행함",
#             "result": "프로젝트를 기한 내 완료함",
#         },
#         "strengths": [
#             "협업 경험이 잘 드러남"
#         ],
#         "improvements": [
#             "정량적인 성과 추가가 필요함"
#         ],
#     }
#
#     pred = {
#         "star": {
#             "situation": "프로젝트 진행 중 일정 문제가 발생함",
#             "task": "팀원들과 해결 방안을 찾아야 함",
#             "action": "역할을 조정하고 협업함",
#             "result": "프로젝트를 성공적으로 마무리함",
#         },
#         "strengths": [
#             "협업 능력이 나타남"
#         ],
#         "improvements": [
#             "구체적인 수치 제시가 필요함"
#         ],
#     }
#
#     print(judge(gt, pred))
