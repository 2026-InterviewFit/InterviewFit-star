import time
from openai import OpenAI, OpenAIError

from src.generation.prompts.system import SYSTEM_PROMPT
from src.config.config import settings


client = OpenAI(
    api_key=settings.UPSTAGE_API_KEY,
    base_url="https://api.upstage.ai/v1",
    timeout=60
)


def generate(prompt, retry=3):
    for i in range(retry):
        try:
            response = client.chat.completions.create(
                model=settings.UPSTAGE_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=settings.TEMPERATURE,
                response_format={
                    "type": "json_object"
                }
            )
            return response.choices[0].message.content
        except OpenAIError as e:
            print(f"LLM ERROR: ({i+1}/{retry})", e)
            time.sleep(10)

    return None