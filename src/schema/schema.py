from pydantic import BaseModel, Field


class STAR(BaseModel):
    situation: str = Field(min_length=10)
    task: str = Field(min_length=10)
    action: str = Field(min_length=10)
    result: str = Field(min_length=10)


class InterviewAnalysis(BaseModel):
    star: STAR
    strengths: list[str] = Field(min_length=1)
    improvements: list[str] = Field(min_length=1)


class EvaluationScore(BaseModel):
    situation_score: int = Field(ge=1, le=5)
    task_score: int = Field(ge=1, le=5)
    action_score: int = Field(ge=1, le=5)
    result_score: int = Field(ge=1, le=5)

    strengths_score: int = Field(ge=1, le=5)
    improvements_score: int = Field(ge=1, le=5)

    overall_score: int = Field(ge=1, le=5)

    feedback: str