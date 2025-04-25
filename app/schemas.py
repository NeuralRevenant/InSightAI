from pydantic import BaseModel

class VQAResponse(BaseModel):
    answer: str
    reasoning: list[str]
