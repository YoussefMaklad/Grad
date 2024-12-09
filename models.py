from pydantic import BaseModel

class MessageItem(BaseModel):
    role: str
    content: str

class ChoiceItem(BaseModel):
    index: int
    message: MessageItem
    finish_reason: str

class RagModelRequestModel(BaseModel):
    model: str
    messages: list[MessageItem]
    
class RagModelResponseModel(BaseModel):
    lm_response: str

class FuzzerRequestModel(BaseModel):
    model: str
    messages: list[MessageItem]
    
class FuzzerResponseModel(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChoiceItem]
    usage: dict