from pydantic import BaseModel

class MessageItem(BaseModel):
    role: str
    content: str

class RequestModel(BaseModel):
    model: str
    messages: list[MessageItem]
    
class ResponseModel(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChoiceItem]
    usage: dict    