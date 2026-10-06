from pydantic import BaseModel, Field


class AuthIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=4, max_length=128)


class UserOut(BaseModel):
    id: int
    username: str
