"""Response model for CSV ticket uploads."""

from typing import Literal

from pydantic import BaseModel, Field


class LoadDocsResponse(BaseModel):
    status: Literal["success"]
    filename: str
    message: str
    chunks: int = Field(ge=0)
