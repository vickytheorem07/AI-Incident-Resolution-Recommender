"""Request and response models for incident recommendations."""

from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    incident_id: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=128)
    issue_description: str = Field(min_length=1, max_length=10_000)

    @field_validator("incident_id", "category", "issue_description")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RecommendationContent(BaseModel):
    response: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @field_validator("response", "reason")
    @classmethod
    def strip_and_require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class RecommendationResponse(RecommendationContent):
    incident_id: str = Field(min_length=1, max_length=128)

    @field_validator("incident_id")
    @classmethod
    def strip_incident_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value
