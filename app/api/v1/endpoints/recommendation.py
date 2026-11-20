from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, File, UploadFile, status

from app.schemas.load_docs import LoadDocsResponse
from app.schemas.recommendation import RecommendationRequest, RecommendationResponse
from app.services.load_docs import load_docs
from app.services.recommendation import get_recommendation

router = APIRouter()


@router.post(
    "/load_doc",
    status_code=status.HTTP_201_CREATED,
    response_model=LoadDocsResponse,
)
async def upload_document(file: UploadFile = File(...)) -> LoadDocsResponse:
    req_id = f"{datetime.now():%H%M%d%m%y}-load_docs-{uuid4().hex[:8]}"
    return await load_docs(req_id=req_id, file=file)


@router.post(
    "/recommendation",
    status_code=status.HTTP_200_OK,
    response_model=RecommendationResponse,
)
async def recommendation(
    payload: RecommendationRequest,
) -> RecommendationResponse:
    req_id = payload.incident_id
    return await get_recommendation(
        req_id=req_id,
        category=payload.category,
        issue_description=payload.issue_description,
        incident_id=payload.incident_id,
    )
