from time import perf_counter

from fastapi import HTTPException, status
from langchain_groq import ChatGroq
from pydantic import SecretStr, ValidationError

from app.core.config import settings
from app.core.logger import setup_logger
from app.schemas.recommendation import RecommendationContent, RecommendationResponse
from app.services.embedding import Hugging_face_embedding
from app.services.prompts import recommendation_prompt
from app.services.vector_db import ChromaVectorDB


def _token_counts(message) -> tuple[int | None, int | None]:
    """Read token counts from LangChain's normalized or provider usage data."""
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    provider_usage = (
        response_metadata.get("token_usage")
        or response_metadata.get("usage")
        or {}
    )
    input_tokens = usage.get(
        "input_tokens",
        provider_usage.get("prompt_tokens", provider_usage.get("input_tokens")),
    )
    output_tokens = usage.get(
        "output_tokens",
        provider_usage.get("completion_tokens", provider_usage.get("output_tokens")),
    )
    return input_tokens, output_tokens


def _message_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        ).strip()
    return str(content).strip()


async def get_recommendation(
    req_id: str,
    category: str,
    issue_description: str,
    incident_id: str,
) -> RecommendationResponse:
    logger = setup_logger(__name__, req_id)
    request_started = perf_counter()
    response_status = status.HTTP_500_INTERNAL_SERVER_ERROR
    stage = "validation"

    logger.info(
        "Recommendation request started: incident_id=%s category=%s issue_chars=%d",
        incident_id,
        category,
        len(issue_description),
    )

    try:
        if not incident_id.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="incident_id must not be empty",
            )
        if not category.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="category must not be empty",
            )
        if not issue_description.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="issue_description must not be empty",
            )

        user_query = f"Category: {category}\nIssue description: {issue_description}"
        retrieval_started = perf_counter()
        stage = "retrieval"
        context = ""
        try:
            embedding = Hugging_face_embedding(
                settings.huggingface_embedding_model_name,
                settings.huggingface_api,
                req_id=req_id,
            )
            vector_db = ChromaVectorDB(
                embedding.embedding,
                settings.collection_name,
                settings.persist_dir,
                req_id=req_id,
            )
            relevant_docs = vector_db.similarity_search(
                user_query,
                k=settings.top_k,
                category=category,
            )
            logger.info(
                "Semantic retrieval completed: result_count=%d collection=%s",
                len(relevant_docs),
                settings.collection_name,
            )
            for rank, document in enumerate(relevant_docs, start=1):
                logger.info(
                    "Retrieved document: rank=%d id=%s source=%s",
                    rank,
                    document.id,
                    document.metadata.get("source", "unknown"),
                )
            context = "\n\n".join(
                f"[ticket_id: {document.id}]\n{document.page_content}"
                for document in relevant_docs
            )
            logger.info(
                "Recommendation retrieval completed: incident_id=%s result_count=%d latency_ms=%.1f",
                incident_id,
                len(relevant_docs),
                (perf_counter() - retrieval_started) * 1000,
            )
        except Exception:
            logger.exception(
                "Recommendation retrieval failed; continuing without retrieved context: incident_id=%s",
                incident_id,
            )

        prompt = recommendation_prompt(
            category=category,
            issue_description=issue_description,
            retrieved_context=context,
        )

        stage = "llm_initialization"
        llm = ChatGroq(
            model=settings.llm_model_name,
            temperature=settings.temperature,
            api_key=SecretStr(settings.groq_llm_api_key),
        )
        stage = "llm_generation"
        generation_started = perf_counter()
        logger.info(
            "Recommendation generation started: incident_id=%s model=%s",
            incident_id,
            settings.llm_model_name,
        )
        llm_message = await llm.ainvoke(prompt)
        generation_latency_ms = (perf_counter() - generation_started) * 1000
        input_tokens, output_tokens = _token_counts(llm_message)
        logger.info(
            "Recommendation generation completed: incident_id=%s input_tokens=%s output_tokens=%s latency_ms=%.1f",
            incident_id,
            input_tokens if input_tokens is not None else "unavailable",
            output_tokens if output_tokens is not None else "unavailable",
            generation_latency_ms,
        )

        stage = "response_validation"
        try:
            model_content = RecommendationContent.model_validate_json(
                _message_text(llm_message.content)
            )
        except ValidationError as exc:
            logger.warning(
                "Recommendation model returned invalid JSON response: incident_id=%s validation_errors=%d",
                incident_id,
                len(exc.errors()),
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="The recommendation model returned an invalid response.",
            ) from exc
        result = RecommendationResponse(
            incident_id=incident_id,
            **model_content.model_dump(),
        )

        response_status = status.HTTP_200_OK
        logger.info(
            "Recommendation request completed: incident_id=%s status_code=%d total_latency_ms=%.1f",
            incident_id,
            response_status,
            (perf_counter() - request_started) * 1000,
        )
        return result
    except HTTPException as exc:
        response_status = exc.status_code
        logger.warning(
            "Recommendation request ended with HTTP error: incident_id=%s status_code=%d stage=%s",
            incident_id,
            response_status,
            stage,
        )
        raise
    except Exception as exc:
        logger.exception(
            "Recommendation request failed: incident_id=%s stage=%s",
            incident_id,
            stage,
        )
        response_status = status.HTTP_502_BAD_GATEWAY if stage.startswith("llm") else status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=response_status,
            detail="Unable to generate an incident recommendation.",
        ) from exc
    finally:
        logger.info(
            "Recommendation request finished: incident_id=%s status_code=%d total_latency_ms=%.1f",
            incident_id,
            response_status,
            (perf_counter() - request_started) * 1000,
        )
