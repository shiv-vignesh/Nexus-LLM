"""
Inference API router.

Provides OpenAI-compatible endpoints for completions and chat.
"""

import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
import json

from nexus.serving.schemas.inference import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    ChatRole,
    FinishReason,
    StreamChoice,
    StreamResponse,
)
from nexus.serving.schemas.common import UsageInfo, ErrorResponse
from nexus.serving.dependencies import get_inference_engine, get_batcher
from nexus.core.interfaces.inference_engine import (
    IInferenceEngine,
    GenerationRequest,
)
from nexus.inference.batching.batcher import DynamicBatcher

router = APIRouter(prefix="/v1", tags=["inference"])


@router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def create_completion(
    request: CompletionRequest,
    http_request: Request,
    engine: IInferenceEngine = Depends(get_inference_engine),
    batcher: DynamicBatcher | None = Depends(get_batcher),
):
    """Create a text completion.

    OpenAI-compatible endpoint for text generation.
    """
    request_id = str(uuid.uuid4())
    created = int(time.time())

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            _stream_completion(request, request_id, created, engine),
            media_type="text/event-stream",
        )

    # Handle batch prompt
    prompts = [request.prompt] if isinstance(request.prompt, str) else request.prompt

    try:
        choices = []

        for i, prompt in enumerate(prompts):
            # Create generation request
            gen_request = GenerationRequest(
                request_id=f"{request_id}-{i}",
                prompt=prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty,
                stop_sequences=request.stop or [],
                user_id=request.user,
            )

            # Use batcher if available, otherwise direct generation
            if batcher and batcher.is_running:
                response = await batcher.submit(gen_request)
            else:
                response = await engine.generate(gen_request)

            choices.append(CompletionChoice(
                index=i,
                text=response.generated_text,
                finish_reason=FinishReason(response.finish_reason),
            ))

        # Calculate usage (from last response)
        usage = UsageInfo(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

        return CompletionResponse(
            id=f"cmpl-{request_id}",
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    responses={500: {"model": ErrorResponse}},
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    engine: IInferenceEngine = Depends(get_inference_engine),
    batcher: DynamicBatcher | None = Depends(get_batcher),
):
    """Create a chat completion.

    OpenAI-compatible endpoint for chat-based generation.
    """
    request_id = str(uuid.uuid4())
    created = int(time.time())

    # Handle streaming
    if request.stream:
        return StreamingResponse(
            _stream_chat_completion(request, request_id, created, engine),
            media_type="text/event-stream",
        )

    try:
        # Convert messages to prompt
        prompt = request.to_prompt()

        # Create generation request
        gen_request = GenerationRequest(
            request_id=request_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            presence_penalty=request.presence_penalty,
            frequency_penalty=request.frequency_penalty,
            stop_sequences=request.stop or [],
            user_id=request.user,
        )

        # Generate
        if batcher and batcher.is_running:
            response = await batcher.submit(gen_request)
        else:
            response = await engine.generate(gen_request)

        # Build response
        choices = [ChatChoice(
            index=0,
            message=ChatMessage(
                role=ChatRole.ASSISTANT,
                content=response.generated_text.strip(),
            ),
            finish_reason=FinishReason(response.finish_reason),
        )]

        usage = UsageInfo(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=created,
            model=request.model,
            choices=choices,
            usage=usage,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _stream_completion(
    request: CompletionRequest,
    request_id: str,
    created: int,
    engine: IInferenceEngine,
) -> AsyncIterator[str]:
    """Stream completion chunks."""
    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=request.stop or [],
        stream=True,
    )

    try:
        async for chunk in engine.generate_stream(gen_request):
            data = {
                "id": f"cmpl-{request_id}",
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "text": chunk.text,
                    "finish_reason": chunk.finish_reason,
                }],
            }
            yield f"data: {json.dumps(data)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


async def _stream_chat_completion(
    request: ChatCompletionRequest,
    request_id: str,
    created: int,
    engine: IInferenceEngine,
) -> AsyncIterator[str]:
    """Stream chat completion chunks."""
    prompt = request.to_prompt()

    gen_request = GenerationRequest(
        request_id=request_id,
        prompt=prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        stop_sequences=request.stop or [],
        stream=True,
    )

    try:
        # Send initial chunk with role
        initial = StreamResponse(
            id=f"chatcmpl-{request_id}",
            created=created,
            model=request.model,
            choices=[StreamChoice(
                index=0,
                delta={"role": "assistant"},
                finish_reason=None,
            )],
        )
        yield f"data: {initial.model_dump_json()}\n\n"

        # Stream content
        async for chunk in engine.generate_stream(gen_request):
            data = StreamResponse(
                id=f"chatcmpl-{request_id}",
                created=created,
                model=request.model,
                choices=[StreamChoice(
                    index=0,
                    delta={"content": chunk.text} if chunk.text else {},
                    finish_reason=FinishReason(chunk.finish_reason) if chunk.finish_reason else None,
                )],
            )
            yield f"data: {data.model_dump_json()}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as e:
        error_data = {"error": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"
