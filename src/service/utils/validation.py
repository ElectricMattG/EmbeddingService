"""
Input validation utilities for API requests.

This module provides comprehensive validation for text inputs, batch sizes,
and other request parameters using configuration-based limits.
"""

from typing import List, Union

from src.config import app_conf


class ValidationError(Exception):
    """Custom exception for validation errors."""

    def __init__(self, message: str, field: str = None, details: dict = None):
        self.message = message
        self.field = field
        self.details = details or {}
        super().__init__(self.message)


def validate_text_input(texts: Union[str, List[str]], field_name: str = "texts") -> None:
    """
    Validate text inputs against configuration limits.

    Args:
        texts: Single text or list of texts to validate
        field_name: Name of the field being validated (for error messages)

    Raises:
        ValidationError: If validation fails
    """
    max_text_length = app_conf.get('service.max_text_length') or 8192
    max_batch_size = app_conf.get('service.max_batch_size') or 100

    if isinstance(texts, str):
        # Single text validation
        if len(texts.strip()) == 0:
            raise ValidationError(
                f"Text cannot be empty or whitespace only",
                field=field_name,
                details={"received_length": len(texts.strip())}
            )

        if len(texts) > max_text_length:
            raise ValidationError(
                f"Text exceeds maximum length",
                field=field_name,
                details={
                    "max_length": max_text_length,
                    "received_length": len(texts)
                }
            )

    elif isinstance(texts, list):
        # List of texts validation
        if len(texts) == 0:
            raise ValidationError(
                f"Text list cannot be empty",
                field=field_name,
                details={"received_count": 0}
            )

        if len(texts) > max_batch_size:
            raise ValidationError(
                f"Too many texts in batch",
                field=field_name,
                details={
                    "max_batch_size": max_batch_size,
                    "received_count": len(texts)
                }
            )

        # Validate each text in the list
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(
                    f"All texts must be strings",
                    field=f"{field_name}[{i}]",
                    details={"received_type": type(text).__name__}
                )

            if len(text.strip()) == 0:
                raise ValidationError(
                    f"Text at index {i} cannot be empty or whitespace only",
                    field=f"{field_name}[{i}]",
                    details={"index": i}
                )

            if len(text) > max_text_length:
                raise ValidationError(
                    f"Text at index {i} exceeds maximum length",
                    field=f"{field_name}[{i}]",
                    details={
                        "max_length": max_text_length,
                        "received_length": len(text),
                        "index": i
                    }
                )
    else:
        raise ValidationError(
            f"Text input must be string or list of strings",
            field=field_name,
            details={"received_type": type(texts).__name__}
        )


def validate_batch_size(batch_size: int, field_name: str = "batch_size") -> None:
    """
    Validate batch size parameter.

    Args:
        batch_size: Batch size to validate
        field_name: Name of the field being validated

    Raises:
        ValidationError: If validation fails
    """
    max_batch_size = app_conf.get('service.max_batch_size') or 100

    if batch_size < 1:
        raise ValidationError(
            f"Batch size must be at least 1",
            field=field_name,
            details={"received_value": batch_size}
        )

    if batch_size > max_batch_size:
        raise ValidationError(
            f"Batch size exceeds maximum",
            field=field_name,
            details={
                "max_batch_size": max_batch_size,
                "received_value": batch_size
            }
        )


def validate_similarity_inputs(texts1: Union[str, List[str]], texts2: Union[str, List[str]]) -> None:
    """
    Validate inputs for similarity computation.

    Args:
        texts1: First set of texts
        texts2: Second set of texts

    Raises:
        ValidationError: If validation fails
    """
    # Validate both text sets
    validate_text_input(texts1, "texts1")
    validate_text_input(texts2, "texts2")

    # Additional validation for similarity operations
    max_candidates = app_conf.get('service.max_candidates') or 1000

    # Count total comparisons
    count1 = 1 if isinstance(texts1, str) else len(texts1)
    count2 = 1 if isinstance(texts2, str) else len(texts2)
    total_comparisons = count1 * count2

    if total_comparisons > max_candidates:
        raise ValidationError(
            f"Too many similarity comparisons",
            field="similarity_computation",
            details={
                "max_comparisons": max_candidates,
                "requested_comparisons": total_comparisons,
                "texts1_count": count1,
                "texts2_count": count2
            }
        )


def estimate_request_size(data: dict) -> int:
    """
    Estimate the size of a request in bytes.

    Args:
        data: Request data dictionary

    Returns:
        Estimated size in bytes
    """
    import json
    try:
        return len(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    except (TypeError, ValueError):
        # Fallback estimation
        total_size = 0
        for key, value in data.items():
            if isinstance(value, str):
                total_size += len(value.encode('utf-8'))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total_size += len(item.encode('utf-8'))
        return total_size


def validate_request_size(data: dict) -> None:
    """
    Validate that request size doesn't exceed limits.

    Args:
        data: Request data to validate

    Raises:
        ValidationError: If request is too large
    """
    max_request_size = app_conf.get('service.max_request_size') or 10485760  # 10MB
    estimated_size = estimate_request_size(data)

    if estimated_size > max_request_size:
        raise ValidationError(
            f"Request size exceeds maximum",
            field="request_size",
            details={
                "max_size_mb": max_request_size / 1024 / 1024,
                "estimated_size_mb": estimated_size / 1024 / 1024,
                "max_size_bytes": max_request_size,
                "estimated_size_bytes": estimated_size
            }
        )
