"""
narrator/claude_narrator.py
----------------------------
Generates institutional-quality derivatives risk commentary using the
Anthropic Claude API (claude-sonnet-4-5 model).

The narrator receives a fully-structured JSON payload and produces a
four-paragraph narrative covering P&L, Greeks, vol environment, and forward
risk watch items.
"""

import json
import logging

import anthropic

from config import CLAUDE_API_KEY

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-5"

_SYSTEM_PROMPT = (
    "You are a senior derivatives risk analyst at a major investment bank. "
    "Write institutional-quality daily risk commentary. "
    "4 paragraphs: "
    "(1) P&L summary and attribution drivers with specific numbers, "
    "(2) Greek exposures vs prior day and vs limits, "
    "(3) vol environment and market context, "
    "(4) specific alerts and watch items for tomorrow. "
    "Professional, assertive, quantitative. No disclaimers."
)


def generate_risk_commentary(payload: dict) -> str:
    """
    Call the Claude API to generate a four-paragraph risk commentary narrative.

    The entire *payload* dict is serialised to indented JSON and sent as the
    user message, giving Claude full quantitative context to produce precise,
    number-driven commentary.

    Parameters
    ----------
    payload : dict
        The structured report payload produced by
        :func:`engine.portfolio.build_report_payload`.  Must include:
        - ``portfolio_metrics`` (current + prior + P&L attribution)
        - ``market_context`` (per-ticker daily moves)
        - ``alerts`` (list of alert strings)
        - ``report_date`` (ISO-8601 string)

    Returns
    -------
    str
        Multi-paragraph text commentary suitable for embedding in the PDF
        report.  Returns an error notice string if the API call fails.

    Raises
    ------
    anthropic.APIError
        Propagates API-level errors after logging; callers should handle this
        and fall back to a placeholder string if needed.

    Examples
    --------
    >>> commentary = generate_risk_commentary(payload)
    >>> print(commentary[:100])
    'Portfolio delivered a net P&L of ...'
    """
    if not CLAUDE_API_KEY:
        logger.warning("ANTHROPIC_API_KEY not set; returning placeholder commentary.")
        return (
            "Risk commentary unavailable: ANTHROPIC_API_KEY environment variable "
            "is not configured. Please set it and re-run the pipeline to generate "
            "AI-powered narrative."
        )

    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    user_message = json.dumps(payload, indent=2, default=str)

    try:
        logger.info("Calling Claude API (%s) for risk commentary...", _MODEL)
        response = client.messages.create(
            model=_MODEL,
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message},
            ],
        )
        commentary: str = response.content[0].text
        logger.info(
            "Claude commentary generated: %d characters, %d input tokens, %d output tokens",
            len(commentary),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )
        return commentary
    except anthropic.APIConnectionError as exc:
        logger.error("Claude API connection error: %s", exc)
        return f"[Commentary generation failed — API connection error: {exc}]"
    except anthropic.RateLimitError as exc:
        logger.error("Claude API rate limit: %s", exc)
        return f"[Commentary generation failed — rate limit exceeded: {exc}]"
    except anthropic.APIStatusError as exc:
        logger.error("Claude API status error %s: %s", exc.status_code, exc.message)
        return f"[Commentary generation failed — API error {exc.status_code}: {exc.message}]"
    except Exception as exc:
        logger.error("Unexpected error calling Claude API: %s", exc)
        return f"[Commentary generation failed — unexpected error: {exc}]"
