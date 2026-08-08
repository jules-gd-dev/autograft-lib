"""Real cost calculation grounded in litellm's published per-model pricing."""

import litellm


def call_cost(response) -> float:
    """Return the real USD cost of a litellm completion response."""
    try:
        return float(litellm.completion_cost(completion_response=response))
    except Exception:  # noqa: BLE001
        return 0.0


def model_unit_prices(model: str) -> tuple[float, float]:
    """Return (input $/1M tokens, output $/1M tokens) for a model from litellm."""
    try:
        info = litellm.get_model_info(model)
        input_cost = float(info.get("input_cost_per_token", 0.0)) * 1_000_000
        output_cost = float(info.get("output_cost_per_token", 0.0)) * 1_000_000
        if input_cost or output_cost:
            return input_cost, output_cost
    except Exception:  # noqa: BLE001
        pass
    return 0.0, 0.0


def cost_from_tokens(
    prompt_tokens: int, completion_tokens: int, model: str
) -> float:
    """Compute USD cost from raw token counts using real unit prices."""
    in_price, out_price = model_unit_prices(model)
    return (prompt_tokens / 1_000_000.0) * in_price + (
        completion_tokens / 1_000_000.0
    ) * out_price
