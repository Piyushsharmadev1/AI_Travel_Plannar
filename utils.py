def budget_bucket(cost_per_day: float) -> str:
    if cost_per_day <= 60:
        return "budget"
    if cost_per_day <= 120:
        return "mid"
    return "premium"
