# ==================== File: selector.py ====================
def select_top_etf(predictions: dict) -> str:
    if not predictions:
        return None
    return max(predictions, key=predictions.get)
