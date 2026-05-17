from __future__ import annotations

from src.core.decision_engine import ActionEvaluation


def summarize_evaluation(evaluation: ActionEvaluation) -> str:
    reason_text = "；".join(evaluation.reasons)
    risk_text = "；".join(evaluation.risks)
    if risk_text:
        return f"{reason_text}。风险：{risk_text}"
    return reason_text
