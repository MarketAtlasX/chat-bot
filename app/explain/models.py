from pydantic import BaseModel, Field
from typing import Optional


class FeatureContribution(BaseModel):
    feature: str
    impact_pct: float
    direction: str = "positive"


class SHAPExplanation(BaseModel):
    prediction: str
    predicted_change_pct: float
    base_value: float = 0.0
    contributions: list[FeatureContribution] = Field(default_factory=list)


class AttentionWeight(BaseModel):
    input_label: str
    weight: float
    rank: int = 0


class AttentionExplanation(BaseModel):
    query: str
    top_events: list[AttentionWeight] = Field(default_factory=list)
    top_features: list[AttentionWeight] = Field(default_factory=list)


class GraphPathStep(BaseModel):
    source: str
    relation: str
    target: str


class GraphExplanation(BaseModel):
    start_entity: str
    end_entity: str
    path: list[GraphPathStep] = Field(default_factory=list)
    path_summary: str = ""


class ExplanationResult(BaseModel):
    shap: Optional[SHAPExplanation] = None
    attention: Optional[AttentionExplanation] = None
    graph: Optional[GraphExplanation] = None
