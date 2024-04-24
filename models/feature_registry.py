import os
import functools

import torch
import orjson as json
from pydantic import BaseModel, computed_field


class SpectreFeatureSample(BaseModel):
    text: str
    act: float


class SpectreFeature(BaseModel):
    index: int
    label: str
    attributes: str
    reasoning: str
    density: float
    confidence: float
    vec: torch.FloatTensor
    high_act_samples: list[SpectreFeatureSample]
    low_act_samples: list[SpectreFeatureSample]

    class Config:
        arbitrary_types_allowed = True

    @computed_field
    @property
    def max_act(self) -> float:
        return self.high_act_samples[0].act

    def __str__(self) -> str:
        return f"SpectreFeature #{self.index}: {self.label} (confidence={self.confidence:.2f}, density={self.density:.4f})"

    def __repr__(self) -> str:
        return f"SpectreFeature(#{self.index}: {self.label} (confidence={self.confidence:.2f}, density={self.density:.4f}))"


class FeatureSortCriteria:
    def raw_activation(feat: SpectreFeature, act: float) -> float:
        return act

    def normalized_activation(feat: SpectreFeature, act: float) -> float:
        return act / feat.max_act

    def blended_activation(feat: SpectreFeature, act: float) -> float:
        return act / feat.max_act + feat.confidence + feat.density / 10

    def confidence(feat: SpectreFeature, act: float) -> float:
        return feat.confidence

    def density(feat: SpectreFeature, act: float) -> float:
        return feat.density


def load_jsonl(path: os.PathLike) -> list[any]:
    xs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line != "":
                xs.append(json.loads(line.encode("utf-8")))
    return xs


@functools.cache
def load_spectre_features(model_id: str) -> list[SpectreFeature]:
    features = load_jsonl(f"../features/spectre_features.{model_id}.jsonl")
    interpreted_features = load_jsonl(
        f"../features/interpreted_spectre_features.{model_id}.jsonl"
    )
    combined_features = []
    for feat in features:
        index = feat["index"]
        interpreted_feat = next(
            it for it in interpreted_features if it["index"] == index
        )
        combined_features.append(
            SpectreFeature(
                index=index,
                label=interpreted_feat["label"],
                attributes=interpreted_feat["attributes"],
                reasoning=interpreted_feat["reasoning"],
                density=feat["density"],
                confidence=interpreted_feat["confidence"],
                vec=torch.tensor(feat["vec"]),
                high_act_samples=feat["highActSamples"],
                low_act_samples=feat["lowActSamples"],
            )
        )
    return combined_features
