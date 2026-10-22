from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from recbole3.model.base import BaseCollator, BaseModel, BaseModelDataset, BaseRankingModel, BaseRetrievalModel, ModelConfig, ModelDatasets
from recbole3.model.tiger import TIGERConfig, TIGERModel, TIGERModelDataset
from recbole3.model.sequential import BaseSequentialModelDataset, HISTORY_ITEM_IDS, SequentialModelConfig, build_history_item_ids
from recbole3.trainer import Trainer
from recbole3.trainer_config import TrainerConfig
from recbole3.pipeline import Pipeline
from recbole3.utils import LazyImport

@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Static model table entry."""
    model_cls: type[BaseModel] | LazyImport | Any
    config_cls: type[ModelConfig]
    model_data_cls: type[BaseModelDataset[Any, Any]] | None = None
    trainer_cls: type[Trainer] = Trainer
    trainer_config_cls: type[TrainerConfig] = TrainerConfig
    pipeline_cls: type[Pipeline] | LazyImport = Pipeline
MODEL_TABLE: dict[str, ModelSpec] = {'tiger': ModelSpec(model_cls=TIGERModel, config_cls=TIGERConfig, model_data_cls=TIGERModelDataset, trainer_cls=Trainer, trainer_config_cls=TrainerConfig, pipeline_cls=Pipeline)}

def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_TABLE[name]
    except KeyError as exc:
        available = ', '.join(sorted(MODEL_TABLE)) or '<empty>'
        raise KeyError(f"Unknown model '{name}'. Available models: {available}") from exc
__all__ = ['BaseCollator', 'BaseModel', 'BaseModelDataset', 'BaseRankingModel', 'BaseRetrievalModel', 'BaseSequentialModelDataset', 'HISTORY_ITEM_IDS', 'MODEL_TABLE', 'ModelConfig', 'ModelDatasets', 'ModelSpec', 'SequentialModelConfig', 'TIGERConfig', 'TIGERModel', 'TIGERModelDataset', 'build_history_item_ids', 'get_model_spec']
