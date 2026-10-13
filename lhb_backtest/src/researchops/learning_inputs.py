"""Project declared learning rows/columns with Arrow; preserve global key checks."""
import json
import pandas as pd
import pyarrow.parquet as pq

from ..alpharesearch.learning import KEYS,CLOCKS,LABELS,_keys
from ..alpharesearch.features.base import FeatureBlock
from ..alpharesearch.assembly import FeatureAssembly


def _logical_columns(file):
    metadata=file.schema_arrow.metadata or {}
    document=json.loads(metadata.get(b"pandas",b"{}"))
    index={x for x in document.get("index_columns",[]) if isinstance(x,str)}
    return [name for name in file.schema_arrow.names if name not in index]


def read_learning_inputs(roles,spec,registry,units,metadata):
    """Original files remain the source; no derived dataset or global fit stats.

    Only narrow key tables span the entire source. Feature values and training
    labels are decoded into pandas only for the explicit fixed time ranges.
    """
    cfg=spec.to_dict();names=cfg["features"]
    if not set(names)<=set(units):raise ValueError("Selected feature was not supplied by assembly")
    counts={}
    required={"decisions":KEYS+CLOCKS,"labels":KEYS+LABELS,
        "assembly_values":KEYS+names+["observed_end","known_at"],"assembly_missing":KEYS+names}
    for role,columns in required.items():
        file=pq.ParquetFile(roles[role]);count=file.metadata.num_rows
        if count>cfg["max_input_rows"]:raise ValueError("Declared input row cap exceeded or invalid table")
        logical=_logical_columns(file)
        if len(logical)!=len(set(logical)) or not set(columns)<=set(logical):
            raise ValueError("Required learning input columns missing or duplicated: "+role)
        if role in {"decisions","labels"} and set(logical)!=set(columns):
            raise ValueError("Strict decision and label table fields required")
        keys=pd.read_parquet(roles[role],columns=KEYS,engine="pyarrow")
        _keys(keys,role)
        if len(keys)!=count:raise ValueError("Learning input footer and keys disagree")
        if role=="decisions":
            train_keys=keys.loc[keys.trade_date.between(cfg["train_start"],cfg["train_end"])].copy()
        elif role=="labels":
            by_date=train_keys.merge(keys,on=["trade_date","stock_code"],suffixes=("_decision","_label"),validate="one_to_one")
            if not by_date.sample_id_decision.eq(by_date.sample_id_label).all():
                raise ValueError("Training label sample identity conflicts with stock/date keys")
            by_id=train_keys.merge(keys,on="sample_id",suffixes=("_decision","_label"),validate="one_to_one")
            if not (by_id.trade_date_decision.eq(by_id.trade_date_label)&by_id.stock_code_decision.eq(by_id.stock_code_label)).all():
                raise ValueError("Training label stock/date keys conflict with sample identity")
            del train_keys,by_date,by_id
        counts[role]=count
        del keys
    train=[("trade_date",">=",cfg["train_start"]),("trade_date","<=",cfg["train_end"])]
    predict=[("trade_date",">=",cfg["predict_start"]),("trade_date","<=",cfg["predict_end"])]
    frames={role:pd.read_parquet(roles[role],columns=columns,filters=[train] if role=="labels" else [train,predict],engine="pyarrow")
            for role,columns in required.items()}
    block=FeatureBlock(frames["assembly_values"],frames["assembly_missing"],
        {name:units[name] for name in names},metadata)
    assembly=FeatureAssembly(block,registry,tuple(metadata["source_bindings"]))
    return frames["decisions"],assembly,frames["labels"],counts["assembly_values"]
