import pandas as pd

from src.services.dataset import DatasetProcessor, get_dataset


def data_preprocessing_pipeline(
    processor: DatasetProcessor = get_dataset()
) -> None:
    df = processor.get_dataframe()
    target = processor.target_column
    
    # check NaN in target and rm them
    df = df.dropna(subset=[target])
    
    # separate features and target
    X = df.drop(target, axis=1)
    y = df[target]

    # NaN processing

