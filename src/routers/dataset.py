from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.schemas.dataset import DatasetInfo, DatasetRowChurn, SplitDataInfo
from src.services.dataset import DatasetProcessor, get_dataset
from src.services.preprocessing import SplitData, describe_split, get_split

router = APIRouter(tags=["Dataset"], prefix="/dataset")


@router.get("/preview")
def get_preview(
    dataset: Annotated[DatasetProcessor, Depends(get_dataset)],
    n: Annotated[int, Query(ge=1, le=100, description="Number of rows to return from the top of the dataset")] = 5,
) -> list[DatasetRowChurn]:
    return dataset.preview(n)


@router.get("/info")
def get_info(dataset: Annotated[DatasetProcessor, Depends(get_dataset)]) -> DatasetInfo:
    return dataset.get_info()


@router.get("/split-info")
def get_split_info(split: Annotated[SplitData, Depends(get_split)]) -> SplitDataInfo:
    return describe_split(split)
