from typing import Annotated

from fastapi import APIRouter, Depends, Query

from src.schemas.dataset import DatasetRowChurn
from src.services.dataset import get_dataset

router = APIRouter(tags=["Dataset"], prefix="/dataset")


@router.get("/preview")
def get_preview(
    dataset: Annotated[list[DatasetRowChurn], Depends(get_dataset)],
    n: Annotated[int, Query(ge=1, le=100, description="Number of rows to return from the top of the dataset")] = 5,
) -> list[DatasetRowChurn]:
    return dataset[:n]
