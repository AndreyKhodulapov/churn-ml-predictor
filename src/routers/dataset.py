from fastapi import APIRouter

from src.dataset_processor import DATASET

router = APIRouter(tags=["Dataset"], prefix="/dataset")


@router.get("/preview")
async def get_preview(row_number: int = 5):
    return DATASET[:row_number]
