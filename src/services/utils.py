import pandas as pd

from src.schemas.dataset import ClassDistribution


def class_distribution(target: pd.Series) -> dict[int, ClassDistribution]:
    """Count and share of every class present in the target; absent classes are omitted."""
    counts = target.value_counts().sort_index()
    return {
        int(label): ClassDistribution(count=int(count), share=count / len(target))
        for label, count in zip(counts.index.tolist(), counts.tolist())
    }
