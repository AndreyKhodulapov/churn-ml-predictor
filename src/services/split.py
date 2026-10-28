import logging
from functools import lru_cache

from src.core.config import CLASS_SHARE_TOLERANCE
from src.ml.preprocessing import SplitData, prepare_data, split_data
from src.schemas.dataset import SplitDataInfo
from src.services.dataset import get_dataset
from src.services.utils import class_distribution

logger = logging.getLogger(__name__)


def describe_split(split: SplitData) -> SplitDataInfo:
    """Sizes of the train and test parts and the churn distribution in each of them."""
    return SplitDataInfo(
        train_size=len(split.y_train),
        train_distribution=class_distribution(split.y_train),
        test_size=len(split.y_test),
        test_distribution=class_distribution(split.y_test),
    )


def check_class_shares(info: SplitDataInfo, tolerance: float = CLASS_SHARE_TOLERANCE) -> None:
    """Log the churn distribution of both parts and warn about a class whose shares differ more than tolerance.

    A stratified split keeps the shares within one row; a gap means the test metrics would be biased.
    """
    train_shares = {label: part.share for label, part in info.train_distribution.items()}
    test_shares = {label: part.share for label, part in info.test_distribution.items()}
    logger.info(
        "Churn shares in train: %s, in test: %s",
        {label: round(share, 4) for label, share in train_shares.items()},
        {label: round(share, 4) for label, share in test_shares.items()},
    )

    for label in sorted(train_shares.keys() | test_shares.keys()):
        # A class absent from a part has a zero share there.
        gap = abs(train_shares.get(label, 0.0) - test_shares.get(label, 0.0))
        if gap > tolerance:
            logger.warning("Churn class %d share differs by %.4f between train and test", label, gap)


@lru_cache
def get_split() -> SplitData:
    """Prepare and split the default dataset once and reuse the result for every request.

    Feature gaps are kept: the churn pipeline fills them with statistics of the train part.
    """
    split = split_data(prepare_data(get_dataset().get_dataframe()))
    info = describe_split(split)
    logger.info("Split the dataset into %d train and %d test rows", info.train_size, info.test_size)
    # Checked once here rather than on every /dataset/split-info request, since the split is cached.
    check_class_shares(info)
    return split


def is_dataset_trainable() -> bool:
    """Whether get_split() has split the dataset, without trying to; a failed split is not cached, as in get_dataset()."""
    return get_split.cache_info().currsize > 0
