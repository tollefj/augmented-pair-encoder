import logging
from typing import List, Tuple

from model.util import PairInput
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


class CorrelationEvaluator:
    def __init__(self, pairs: List[Tuple[str]], scores: List[float], use_pearson=False):
        self.pairs = pairs
        self.scores = scores
        self.corr = pearsonr if use_pearson else spearmanr

    @classmethod
    def load(cls, inputs: List[PairInput], **kwargs):
        return cls([i.pair for i in inputs], [i.label for i in inputs], **kwargs)

    def __call__(self, model, epoch: int = -1, steps: int = -1) -> float:
        pred_scores = model.predict(self.pairs)
        corr, _ = self.corr(self.scores, pred_scores)
        logging.info(
            f"Evaluation at Epoch {epoch} (step {steps})\n\
                Correlation ({self.corr.__name__}):\t{corr:.4f}"
        )
        return corr
