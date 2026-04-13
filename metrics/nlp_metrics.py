from .base_metric import BaseMetric, MetricResult, MetricRegistry

try:
    from rouge_score import rouge_scorer
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    # Try to download if missing
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    rouge_scorer = None
    nltk = None

@MetricRegistry.register
class ROUGEMetric(BaseMetric):
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) if rouge_scorer else None

    @property
    def name(self) -> str:
        return "rouge_l"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        if not self.scorer or not reference:
            return MetricResult(name=self.name, value=0.0)
        scores = self.scorer.score(reference, prediction)
        return MetricResult(name=self.name, value=scores['rougeL'].fmeasure)

@MetricRegistry.register
class BLEUMetric(BaseMetric):
    @property
    def name(self) -> str:
        return "bleu"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        if not nltk or not reference:
            return MetricResult(name=self.name, value=0.0)
        # Basic tokenization
        ref_tokens = reference.lower().split()
        pred_tokens = prediction.lower().split()
        
        # Avoid zero scores for short sequences
        smooth = SmoothingFunction().method1
        val = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
        return MetricResult(name=self.name, value=val)

@MetricRegistry.register
class ToxicityMetric(BaseMetric):
    """
    Heuristic-based toxicity detection (basic rule-based for demo).
    In production, use a dedicated model or API.
    """
    @property
    def name(self) -> str:
        return "toxicity_score"

    def compute(self, prediction: str, reference: str, **kwargs) -> MetricResult:
        toxic_words = ["hate", "kill", "stupid", "idiot", "violence"]  # Sample list
        score = 0
        words = prediction.lower().split()
        for w in words:
            if any(tw in w for tw in toxic_words):
                score += 1
        val = min(1.0, score / 5.0) # Normalize
        return MetricResult(name=self.name, value=val)
