from evals.base import Evaluator


class MYEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("MY", eval_cfg, **kwargs)
