class WaveView:
    """
    User-facing view for inspecting a Wave.

    Provides access to:
        - sample: ground-truth / generative sample (if set)
        - mean: posterior mean
        - precision: posterior precision
        - variance: posterior variance
    """
    def __init__(self, wave):
        self._wave = wave
        self._belief_cache = None

    def _belief(self):
        if self._belief_cache is not None:
            return self._belief_cache

        try:
            belief = self._wave.compute_belief()
        except RuntimeError as e:
            # Narrowly catch the expected failure mode
            msg = str(e)

            if "No child messages" in msg or "child messages" in msg:
                raise RuntimeError(
                    f"Posterior belief for wave '{self._wave.label}' "
                    "is not available yet. "
                    "Did you forget to run inference via `graph.run()`?"
                ) from None

            # Unexpected runtime error: re-raise
            raise

        self._belief_cache = belief
        return belief
    

    def __getitem__(self, key: str):
        if key == "sample":
            return self._wave.get_sample()

        belief = self._belief()
        if key == "mean":
            return belief.data
        elif key == "variance":
            return 1/belief.precision(raw = False)
        elif key == "precision":
            return belief.precision(raw = False)
        else:
            raise KeyError(
                f"Unknown attribute '{key}'. "
                f"Available: mean, variance, precision, sample"
            )

    def keys(self):
        return ["mean", "variance", "precision", "sample"]