from ..wave import Wave
from ..factor import Factor
from ...core.backend import np
from ...core.uncertain_array import UncertainArray as UA
from ...core.types import PrecisionMode
from ...core.rng_utils import get_rng
from .wave_view import WaveView
import contextlib
import threading
from typing import Literal, Optional, Callable

_current_graph = threading.local()


class Graph:
    """
    The central coordinator of a Computational Factor Graph (CFG) in gPIE.

    A Graph manages the connectivity, compilation, scheduling, and execution
    of all `Wave` (latent variables) and `Factor` (probabilistic operators) nodes.

    Responsibilities:
    - Collect Wave and Factor instances into a coherent model
    - Compile the model into a topologically sorted DAG
    - Propagate precision modes forward/backward for consistency
    - Perform forward-backward inference via message passing
    - Support sampling and observation via Measurement nodes

    Key Concepts:
    - Compilation is required before inference; it discovers all reachable nodes.
    - Message passing is done in topological order (`forward`) and reverse (`backward`).
    - The graph supports sampling (`generate_sample`) and visualization (`visualize`).

    Usage Example:
        >>> g = Graph()
        >>> with g.observe():
        >>>     z = GaussianMeasurement(...) @ (x + y)
        >>> g.compile()
        >>> g.run(n_iter=10)

    Internal State:
        _nodes (set): All Wave and Factor nodes in the graph
        _waves (set): Subset of Wave instances
        _factors (set): Subset of Factor instances
        _nodes_sorted (list): Nodes in forward topological order
        _nodes_sorted_reverse (list): Nodes in reverse topological order
        _rng (np.random.Generator): Default RNG for sampling and initialization
    """

    def __init__(self):
        self._nodes = set()
        self._waves = set()
        self._factors = set()
        self._nodes_sorted = None
        self._nodes_sorted_reverse = None
        self._rng = get_rng()  # default RNG for sampling

        # Scheduling related
        self._full_batch_size: Optional[int] = None
    
    @contextlib.contextmanager
    def observe(self):
        """
        Context manager for automatic measurement registration.
        Usage:
            with self.observe():
                Z = AmplitudeMeasurement(...) @ x
        """
        _current_graph.value = self
        try:
            yield
        finally:
            _current_graph.value = None
    
    @staticmethod
    def get_active_graph():
        """Return the current graph context if inside a `with observe()` block."""
        return getattr(_current_graph, "value", None)


    def compile(self):
        """
        Discover the full computational factor graph topology starting from Measurement nodes.

        This method performs the following steps:
            1. Detects Measurement objects defined on the Graph.
            2. Traverses the graph in reverse from Measurements to Priors,
            registering all Waves and Factors.
            3. Sorts all nodes topologically based on generation index.
            4. Propagates precision mode (scalar/array) forward and backward.
            5. Assigns default precision mode where unresolved.
            6. Finalizes wave structure (e.g., shape/dtype assertions).

        Scheduling semantics:
            - full_batch_size is defined as the maximum batch_size
              among all Wave nodes.
            - Nodes whose batch_size < full_batch_size are always
              executed in parallel (block=None).
        """

        self._nodes.clear()
        self._waves.clear()
        self._factors.clear()

        # --- Step 1: Initialize unseen set with Measurement objects ---
        unseen = set()
        for name in dir(self):
            obj = getattr(self, name)
            if isinstance(obj, Factor) and obj.output is None:
                unseen.add(obj)

        # --- Step 2: Recursively traverse the graph in reverse ---
        while unseen:
            factor = unseen.pop()
            if factor in self._factors:
                continue  # already processed

            self._factors.add(factor)
            self._nodes.add(factor)

            for wave in factor.inputs.values():
                if wave not in self._waves:
                    self._waves.add(wave)
                    self._nodes.add(wave)

                parent = wave.parent
                if parent is not None and parent not in self._factors:
                    unseen.add(parent)

        # --- Step 3: Topological sort by generation ---
        self._nodes_sorted = sorted(self._nodes, key=lambda x: x.generation)
        self._nodes_sorted_reverse = list(reversed(self._nodes_sorted))

        # --- Step 4: Precision mode propagation ---
        for node in self._nodes_sorted:
            if hasattr(node, "set_precision_mode_forward"):
                node.set_precision_mode_forward()

        for node in self._nodes_sorted_reverse:
            if hasattr(node, "set_precision_mode_backward"):
                node.set_precision_mode_backward()

        # --- Step 5: Default precision mode fallback ---
        for wave in self._waves:
            if wave.precision_mode is None:
                wave._set_precision_mode("scalar")

        for factor in self._factors:
            if factor.precision_mode is None:
                factor._set_precision_mode("scalar")
                
        # --- Step 6 : initialize messages ---
        for wave in self._waves:
            for factor in wave.children:
                wave.child_messages[factor] = UA.zeros(
                    event_shape=wave.event_shape,
                    batch_size=wave.batch_size,
                    dtype=wave.dtype,
                    precision=1.0,
                    scalar_precision=(wave.precision_mode_enum == PrecisionMode.SCALAR),
                )
        # ------------------------------------------------------------
        # Determine full batch size for block scheduling
        # ------------------------------------------------------------
        batch_sizes = [wave.batch_size for wave in self._waves]

        if not batch_sizes:
            raise RuntimeError("No Wave nodes found during compilation.")

        self._full_batch_size = max(batch_sizes)

    
    def to_backend(self) -> None:
        """
        Move all graph data (Waves, Factors, and their internal arrays) to the current backend (NumPy or CuPy),
        and resync RNGs. Sample arrays and observed values are delegated to node-local logic.
        """
        from ...core.rng_utils import get_rng

        # Waves
        for wave in self._waves:
            wave.to_backend()

        # Factors (Prior, Measurement, Propagator etc.)
        for factor in self._factors:
            factor.to_backend()

    
    def get_wave(self, label: str):
        """
        Retrieve the Wave instance with the given label.

        Args:
            label (str): Label assigned to the Wave.

        Returns:
            Wave instance with the specified label.

        Raises:
            ValueError: If no wave with the given label exists or if multiple waves share the label.
        """
        matches = [w for w in self._waves if getattr(w, "label", None) == label]
        if not matches:
            raise ValueError(f"No wave found with label '{label}'")
        if len(matches) > 1:
            raise ValueError(f"Multiple waves found with label '{label}'")
        return matches[0]
    
    def get_factor(self, label: str):
        """
        Retrieve the Factor instance with the given label.

        Args:
            label (str): Label assigned to the Factor.

        Returns:
            Factor instance with the specified label.

        Raises:
            ValueError: If no factor with the given label exists or if multiple factors share the label.
        """
        matches = [f for f in self._factors if getattr(f, "label", None) == label]
        if not matches:
            raise ValueError(f"No factor found with label '{label}'")
        if len(matches) > 1:
            raise ValueError(f"Multiple factors found with label '{label}'")
        return matches[0]
    

    def set_init_strategy(self, label: str, mode: str, data: Optional[np().ndarray] = None, verbose = True) -> None:
        """
        Set initialization strategy for the Prior associated with the given Wave label.

        Args:
            label (str): Label of the Wave node defined in the model DSL.
            mode (str): Initialization strategy ("uninformative", "sample", "manual").
            data (ndarray, optional): Manual initialization data (required if mode='manual').

        Raises:
            KeyError: If the given label does not correspond to any Wave.
            TypeError: If the corresponding parent node is not a Prior.
            ValueError: If mode is invalid or 'manual' is selected without data.
        """
        wave = self.get_wave(label)
        parent = wave.parent
        from ..prior.base import Prior 

        if not isinstance(parent, Prior):
            raise TypeError(f"Wave '{label}' is not generated by a Prior (found {type(parent).__name__}).")

        if mode == "manual":
            if data is None:
                raise ValueError("Manual initialization selected, but 'data' argument is missing.")
            parent.set_manual_init(data)
            parent.set_init_strategy("manual")
        else:
            parent.set_init_strategy(mode)

        if verbose:
            print(f"[Graph] Set init strategy for Prior '{type(parent).__name__}' (wave='{label}') → '{mode}'")


    def set_all_init_strategies(self, strategy_dict: dict[str, tuple[str, Optional[np().ndarray]]], verbose = True) -> None:
        """
        Set initialization strategies for multiple Priors at once.

        Args:
            strategy_dict (dict): A dictionary mapping label -> (mode, data)
                Example:
                    {
                        "x": ("manual", ndarray_x),
                        "y": ("uninformative", None),
                        "z": ("sample", None),
                    }

        Raises:
            ValueError: If an invalid mode is provided or data is missing for manual mode.
        """
        from ..prior.base import Prior
        count = 0

        for label, (mode, data) in strategy_dict.items():
            wave = self.get_wave(label)
            parent = wave.parent

            if parent is None or not isinstance(parent, Prior):
                raise TypeError(f"Wave '{label}' is not generated by a Prior node.")

            if mode == "manual":
                if data is None:
                    raise ValueError(f"Manual init for '{label}' requires ndarray data.")
                parent.set_manual_init(data)
                parent.set_init_strategy("manual")
            else:
                parent.set_init_strategy(mode)

            count += 1

        if verbose:
            print(f"[Graph] Applied initialization strategies for {count} Priors.")



    def forward(self, block=None):
        """
        Execute forward message passing.

        If `block` is provided, it is passed only to nodes whose
        batch_size matches the graph's full batch size.
        Other nodes are called with `block=None`.
        """
        B = self._full_batch_size

        for node in self._nodes_sorted:
            if block is not None and node.batch_size == B:
                node.forward(block=block)
            else:
                node.forward()


    def backward(self, block=None):
        """
        Execute backward message passing.

        If `block` is provided, it is passed only to nodes whose
        batch_size matches the graph's full batch size.
        Other nodes are called with `block=None`.
        """
        B = self._full_batch_size

        for node in self._nodes_sorted_reverse:
            if block is not None and node.batch_size == B:
                node.backward(block=block)
            else:
                node.backward()


    def run(
        self,
        n_iter: int = 10,
        *,
        schedule: Literal["parallel", "sequential"] = "parallel",
        block_size: Optional[int] = 1,
        device: Literal["cpu", "cuda"] = "cpu",
        callback: Optional[Callable[["Graph", int], None]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Run Expectation Propagation (EP) inference on the compiled graph.

        Device policy (session-based):
            - All non-EP workloads (model definition, compile, data generation, visualization) are CPU-first.
            - This method creates an inference session:
                * at session start: move the entire graph state to the specified execution device
                * at session end: ALWAYS move the entire graph state back to CPU (NumPy backend)

        Notes:
            - The global backend is an internal implementation detail; users are not expected to manage it.
            - RNG is NumPy-based; if device="cuda", random numbers are generated on CPU and then transferred.

        Args:
            n_iter: Number of EP iterations.
            schedule:
                - "parallel": full-batch updates only (block=None).
                - "sequential": block-wise updates over the full-batch wave(s).
            block_size:
                Block size for sequential updates. If None or >= full batch size, falls back to full-batch.
            device:
                - "cpu": NumPy backend
                - "cuda": CuPy backend (requires CuPy)
            callback:
                Optional callback called as callback(graph, t) after each iteration t.
            verbose:
                If True, show a progress bar (requires tqdm).

        Raises:
            RuntimeError: If graph is not compiled.
            ValueError: If invalid device/schedule parameters are given.
        """
        # Import locally to keep module-level import surface small
        import numpy as _np
        from ...core.backend import set_backend

        # ------------------------------
        # Preconditions
        # ------------------------------
        if self._full_batch_size is None or self._nodes_sorted is None or self._nodes_sorted_reverse is None:
            raise RuntimeError("Graph must be compiled before run().")

        if n_iter < 0:
            raise ValueError(f"n_iter must be non-negative, got {n_iter}")

        if schedule not in ("parallel", "sequential"):
            raise ValueError(f"Unknown schedule: {schedule}")

        # ------------------------------
        # Helper: select execution backend module
        # ------------------------------
        def _select_backend(exec_device: str):
            if exec_device == "cpu":
                return _np
            if exec_device == "cuda":
                try:
                    import cupy as cp
                except ImportError as e:
                    raise RuntimeError("device='cuda' was requested, but CuPy is not installed.") from e
                return cp
            raise ValueError(f"Unknown device: {exec_device}")

        # Always restore to CPU at the end of this method (even on exceptions)
        exec_backend = _select_backend(device)

        try:
            # ------------------------------
            # Enter inference session: switch backend + move graph state
            # ------------------------------
            set_backend(exec_backend)
            self.to_backend()

            B = self._full_batch_size

            # ------------------------------
            # Build block schedule
            # ------------------------------
            if schedule == "parallel" or block_size is None or block_size >= B:
                blocks = [None]
            else:
                if block_size <= 0:
                    raise ValueError(f"block_size must be positive, got {block_size}")
                from ...core.blocks import BlockGenerator
                blocks = list(BlockGenerator(B=B, block_size=block_size).iter_blocks())

            # ------------------------------
            # Iteration driver (optional progress bar)
            # ------------------------------
            if verbose:
                try:
                    from tqdm import tqdm
                    iterator = tqdm(range(n_iter), desc="EP Iteration")
                except ImportError:
                    iterator = range(n_iter)
            else:
                iterator = range(n_iter)

            # ------------------------------
            # Warm-start:
            #   - ensures last_forward_message / last_backward_messages buffers exist
            # ------------------------------
            self.forward(block=None)
            self.backward(block=None)

            # ------------------------------
            # Main EP loop
            # ------------------------------
            for t in iterator:
                for blk in blocks:
                    self.forward(block=blk)
                    self.backward(block=blk)

                if callback is not None:
                    callback(self, t)

        finally:
            # ------------------------------
            # Exit inference session: ALWAYS restore CPU backend + move state back
            # ------------------------------
            set_backend(_np)
            self.to_backend()



    def generate_sample(self, rng=None, update_observed: bool = True, mask: Optional[np().ndarray] = None):
        """
        Generate a full sample from the generative model defined by the graph.

        Args:
            rng: RNG used for latent and observed sampling (optional).
            update_observed: If True, observed data is updated from the sample.
            mask: Optional mask to apply to Measurement nodes during observation update.
        """
        rng = rng or get_rng()

        # 1. Generate latent samples
        for node in self._nodes_sorted:
            if isinstance(node, Wave):
                node._generate_sample(rng=rng)

        # 2. Generate noisy observed samples
        for meas in self._factors:
            if hasattr(meas, "_generate_sample") and callable(meas._generate_sample):
                meas._generate_sample(rng)

        # 3. Promote to observed (with optional mask)
        if update_observed:
            for meas in self._factors:
                if hasattr(meas, "update_observed_from_sample"):
                    meas.update_observed_from_sample(mask=mask)
    
    def generate_observations(self, rng=None, mask=None):
        """
        Generate synthetic observations from the current generative model.

        This method:
            1. Generates latent samples if not already set.
            2. Generates noisy measurement samples.
            3. Promotes them to observed data.

        Args:
            rng: RNG used for sampling.
            mask: Optional mask applied when updating observed data.
        """
        self.generate_sample(rng=rng, update_observed=True, mask=mask)



    def set_init_rng(self, rng):
        """
        Propagate RNG to all factors that support message initialization.
        This is separate from sample RNG, and is used for initial message setup.

        Args:
            rng (np.random.Generator): RNG to be used for initializing messages.
        """
        for factor in self._factors:
            if hasattr(factor, "set_init_rng"):
                factor.set_init_rng(rng)

        for wave in self._waves:
            if hasattr(wave, "set_init_rng"):
                wave.set_init_rng(rng)

    def clear_sample(self):
        """
        Clear all sample values stored in Wave nodes in the graph.
        """
        for wave in self._waves:
            wave.clear_sample()

    def summary(self):
        """Print a summary of the graph structure."""
        print("Graph Summary:")
        print(f"- {len(self._waves)} Wave nodes")
        print(f"- {len(self._factors)} Factor nodes")
    

    def to_networkx(self) -> "nx.DiGraph":
        import networkx as nx

        G = nx.DiGraph()
        for wave in self._waves:
            G.add_node(
                id(wave),
                label=(wave.label if getattr(wave, "label", None) else "Wave"),
                type="wave",
                ref=wave,
            )
        for factor in self._factors:
            factor_label = getattr(factor, "label", None)
            if not isinstance(factor_label, str) or not factor_label:
                factor_label = factor.__class__.__name__

            G.add_node(
                id(factor),
                label=factor_label,
                type="factor",
                ref=factor,
            )
            for wave in factor.inputs.values():
                G.add_edge(id(wave), id(factor))
            if getattr(factor, "output", None):
                G.add_edge(id(factor), id(factor.output))
        return G


    def visualize(
        self,
        backend: str = "bokeh",
        layout: str = "graphviz",
        output_path: Optional[str] = None
        ):
        from .visualization import visualize_graph
        return visualize_graph(self, backend=backend, layout=layout, output_path=output_path)
    

    def __getitem__(self, key: str):
        wave = self.get_wave(key)
        if wave is None:
            raise KeyError(f"No wave named '{key}'")
        return WaveView(wave)
    
    def set_sample(self, name: str, value):
        wave = self.get_wave(name)
        if wave is None:
            raise KeyError(f"No wave named '{name}'")
        wave.set_sample(value)
