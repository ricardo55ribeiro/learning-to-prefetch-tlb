"""Online linear and one-hidden-layer reward models for direct offsets."""

from random import Random

import numpy as np
from numpy.typing import NDArray

from tlb_prefetch.direct_offset import (
    ABSTAIN, ACTIONS, CONTEXT_COUNT, EPSILON, OFFSET_LIMIT, OFFSETS,
    WASTED_WALK_PENALTY, _PageHistory, _VALID_ACTIONS,
)
from tlb_prefetch.simulator import Access, PrefetchFeedback, _validate_integer


FloatArray = NDArray[np.float64]
INPUTS = CONTEXT_COUNT // 2 + 1
HIDDEN = 16
LEARNING_RATE = 0.05
INIT_SCALE = 0.1


def _features(delta: int | None, repeated: bool) -> FloatArray:
    if delta is None:
        category = 19
    elif delta < -OFFSET_LIMIT:
        category = 17
    elif delta > OFFSET_LIMIT:
        category = 18
    else:
        category = delta + OFFSET_LIMIT
    features = np.zeros(INPUTS, dtype=np.float64)
    features[category] = 1
    features[-1] = repeated
    features.flags.writeable = False
    return features


class _NeuralOffsetLearner:
    def __init__(self, *, seed: int = 0, page_walk_latency: int = 60) -> None:
        _validate_integer("seed", seed)
        _validate_integer("page_walk_latency", page_walk_latency, minimum=1)
        self._seed = seed
        self._latency = page_walk_latency
        self.reset()

    def reset(self) -> None:
        self._history = _PageHistory()
        self._rng = Random(self._seed)
        self._choice: tuple[FloatArray, int] | None = None
        self._pending: dict[int, tuple[FloatArray, int]] = {}
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.output_weights = np.zeros((len(OFFSETS), INPUTS), dtype=np.float64)
        self.output_bias = np.zeros(len(OFFSETS), dtype=np.float64)

    @property
    def parameters(self) -> tuple[FloatArray, ...]:
        return self.output_weights, self.output_bias

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for parameter in self.parameters)

    def observe(self, access: Access) -> int | None:
        features = _features(*self._history.observe(access))
        available = _VALID_ACTIONS[min(access.vpn, OFFSET_LIMIT)]
        if self._rng.random() < EPSILON:
            action = self._rng.choice(available)
        else:
            scores = self.scores(features)
            action = max(available, key=lambda index: 0.0 if index == ABSTAIN else scores[index - 1])
        self._choice = None
        if action == ABSTAIN:
            return None
        self._choice = features, action - 1
        return access.vpn + ACTIONS[action]

    def on_feedback(self, event: PrefetchFeedback) -> None:
        if event.kind in ("accepted", "rejected"):
            choice = self._choice
            self._choice = None
            if choice is None:
                return
            if event.kind == "accepted":
                self._pending[event.issued_access_index] = choice
                return
            reward = 0
        else:
            choice = self._pending.pop(event.issued_access_index, None)
            if choice is None:
                return
            reward = (
                event.saved_cycles if event.kind in ("timely", "late") else -WASTED_WALK_PENALTY
            )
        features, output = choice
        self._train(features, output, reward / self._latency)

    def scores(self, features: FloatArray) -> FloatArray:
        raise NotImplementedError

    def _train(self, features: FloatArray, output: int, target: float) -> None:
        raise NotImplementedError


class LinearOffsetLearner(_NeuralOffsetLearner):
    def scores(self, features: FloatArray) -> FloatArray:
        return self.output_weights @ features + self.output_bias

    def _train(self, features: FloatArray, output: int, target: float) -> None:
        weights = self.output_weights[output]
        error = weights @ features + self.output_bias[output] - target
        step = LEARNING_RATE * error
        weights -= step * features
        self.output_bias[output] -= step


class TinyMLPOffsetLearner(_NeuralOffsetLearner):
    def _reset_parameters(self) -> None:
        # Keep parameter draws separate from exploration draws
        init = np.random.default_rng(self._seed)
        self.hidden_weights = init.uniform(-INIT_SCALE, INIT_SCALE, (HIDDEN, INPUTS))
        self.hidden_bias = np.zeros(HIDDEN, dtype=np.float64)
        self.output_weights = np.zeros((len(OFFSETS), HIDDEN), dtype=np.float64)
        self.output_bias = np.zeros(len(OFFSETS), dtype=np.float64)

    @property
    def parameters(self) -> tuple[FloatArray, ...]:
        return self.hidden_weights, self.hidden_bias, self.output_weights, self.output_bias

    def _hidden(self, features: FloatArray) -> FloatArray:
        return np.tanh(self.hidden_weights @ features + self.hidden_bias)

    def scores(self, features: FloatArray) -> FloatArray:
        return self.output_weights @ self._hidden(features) + self.output_bias

    def _train(self, features: FloatArray, output: int, target: float) -> None:
        hidden = self._hidden(features)
        weights = self.output_weights[output]
        error = weights @ hidden + self.output_bias[output] - target
        step = LEARNING_RATE * error
        # Propagate through the selected output row before changing its weights
        hidden_step = step * weights * (1 - hidden * hidden)
        self.hidden_weights -= np.outer(hidden_step, features)
        self.hidden_bias -= hidden_step
        weights -= step * hidden
        self.output_bias[output] -= step
