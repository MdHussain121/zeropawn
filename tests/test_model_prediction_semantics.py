"""I-1 / I-3 regression tests: model prediction semantics.

The 2026-10-31 model-architecture/inference audit (finding I-1) found that
nothing automated asserted the model's OUTPUT SEMANTICS on real predictions:
WDL class ordering ([win, draw, loss] relative to the side to move), the
exactly-one-softmax consumption chain, logits-vs-probabilities, or the
conversion to the scalar search value.  ``scripts/test_value_signs.py`` (the
only artifact exercising value signs on real weights) sat outside
``testpaths`` and contained zero assertions; its useful intent is promoted
here as a checkpoint-gated test.

Two layers:

* Controlled deterministic models (zero/biased final layers) pin the
  interpretation: a deliberately dominant winning class MUST be read as a
  win for the side to move, and every consumer chain (CLI factory, UCI
  callback, worker aggregator) must agree with the canonical conversion.
* Real-checkpoint tests (skip-if-absent) pin perspective behavior on real
  weights: color-mirrored positions produce identical predictions, and an
  extreme lost position (bare king, side to move) is negative.

Note (recorded, not asserted): the designated 2026-10-31 baseline
(ckpt-10) is perspective-consistent but NOT material-calibrated (side-to-move
up a queen evaluates ~ -0.03): semantics are correct, prediction *quality*
is bounded by the artifact (audit I-5; no strength claim attached).
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import zeropawn_cpp  # noqa: E402

from python.model.chess_net import ChessNet  # noqa: E402

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def _controlled_model(
    wdl_bias=(0.0, 0.0, 0.0),
    policy_bias_index: Optional[int] = None,
    policy_bias_value: float = 0.0,
) -> ChessNet:
    """Deterministic ChessNet fixture: the output IS the final-layer bias.

    All other parameters are zeroed, so ``forward`` returns exactly the
    configured logits for any input (bit-deterministic, no random backbone
    contribution).  All configured values are exactly representable in bf16,
    so the fixture is also stable under the CPU bf16 autocast the consumer
    chains apply.
    """
    model = ChessNet(num_blocks=1, num_filters=8)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.value_fc2.bias.copy_(torch.tensor([float(v) for v in wdl_bias]))
        if policy_bias_index is not None:
            model.policy_fc.bias[policy_bias_index] = policy_bias_value
    model.eval()
    return model


def _planes_tensor(fen: str = START_FEN) -> torch.Tensor:
    game = zeropawn_cpp.ChessGame(fen)
    return torch.from_numpy(zeropawn_cpp.encode_state(game)).unsqueeze(0)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


class _StubCpp:
    """Injected cpp module: fixed planes for any game (consumer tests)."""

    def __init__(self, planes: np.ndarray):
        self._planes = planes

    def encode_state(self, game):
        return self._planes


# ---------------------------------------------------------------------------
# WDL class ordering and the scalar conversion (controlled models)
# ---------------------------------------------------------------------------


def test_win_dominant_class_is_read_as_a_win_for_the_side_to_move():
    model = _controlled_model(wdl_bias=(10.0, 0.0, 0.0))
    x = _planes_tensor()
    with torch.no_grad():
        _, wdl_logits = model(x)
    value = float(ChessNet.wdl_logits_to_value(wdl_logits.float())[0])
    assert value > 0.99, "class 0 dominant must map to a near-certain WIN for STM"
    with torch.no_grad():
        assert float(model.predict_value(x)[0]) == pytest.approx(value, abs=1e-7)


def test_loss_dominant_class_is_read_as_a_loss_for_the_side_to_move():
    model = _controlled_model(wdl_bias=(0.0, 0.0, 10.0))
    x = _planes_tensor()
    with torch.no_grad():
        _, wdl_logits = model(x)
    value = float(ChessNet.wdl_logits_to_value(wdl_logits.float())[0])
    assert value < -0.99, "class 2 dominant must map to a near-certain LOSS for STM"


def test_no_class_inversion_and_monotone_conversion():
    """As logit[win] - logit[loss] grows, the scalar value must grow strictly:
    no ordering flip anywhere in the range."""
    x = _planes_tensor()
    values = []
    for d in (-6.0, -2.0, 0.0, 2.0, 6.0):
        model = _controlled_model(wdl_bias=(d, 0.0, -d))
        with torch.no_grad():
            value = float(ChessNet.wdl_logits_to_value(model(x)[1].float())[0])
        values.append(value)
    assert values == sorted(values)
    assert values[0] < -0.99 and values[-1] > 0.99
    assert values[2] == pytest.approx(0.0, abs=1e-6)


def test_value_matches_win_minus_loss_probability_exactly():
    model = _controlled_model(wdl_bias=(1.5, 0.2, -0.7))
    x = _planes_tensor()
    with torch.no_grad():
        _, wdl_logits = model(x)
    probs = torch.softmax(wdl_logits.float(), dim=-1)
    expected = float(probs[0, 0] - probs[0, 2])
    assert float(ChessNet.wdl_logits_to_value(wdl_logits.float())[0]) == pytest.approx(
        expected, abs=1e-9
    )
    with torch.no_grad():
        assert float(model.predict_value(x)[0]) == pytest.approx(expected, abs=1e-7)


# ---------------------------------------------------------------------------
# Logits vs probabilities: the model must NOT softmax internally
# ---------------------------------------------------------------------------


def test_model_returns_raw_logits_not_probabilities():
    model = _controlled_model(wdl_bias=(10.0, 0.0, -10.0), policy_bias_index=0,
                             policy_bias_value=20.0)
    x = _planes_tensor()
    with torch.no_grad():
        p_logits, w_logits = model(x)
    # Raw logits keep their magnitude and do not sum to 1 or lie in [0, 1].
    assert float(p_logits[0, 0]) == pytest.approx(20.0, abs=0.5)
    assert float(w_logits[0, 0]) == pytest.approx(10.0, abs=0.5)
    assert not torch.isclose(p_logits.sum(), torch.tensor(1.0), atol=1e-3)
    assert not torch.isclose(w_logits.sum(), torch.tensor(1.0), atol=1e-3)


# ---------------------------------------------------------------------------
# Every consumer chain applies exactly one softmax and agrees
# ---------------------------------------------------------------------------


def test_cli_factory_chain_agrees_with_canonical_conversion():
    from python.cli.eval_callbacks import make_serial_eval_fn

    model = _controlled_model(wdl_bias=(2.0, 0.0, -2.0), policy_bias_index=7,
                              policy_bias_value=5.0)
    planes = zeropawn_cpp.encode_state(zeropawn_cpp.ChessGame(START_FEN))
    eval_fn = make_serial_eval_fn(model, "cpu", cpp_module=_StubCpp(planes))

    policy, value = eval_fn(zeropawn_cpp.ChessGame(START_FEN))

    x = torch.from_numpy(planes).unsqueeze(0)
    with torch.no_grad():
        p_logits, w_logits = model(x)
    expected_policy = torch.softmax(p_logits.float(), -1)[0].numpy()
    expected_value = float(torch.softmax(w_logits.float(), -1)[0, 0] - torch.softmax(
        w_logits.float(), -1
    )[0, 2])
    assert len(policy) == 4672
    assert np.allclose(np.asarray(policy), expected_policy, atol=1e-6)
    assert np.isclose(np.sum(policy), 1.0, atol=1e-5)
    assert value == pytest.approx(expected_value, abs=1e-6)
    assert value > 0, "win-dominant logits must yield a positive STM value"


def test_uci_callback_chain_agrees_with_canonical_conversion():
    from types import SimpleNamespace

    from python.uci.uci_engine import UCIEngine

    model = _controlled_model(wdl_bias=(-2.0, 0.0, 2.0))
    planes = zeropawn_cpp.encode_state(zeropawn_cpp.ChessGame(START_FEN))
    stub = SimpleNamespace(
        device="cpu", model=model, _lock=threading.Lock(), zeropawn_cpp=_StubCpp(planes)
    )

    policy, value = UCIEngine._eval_callback(stub, zeropawn_cpp.ChessGame(START_FEN))

    x = torch.from_numpy(planes).unsqueeze(0)
    with torch.no_grad():
        w_logits = model(x)[1]
    probs = torch.softmax(w_logits.float(), -1)[0]
    assert value == pytest.approx(float(probs[0] - probs[2]), abs=1e-6)
    assert value < 0, "loss-dominant logits must yield a negative STM value"
    assert len(policy) == 4672


def test_worker_aggregator_chain_applies_exactly_one_softmax():
    from python.selfplay.worker import WorkerBatchAggregator

    p_logits = np.zeros((2, 4672), dtype=np.float32)
    p_logits[0, 3] = 4.0
    p_logits[1, 10] = -3.0
    w_logits = np.array([[1.0, 0.0, -1.0], [-1.0, 0.0, 1.0]], dtype=np.float32)

    class _StubClient:
        def evaluate_leaves(self, states):
            return p_logits, w_logits, 1

    agg = WorkerBatchAggregator(client=_StubClient())
    results = []
    batch = []
    for _ in range(2):
        ev = threading.Event()
        slot = [None, None]
        batch.append((np.zeros((19, 8, 8), np.float32), slot, ev))
        results.append((ev, slot))
    agg._flush_batch(batch)

    expected_p = _softmax_np(p_logits)
    expected_w = _softmax_np(w_logits)
    for i, (ev, slot) in enumerate(results):
        assert ev.is_set()
        policy, value = slot
        assert np.allclose(policy, expected_p[i], atol=1e-6)
        assert value == pytest.approx(float(expected_w[i, 0] - expected_w[i, 2]), abs=1e-6)
    # Test power: a (wrong) second softmax would be measurably different.
    assert not np.allclose(_softmax_np(expected_p[0]), expected_p[0], atol=1e-3)


def test_full_softmax_then_legal_renormalization_equals_masked_softmax():
    """The C++ prior contract (mask + renormalize the consumer's softmax) is
    EXACTLY softmax over legal logits — one softmax total.  A second softmax
    would violate the identity, so this pins the single-softmax chain."""
    rng = np.random.default_rng(7)
    logits = rng.normal(0.0, 3.0, size=4672).astype(np.float64)
    legal = rng.choice(4672, size=30, replace=False)

    full = _softmax_np(logits)
    renorm = full[legal] / full[legal].sum()
    masked = _softmax_np(logits[legal])

    assert np.allclose(renorm, masked, atol=1e-9)
    # Test power: a (wrong) additional softmax applied to the probabilities
    # would measurably flatten the distribution.
    double = _softmax_np(renorm)
    assert not np.allclose(double, renorm, atol=1e-3), "test would not detect a double softmax"


# ---------------------------------------------------------------------------
# Side-to-move perspective through the encoder (no checkpoint needed)
# ---------------------------------------------------------------------------


def test_color_mirrored_positions_encode_identically():
    """Side-to-move perspective: 'White up a queen, White to move' and
    'Black up a queen, Black to move' are the same position from the mover's
    view and must encode identically."""
    white_up = zeropawn_cpp.ChessGame(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    black_up = zeropawn_cpp.ChessGame(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    )
    p_white = zeropawn_cpp.encode_state(white_up)
    p_black = zeropawn_cpp.encode_state(black_up)
    assert np.array_equal(p_white, p_black)
    # ...so any deterministic model must produce identical predictions.
    model = _controlled_model(wdl_bias=(1.0, 0.0, -1.0))
    with torch.no_grad():
        out_white = model(torch.from_numpy(p_white).unsqueeze(0))
        out_black = model(torch.from_numpy(p_black).unsqueeze(0))
    assert torch.equal(out_white[1], out_black[1])


# ---------------------------------------------------------------------------
# Real checkpoint (skip-if-absent): perspective on real weights
# ---------------------------------------------------------------------------


def _load_real_default():
    from python.cli.models import try_get_default_model_path

    path = try_get_default_model_path()
    if path is None:
        pytest.skip("no designated default checkpoint available in this tree")
    from python.cli.common import load_engine_model

    model, _ = load_engine_model(path, device="cpu")
    return model


def test_real_model_predictions_are_perspective_consistent():
    model = _load_real_default()
    white_up = zeropawn_cpp.ChessGame(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )
    black_up = zeropawn_cpp.ChessGame(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1"
    )
    with torch.no_grad():
        w1 = model(torch.from_numpy(zeropawn_cpp.encode_state(white_up)).unsqueeze(0))[1]
        w2 = model(torch.from_numpy(zeropawn_cpp.encode_state(black_up)).unsqueeze(0))[1]
    assert torch.equal(w1, w2), "mirrored STM positions must get identical WDL outputs"
    # And the interpretation is a real probability triple (single softmax).
    probs = torch.softmax(w1.float(), -1)[0]
    assert float(probs.sum()) == pytest.approx(1.0, abs=1e-6)


def test_real_model_extreme_lost_position_is_negative():
    """Bare king to move against king+queen: the STM value must be negative
    (a semantics/sign check that holds for any directionally-trained head)."""
    model = _load_real_default()
    lost = zeropawn_cpp.ChessGame("8/8/8/8/8/8/8/KQ5k b - - 0 1")  # Black to move, KQ vs K
    with torch.no_grad():
        value = float(
            ChessNet.wdl_logits_to_value(
                model(torch.from_numpy(zeropawn_cpp.encode_state(lost)).unsqueeze(0))[1].float()
            )[0]
        )
    assert value < 0.0, f"bare king to move against KQ must be negative, got {value}"
