"""Hardening tests (Stage 1b, CPU-only): determinism, kill-and-resume,
replay sampler lock, inference hot-reload lock, puzzle-decline noise band."""

from __future__ import annotations

import importlib.util
import threading

import numpy as np
import pytest
import torch

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.selfplay.worker import SavedPosition
from python.selfplay.replay_buffer import ReplayBuffer, GpuResidentReplayCache
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer


def _rl_scripts():
    spec = importlib.util.spec_from_file_location("run_phase4_rl_mod", "scripts/run_phase4_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# 1. Determinism: same seed => identical visit distributions (no model).
def test_seeded_search_deterministic():
    def run():
        cfg = zeropawn_cpp.MCTSConfig()
        cfg.add_noise = False
        cfg.seed = 20260613
        mcts = zeropawn_cpp.MCTS(cfg)
        res = mcts.search(zeropawn_cpp.ChessGame(), 150, lambda g: ([1.0 / 4672] * 4672, 0.0))
        return res.best_action, list(res.action_visits)

    b1, v1 = run()
    b2, v2 = run()
    assert b1 == b2
    assert v1 == v2


def _dummy_pos(i: int) -> SavedPosition:
    planes = np.zeros((19, 8, 8), dtype=np.float32)
    planes[0, 0, 0] = float(i)
    visits = np.zeros(4672, dtype=np.float32)
    visits[i % 4672] = 1.0
    return SavedPosition(
        planes=planes, visits=visits, outcome=i % 3, is_full_search=True, model_version=1, game_id=i
    )


# 2. Kill-and-resume on CPU: trainer step/params + buffer size survive.
def test_kill_and_resume_cpu(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    buf.add_positions([_dummy_pos(i) for i in range(40)])
    model = ChessNet(num_blocks=2, num_filters=16)
    trainer = RLTrainer(
        model=model,
        replay_buffer=buf,
        device="cpu",
        batch_size=8,
        push_freq=1000,
        checkpoint_dir=tmp_path / "ckpt",
        sync_weights_path=tmp_path / "live.pt",
    )
    trainer.train_step()
    trainer.train_step()
    assert trainer.step_num == 2
    before_params = {k: v.clone() for k, v in model.state_dict().items()}
    saved = trainer.save_checkpoint()
    del trainer
    del model

    buf2 = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=False)
    assert len(buf2) == 40
    model2 = ChessNet(num_blocks=2, num_filters=16)
    trainer2 = RLTrainer(
        model=model2,
        replay_buffer=buf2,
        device="cpu",
        batch_size=8,
        push_freq=1000,
        checkpoint_dir=tmp_path / "ckpt",
        sync_weights_path=tmp_path / "live.pt",
    )
    step = trainer2.load_checkpoint(saved)
    assert step == 2
    assert trainer2.step_num == 2
    for k, v in model2.state_dict().items():
        assert torch.equal(v, before_params[k])
    trainer2.train_step()
    assert trainer2.step_num == 3


# 3. Sampler lock: concurrent writers + samplers stay valid (CPU).
def test_sampler_lock_concurrent_cpu(tmp_path):
    buf = ReplayBuffer(buffer_dir=tmp_path / "buf", capacity=64, create_new=True)
    buf.add_positions([_dummy_pos(i) for i in range(64)])
    cache = GpuResidentReplayCache(capacity=32, device="cpu")
    # Prefill so samplers never hit an empty cache at thread start.
    cache.add_positions(
        np.zeros((32, 19, 8, 8), np.float32),
        np.zeros((32, 4672), np.float32),
        np.zeros(32, dtype=np.int64),
        np.ones(32, dtype=np.bool_),
    )
    errors: list = []

    def writer(n):
        try:
            for i in range(n):
                buf.add_position(_dummy_pos(1000 + i))
                cache.add_positions(
                    np.zeros((1, 19, 8, 8), np.float32),
                    np.zeros((1, 4672), np.float32),
                    np.array([i % 3]),
                    np.array([True]),
                )
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def sampler(n):
        try:
            for _ in range(n):
                b, _ = buf.sample_batch(16)
                assert b["planes"].shape == (16, 19, 8, 8)
                assert set(np.unique(b["outcomes"])) <= {0, 1, 2}
                bt, _ = buf.sample_batch_tensors(16, device="cpu")
                assert bt["planes"].shape == (16, 19, 8, 8)
                c = cache.sample_batch(8)
                assert c["planes"].shape == (8, 19, 8, 8)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(60,)) for _ in range(2)] + [
        threading.Thread(target=sampler, args=(60,)) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert errors == []
    assert buf.total_added == 64 + 120


# 4. Hot-reload lock: predicts during reloads never tear (CPU, tiny net).
def test_hot_reload_lock_cpu(tmp_path):
    server = InferenceServer(model=ChessNet(num_blocks=2, num_filters=16), device="cpu")
    ckpts = []
    for v in range(2):
        m = ChessNet(num_blocks=2, num_filters=16)
        with torch.no_grad():
            for p in m.parameters():
                p.fill_(float(v + 1))
        p = tmp_path / f"w{v}.pt"
        torch.save({"model": m.state_dict()}, p)
        ckpts.append(str(p))
    errors: list = []
    x = np.zeros((4, 19, 8, 8), dtype=np.float32)

    def predictor():
        try:
            for _ in range(30):
                pol, wdl, _ = server.predict_batch(x)
                assert pol.shape == (4, 4672) and wdl.shape == (4, 3)
                assert np.isfinite(pol).all()
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def reloader():
        try:
            for i in range(10):
                server.load_checkpoint(ckpts[i % 2])
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=predictor) for _ in range(2)]
    ts.append(threading.Thread(target=reloader))
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=300)
    assert errors == []
    assert server.model_version == 10


# 5. Noise band: jitter no longer trips; real slides still trip.
def test_puzzle_decline_noise_band():
    mod = _rl_scripts()
    new = mod.puzzle_decline_trips

    def old(history):
        if len(history) < 4:
            return False
        a, b, c, d = history[-4:]
        return b < a and c < b and d < c

    jitter = [0.5000, 0.4996, 0.4992, 0.4988]
    assert old(jitter) is True  # before: trips on noise
    assert new(jitter) is False  # after: band absorbs it
    slide = [0.50, 0.49, 0.48, 0.47]
    assert old(slide) is True and new(slide) is True
    flat = [0.50, 0.50, 0.50, 0.50]
    assert old(flat) is False and new(flat) is False
    assert new([0.5, 0.4]) is False  # short history never trips
    # Boundary: drop of exactly the band does not count (strictly greater).
    edge = [0.50, 0.496, 0.492, 0.488]
    assert new(edge, band=0.005) is False
    assert new([0.50, 0.493, 0.486, 0.479], band=0.005) is True
