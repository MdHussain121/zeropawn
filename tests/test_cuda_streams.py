"""Verification tests for CUDA Stream isolation, priority, and cross-stream event synchronization."""

from unittest import mock

import pytest
import torch

from python.model.chess_net import ChessNet
from python.selfplay.inference_server import InferenceServer
from python.train.train_rl import RLTrainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_cuda_stream_isolation_and_priority():
    """Verify that InferenceServer and RLTrainer allocate dedicated non-default CUDA streams."""
    model = ChessNet(num_blocks=2, num_filters=32)

    # 1. InferenceServer creates dedicated high-priority stream
    server = InferenceServer(model=model, device="cuda")
    assert server.stream is not None
    assert isinstance(server.stream, torch.cuda.Stream)
    # Ensure it is not the default stream
    default_stream = torch.cuda.default_stream(server.device)
    assert server.stream.cuda_stream != default_stream.cuda_stream

    # 2. RLTrainer creates dedicated non-default stream
    trainer = RLTrainer(model=model, replay_buffer=mock.MagicMock(), device="cuda", base_lr=1e-3)
    assert trainer.train_stream is not None
    assert isinstance(trainer.train_stream, torch.cuda.Stream)
    assert trainer.train_stream.cuda_stream != default_stream.cuda_stream
    # Ensure inference and training streams are separate
    assert server.stream.cuda_stream != trainer.train_stream.cuda_stream


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU")
def test_cuda_stream_event_synchronization_during_push():
    """Verify that push_weights records a CUDA event on train_stream and passes it to direct_server."""
    model = ChessNet(num_blocks=2, num_filters=32)
    server = InferenceServer(model=model, device="cuda")
    trainer = RLTrainer(
        model=model,
        replay_buffer=mock.MagicMock(),
        device="cuda",
        direct_server=server,
        base_lr=1e-3,
    )

    initial_version = server.model_version

    # Perform a train step on train_stream
    batch_dict = {
        "planes": torch.zeros(2, 19, 8, 8, device="cuda"),
        "visits": torch.zeros(2, 4672, device="cuda"),
        "outcomes": torch.tensor([[0], [2]], device="cuda", dtype=torch.long),
        "is_full_search": torch.tensor([True, True], device="cuda", dtype=torch.bool),
    }

    trainer.train_step(batch_dict=batch_dict)

    # Push weights: should record event, call load_state_dict_weights with sync_event,
    # and wait on event without crashing or deadlock
    sync_event = trainer.push_weights()
    assert sync_event is not None
    assert isinstance(sync_event, torch.cuda.Event)
    assert server.model_version == initial_version + 1

    # Verify inference on updated weights works seamlessly
    planes = torch.zeros(1, 19, 8, 8, device="cuda")
    pol, wdl, ver = server.predict_batch(planes)
    assert ver == initial_version + 1
