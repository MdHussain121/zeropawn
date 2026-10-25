import pytest
import torch
from python.model.chess_net import ChessNet


def test_chess_net_parameter_count():
    model_10 = ChessNet(in_channels=19, num_filters=128, num_blocks=10, num_actions=4672)
    params_10 = model_10.count_parameters()
    print(f"Total parameters (10 blocks): {params_10:,}")
    assert 5_000_000 <= params_10 <= 10_000_000, f"Param count {params_10} outside [5M, 10M]"

    model_12 = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672)
    params_12 = model_12.count_parameters()
    print(f"Total parameters (12 blocks): {params_12:,}")
    assert 5_000_000 <= params_12 <= 10_000_000, f"Param count {params_12} outside [5M, 10M]"


def test_chess_net_output_shapes():
    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672)
    model.eval()

    batch_size = 4
    x = torch.randn(batch_size, 19, 8, 8)
    with torch.no_grad():
        policy, wdl = model(x)

    assert policy.shape == (batch_size, 4672), (
        f"Policy shape {policy.shape} != ({batch_size}, 4672)"
    )
    assert wdl.shape == (batch_size, 3), f"WDL shape {wdl.shape} != ({batch_size}, 3)"

    # Test search value calculation
    search_val = model.predict_value(x)
    assert search_val.shape == (batch_size,)
    assert (search_val >= -1.0).all() and (search_val <= 1.0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_chess_net_peak_vram_batch_256_bfloat16():
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ChessNet(in_channels=19, num_filters=128, num_blocks=12, num_actions=4672)
    model = model.to(device=device, memory_format=torch.channels_last)

    batch_size = 256
    x = torch.randn(batch_size, 19, 8, 8, device=device).to(memory_format=torch.channels_last)

    # Perform forward + backward pass under bfloat16 autocast to measure peak training VRAM
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    target_p = torch.randint(0, 4672, (batch_size,), device=device)
    target_v = torch.randn(batch_size, 3, device=device).softmax(dim=-1)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        policy, wdl = model(x)
        loss = torch.nn.functional.cross_entropy(
            policy, target_p
        ) + 0.5 * torch.nn.functional.cross_entropy(wdl, target_v)

    loss.backward()
    optimizer.step()

    peak_bytes = torch.cuda.max_memory_allocated()
    peak_gb = peak_bytes / (1024**3)
    print(
        f"\n[VRAM TEST] Peak memory allocated at batch 256 (bfloat16): {peak_gb:.3f} GB ({peak_bytes:,} bytes)"
    )

    # Must be under 6 GB as mandated by the Contract
    assert peak_gb < 6.0, f"Peak memory {peak_gb:.2f} GB exceeds 6 GB limit!"


def test_checkpoint_architecture_and_autodetection(tmp_path):
    import os
    from python.selfplay.inference_server import InferenceServer

    default_model = ChessNet()
    assert default_model.num_blocks == 12
    assert default_model.num_filters == 128
    assert default_model.count_parameters() == 8_624_931
    assert sum(p.numel() for p in default_model.state_dict().values()) == 8_631_454

    server = InferenceServer()

    candidate_paths = [
        "models/checkpoints_scaled/best_model.pt",
        "models/checkpoints_rl/latest_checkpoint.pt",
        "models/checkpoints_rl/phase4_final_checkpoint.pt",
    ]
    test_paths = [p for p in candidate_paths if os.path.isfile(p)]
    if not test_paths:
        fixture_path = tmp_path / "fixture_checkpoint.pt"
        torch.save(
            {
                "model": default_model.state_dict(),
                "num_blocks": 12,
                "num_filters": 128,
                "step": 0,
            },
            fixture_path,
        )
        test_paths = [str(fixture_path)]

    for ckpt_path in test_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

        block_indices = {
            int(k.split(".")[1])
            for k in sd.keys()
            if k.startswith("blocks.") and k.split(".")[1].isdigit()
        }
        assert len(block_indices) == 12
        assert min(block_indices) == 0 and max(block_indices) == 11

        stem_weight = sd["stem_conv.weight"]
        assert stem_weight.shape == (128, 19, 3, 3)

        total_sd_elements = sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))
        assert total_sd_elements == 8_631_454

        auto_model = server._create_model_from_checkpoint(ckpt_path)
        assert auto_model.num_blocks == default_model.num_blocks == 12
        assert auto_model.num_filters == default_model.num_filters == 128
        assert auto_model.count_parameters() == default_model.count_parameters() == 8_624_931
        assert sum(p.numel() for p in auto_model.state_dict().values()) == 8_631_454
