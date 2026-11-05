"""Contract test for the ``--max-wait-ms`` CLI flag (SA5-02 remediation).

``--max-wait-ms`` is ACCEPTED BUT INERT by design (audit S4-08): it feeds
``Phase4Orchestrator.max_wait_ms``, which no runtime path consumes — the
InferenceServer's own wait knob is legacy-SHM-only and the production
batch-dispatch wait is governed by the self-play worker's internal
``WorkerBatchAggregator`` timeout (``SelfPlayWorker.max_wait_ms``, 1.5 ms
default). The flag is kept for interface stability, and the CLI/help surface
documents the no-op honestly (audit SA5-02).

This test pins the documented contract from both sides, so any future change
(wiring the flag into the worker's aggregator, or removing the flag) must
update the contract surface and this test together:

- the flag still parses (compatibility) and its help says it is ignored;
- the config registry entry says the same;
- the orchestrator accepts the parameter but never consumes it;
- the live dispatcher timeout default (1.5 ms) is unchanged.
"""

from __future__ import annotations

import argparse
import inspect

from python.cli.commands_train import register_parser
from python.cli.config import CONFIG_REGISTRY
from python.selfplay.worker import SelfPlayWorker


def _train_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="zeropawn")
    sub = parser.add_subparsers(dest="command")
    register_parser(sub)
    return next(a for a in sub.choices.values() if a.prog.endswith("train"))


def test_max_wait_ms_still_parses_for_compatibility() -> None:
    tp = _train_parser()
    args = tp.parse_args(["--max-wait-ms", "7.5"])
    assert args.max_wait_ms == 7.5
    args = tp.parse_args([])
    assert args.max_wait_ms == 1.0


def test_max_wait_ms_help_documents_the_no_op() -> None:
    tp = _train_parser()
    action = next(a for a in tp._actions if "--max-wait-ms" in a.option_strings)
    assert action.help is not None
    assert "IGNORED" in action.help
    assert "no-op" in action.help


def test_max_wait_ms_config_field_documents_the_no_op() -> None:
    field = CONFIG_REGISTRY["selfplay.max_wait_ms"]
    assert "--max-wait-ms" in field.cli_flags  # compatibility surface kept
    assert field.default == 1.0
    assert "IGNORED" in field.description
    assert field.runtime_dest is not None
    assert "not consumed" in field.runtime_dest


def test_phase4_orchestrator_accepts_but_never_consumes_max_wait_ms() -> None:
    from scripts.run_phase4_rl import Phase4Orchestrator

    params = inspect.signature(Phase4Orchestrator.__init__).parameters
    assert "max_wait_ms" in params  # accepted for interface stability
    source = inspect.getsource(Phase4Orchestrator)
    # Inert by contract (audit S4-08/SA5-02): never stored on self, so it
    # cannot reach the worker's aggregator or the inference server. If this
    # fails, the flag was (partially) wired — update the CLI help, the config
    # registry description and this test together.
    assert "self.max_wait_ms" not in source


def test_worker_dispatcher_timeout_is_the_live_knob() -> None:
    # The production batch-dispatch wait lives here, not in --max-wait-ms.
    default = inspect.signature(SelfPlayWorker.__init__).parameters["max_wait_ms"].default
    assert default == 1.5
