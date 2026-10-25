"""Exhaustive Argument Combination Engine and Constraint Evaluator for ZeroPawn."""

from __future__ import annotations

import argparse
import itertools
import shlex
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

from python.cli.discovery import ArgumentMetadata, CommandCatalog, CommandNode, get_command_catalog


@dataclass
class ArgumentCombination:
    """A specific structural combination of arguments and options for a command."""

    index: int
    command_name: str
    command_path: List[str]
    parameters: Dict[str, Any]
    explicit_flags: Set[str]  # Names of arguments explicitly passed
    is_valid: bool
    rejection_reason: Optional[str] = None
    mode: Optional[str] = None
    parser_accepted: Optional[bool] = None
    parser_error: Optional[str] = None

    def to_argv(self, include_command: bool = True) -> List[str]:
        """Convert combination into an argument vector list for argparse or subprocess."""
        tokens: List[str] = []
        if include_command:
            tokens.extend(self.command_path)

        # Separate positional and flag arguments
        catalog = get_command_catalog()
        node = catalog.get_command(self.command_path)

        pos_tokens: List[str] = []
        flag_tokens: List[str] = []

        if node:
            # Process positionals in order
            for pos_arg in node.positional_arguments:
                val = self.parameters.get(pos_arg.name)
                if val is not None and str(val).strip() != "":
                    pos_tokens.append(str(val))

            # Process flags
            for arg in node.arguments:
                if arg.is_positional:
                    continue
                if arg.name in self.explicit_flags:
                    val = self.parameters.get(arg.name)
                    flag_str = arg.flags[0] if arg.flags else f"--{arg.name.replace('_', '-')}"
                    if arg.is_flag_only:
                        if val:
                            flag_tokens.append(flag_str)
                    else:
                        if val is not None:
                            flag_tokens.extend([flag_str, str(val)])
        else:
            for k, val in self.parameters.items():
                if k in self.explicit_flags:
                    flag_tokens.extend([f"--{k.replace('_', '-')}", str(val)])

        return tokens + flag_tokens + pos_tokens

    def to_command_line(self) -> str:
        """Render safely escaped and quoted command-line string."""
        tokens = ["zeropawn"] + self.to_argv(include_command=True)
        return " ".join(shlex.quote(tok) for tok in tokens)

    def summary_description(self) -> str:
        """Concise human-readable summary of active flags."""
        if not self.explicit_flags:
            return "Default configuration (zero explicit flags)"
        parts = []
        for k in sorted(self.explicit_flags):
            v = self.parameters.get(k)
            if v is True:
                parts.append(f"--{k.replace('_', '-')}")
            elif v is not None:
                parts.append(f"--{k.replace('_', '-')}={v}")
        return ", ".join(parts)


# Representative sample values for unrestricted argument types to enable safe structural testing
SAMPLE_VALUES: Dict[str, Any] = {
    "str": "canonical_sample",
    "int": 42,
    "float": 1.5,
    "path": "data/canonical_sample.bin",
    "checkpoint": "models/checkpoints_scaled/best_model.pt",
    "model_path": "models/checkpoints_scaled/best_model.pt",
    "model": "models/checkpoints_scaled/best_model.pt",
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "key": "training.batch_size",
    "value": "512",
    "preset_id": "fast-test",
    "run_id": "run-20261025-test",
    "output": "tmp/test_output.bin",
    "target": "tmp/cache_sample",
    "suite": "fast",
}


def _get_sample_value_for_argument(arg: ArgumentMetadata) -> Any:
    """Return a safe representative sample value for testing argument structures."""
    if arg.choices:
        return arg.choices[0]
    if arg.type == "bool":
        return True
    dest_lower = arg.name.lower()
    for k, sample in SAMPLE_VALUES.items():
        if k in dest_lower:
            return sample
    if arg.type == "int":
        if arg.min_val is not None:
            return int(arg.min_val)
        return 10
    if arg.type == "float":
        if arg.min_val is not None:
            return float(arg.min_val)
        return 0.5
    if arg.type == "path":
        return "data/test_sample.bin"
    return "sample_val"


def evaluate_combination_constraints(
    node: CommandNode,
    parameters: Dict[str, Any],
    explicit_flags: Set[str],
) -> Tuple[bool, Optional[str]]:
    """Evaluate all parser and semantic constraints for a specific parameter combination."""
    # 1. Required positional arguments check
    for pos in node.positional_arguments:
        if pos.required and (pos.name not in parameters or parameters[pos.name] is None):
            return False, f"Missing required positional argument: <{pos.name}>"

    # 2. Required flags check
    for req in node.required_arguments:
        if req.name not in explicit_flags and (
            req.name not in parameters or parameters[req.name] is None
        ):
            return False, f"Missing required option: {req.flags[0] if req.flags else req.name}"

    # 3. Model requirement check
    if node.requires_model == "required":
        has_model = False
        for m_name in ("checkpoint", "weights", "resume", "model_path", "model"):
            if m_name in explicit_flags or (
                m_name in parameters and parameters[m_name] is not None
            ):
                has_model = True
                break
        if not has_model:
            # Note: models export/inspect have model_path as positional
            if any(p.name in ("model_path", "model", "path") for p in node.positional_arguments):
                has_model = True
        if not has_model:
            return False, f"Command '{node.full_command}' requires a pre-trained model checkpoint"

    # 4. Check explicit ConstraintRules on CommandNode
    for rule in node.constraints:
        if rule.rule_type == "MUTUALLY_EXCLUSIVE":
            # If source_arg is present, none of target_args can be present
            source_active = (
                rule.source_arg in explicit_flags or parameters.get(rule.source_arg) is True
            )
            if source_active:
                for target in rule.target_args:
                    target_active = target in explicit_flags or (
                        parameters.get(target) is not None and parameters.get(target) is not False
                    )
                    if target_active:
                        return False, rule.description

        elif rule.rule_type == "REQUIRES":
            # If source_arg is present, target_args must also be present
            source_active = rule.source_arg in explicit_flags or (
                parameters.get(rule.source_arg) is not None
                and parameters.get(rule.source_arg) is not False
            )
            if source_active:
                for target in rule.target_args:
                    target_active = target in explicit_flags or (
                        parameters.get(target) is not None and parameters.get(target) is not False
                    )
                    if not target_active:
                        return False, rule.description

        elif rule.rule_type == "MODE_ONLY":
            # Argument is only valid if command mode equals specified mode
            source_active = rule.source_arg in explicit_flags
            if source_active and rule.mode:
                active_mode = parameters.get("mode", "rl")
                if parameters.get("supervised"):
                    active_mode = "supervised"
                if active_mode != rule.mode:
                    return False, rule.description

    return True, None


class SafeParserValidator:
    """Safe validator that intercepts argparse parse_args without sys.exit or printing to stdout/stderr."""

    @staticmethod
    def validate_tokens(tokens: List[str]) -> Tuple[bool, Optional[str]]:
        from python.cli.main import build_parser

        parser = build_parser()
        error_messages: List[str] = []

        class InterceptExit(Exception):
            pass

        def custom_exit(status=0, message=None):
            if message:
                error_messages.append(str(message))
            raise InterceptExit()

        def custom_error(message):
            error_messages.append(str(message))
            raise InterceptExit()

        parser.exit = custom_exit
        parser.error = custom_error

        # Also patch nested subparsers
        def patch_subparsers(p):
            p.exit = custom_exit
            p.error = custom_error
            for action in p._actions:
                if isinstance(action, argparse._SubParsersAction):
                    for sub_p in action.choices.values():
                        patch_subparsers(sub_p)

        patch_subparsers(parser)

        try:
            parser.parse_args(tokens)
            return True, None
        except InterceptExit:
            err = "; ".join(error_messages) if error_messages else "Rejected by parser constraints"
            return False, err
        except Exception as e:
            return False, str(e)


class CombinationEngine:
    """Deterministic, lazy combination generator and constraint validator."""

    def __init__(self, catalog: Optional[CommandCatalog] = None) -> None:
        self.catalog = catalog or get_command_catalog()

    def get_command_combination_space_size(self, node: CommandNode) -> Tuple[int, int, int]:
        """Compute (total_combinations, estimated_valid, estimated_invalid) for a command."""
        combos = list(self.enumerate_combinations(node, max_limit=10000, include_invalid=True))
        valid_count = sum(1 for c in combos if c.is_valid)
        invalid_count = len(combos) - valid_count
        return len(combos), valid_count, invalid_count

    def enumerate_combinations(
        self,
        node: CommandNode,
        max_limit: Optional[int] = None,
        include_invalid: bool = False,
        filter_flag: Optional[str] = None,
    ) -> Generator[ArgumentCombination, None, None]:
        """Deterministically and lazily generate combinations for a command node.

        For commands with finite option dimensions, exhaustively yields all states.
        For large command spaces (> 12 options), decomposes by mode / functional profile.
        """
        # Check if command has sub-modes (e.g. train has RL vs Supervised vs Verify)
        modes = self._determine_modes(node)

        idx = 1
        yielded_signatures: Set[str] = set()

        for mode_name, mode_args in modes:
            # Build dimension choices for this mode
            dimensions: List[Tuple[str, List[Tuple[bool, Any]]]] = []

            for arg in mode_args:
                if arg.is_positional:
                    sample = _get_sample_value_for_argument(arg)
                    if arg.required:
                        dimensions.append((arg.name, [(True, sample)]))
                    else:
                        dimensions.append((arg.name, [(False, None), (True, sample)]))
                elif arg.is_flag_only:
                    # Boolean flag: Absent (False) or Present (True)
                    dimensions.append((arg.name, [(False, False), (True, True)]))
                elif arg.choices:
                    # Finite choices: Absent (default) or Present with each choice
                    states: List[Tuple[bool, Any]] = []
                    if not arg.required:
                        states.append((False, arg.default))
                    for c in arg.choices:
                        states.append((True, c))
                    dimensions.append((arg.name, states))
                else:
                    # Valued option: Absent (default) or Present (sample value)
                    sample = _get_sample_value_for_argument(arg)
                    if arg.required:
                        dimensions.append((arg.name, [(True, sample)]))
                    else:
                        dimensions.append((arg.name, [(False, arg.default), (True, sample)]))

            # If dimensions are small (<= 10), do cartesian product
            # If dimensions are larger, generate mode-tailored orthogonal array + individual toggles
            if len(dimensions) <= 10:
                dim_names = [d[0] for d in dimensions]
                dim_values = [d[1] for d in dimensions]

                for combo_vals in itertools.product(*dim_values):
                    params: Dict[str, Any] = {}
                    explicit: Set[str] = set()

                    for name, (is_explicit, val) in zip(dim_names, combo_vals):
                        params[name] = val
                        if is_explicit:
                            explicit.add(name)

                    # Set mode attribute if applicable
                    if mode_name == "supervised":
                        params["supervised"] = True
                        params["mode"] = "supervised"
                        explicit.add("supervised")
                    elif mode_name == "rl":
                        params["mode"] = "rl"
                    elif mode_name == "verify":
                        params["verify"] = True
                        explicit.add("verify")

                    # Check filter
                    if filter_flag and filter_flag not in explicit:
                        continue

                    # Uniqueness signature
                    sig = f"{mode_name}:" + ",".join(sorted(f"{k}={params[k]}" for k in explicit))
                    if sig in yielded_signatures:
                        continue
                    yielded_signatures.add(sig)

                    is_valid, reason = evaluate_combination_constraints(node, params, explicit)

                    if not is_valid and not include_invalid:
                        continue

                    combo = ArgumentCombination(
                        index=idx,
                        command_name=node.name,
                        command_path=node.path,
                        parameters=params,
                        explicit_flags=explicit,
                        is_valid=is_valid,
                        rejection_reason=reason,
                        mode=mode_name,
                    )
                    idx += 1
                    yield combo

                    if max_limit and idx > max_limit:
                        return
            else:
                # Mode-based partitioned enumeration for commands with many options (>10)
                # 1. Base default combination for mode
                # 2. Single-flag variations (toggle each flag on)
                # 3. Pairwise interaction combinations between core flags
                # 4. Multi-flag standard configurations
                for combo in self._generate_partitioned_combinations(
                    node=node,
                    mode_name=mode_name,
                    mode_args=mode_args,
                    start_idx=idx,
                    yielded_signatures=yielded_signatures,
                    include_invalid=include_invalid,
                    filter_flag=filter_flag,
                ):
                    idx += 1
                    yield combo
                    if max_limit and idx > max_limit:
                        return

    def _determine_modes(self, node: CommandNode) -> List[Tuple[str, List[ArgumentMetadata]]]:
        """Decompose complex commands into mode-specific argument subsets."""
        cmd_path = " ".join(node.path)
        command_args = [a for a in node.arguments if not a.is_inherited]

        if cmd_path == "train":
            # Decompose train into 4 distinct modes:
            rl_args = [
                a
                for a in command_args
                if a.name
                in (
                    "max_hours",
                    "concurrent_games",
                    "full_sims",
                    "fast_sims",
                    "batch_size",
                    "lr",
                    "checkpoint",
                    "random_init",
                    "device",
                    "buffer_capacity",
                    "cleanup_buffer_on_exit",
                    "keep_checkpoints",
                    "tablebase_path",
                    "watchdog_rss_gb",
                )
            ]
            sup_args = [
                a
                for a in command_args
                if a.name
                in (
                    "supervised",
                    "train_dir",
                    "val_dir",
                    "epochs",
                    "max_steps",
                    "batch_size",
                    "micro_batch_size",
                    "gradient_accumulation_steps",
                    "lr",
                    "warmup_steps",
                    "no_gpu_resident",
                    "workers",
                    "num_blocks",
                    "num_filters",
                    "tactics_ratio",
                    "tactics_dir",
                    "seed",
                    "checkpoint",
                )
            ]
            verify_args = [a for a in command_args if a.name in ("verify", "checkpoint", "device")]
            hw_args = [
                a for a in command_args if a.name in ("auto_hardware", "calibrate", "device")
            ]

            return [
                ("rl", rl_args),
                ("supervised", sup_args),
                ("verify", verify_args),
                ("hardware", hw_args),
            ]

        if cmd_path == "match":
            return [
                (
                    "run",
                    [
                        a
                        for a in command_args
                        if a.name in ("elo", "mcts_sims", "engine_color", "games", "checkpoint")
                    ],
                ),
                (
                    "live",
                    [
                        a
                        for a in command_args
                        if a.name in ("engine_color", "delay", "mcts_sims", "checkpoint")
                    ],
                ),
                (
                    "human",
                    [
                        a
                        for a in command_args
                        if a.name in ("human_color", "mcts_sims", "checkpoint")
                    ],
                ),
            ]

        if cmd_path == "puzzles":
            return [
                (
                    "eval",
                    [
                        a
                        for a in command_args
                        if a.name in ("csv", "n", "mcts_sims", "checkpoint", "threads")
                    ],
                ),
                (
                    "solve",
                    [
                        a
                        for a in command_args
                        if a.name in ("csv", "puzzle_id", "human_guess", "checkpoint")
                    ],
                ),
                (
                    "learn",
                    [
                        a
                        for a in command_args
                        if a.name in ("csv", "epochs", "batch_size", "lr", "checkpoint")
                    ],
                ),
            ]

        return [("default", command_args)]

    def _generate_partitioned_combinations(
        self,
        node: CommandNode,
        mode_name: str,
        mode_args: List[ArgumentMetadata],
        start_idx: int,
        yielded_signatures: Set[str],
        include_invalid: bool,
        filter_flag: Optional[str],
    ) -> Generator[ArgumentCombination, None, None]:
        """Generate combinations for high-dimensional modes using single, pairwise, and cluster variations."""
        idx = start_idx

        # 1. Base configuration (defaults for mode)
        base_params: Dict[str, Any] = {a.name: a.default for a in mode_args}
        base_explicit: Set[str] = set()
        if mode_name == "supervised":
            base_params["supervised"] = True
            base_params["mode"] = "supervised"
            base_explicit.add("supervised")
        elif mode_name == "verify":
            base_params["verify"] = True
            base_explicit.add("verify")

        for pos in [a for a in mode_args if a.is_positional and a.required]:
            base_params[pos.name] = _get_sample_value_for_argument(pos)
            base_explicit.add(pos.name)

        sig = f"{mode_name}:base:" + ",".join(sorted(base_explicit))
        if sig not in yielded_signatures:
            yielded_signatures.add(sig)
            is_valid, reason = evaluate_combination_constraints(node, base_params, base_explicit)
            if is_valid or include_invalid:
                if not filter_flag or filter_flag in base_explicit:
                    yield ArgumentCombination(
                        index=idx,
                        command_name=node.name,
                        command_path=node.path,
                        parameters=dict(base_params),
                        explicit_flags=set(base_explicit),
                        is_valid=is_valid,
                        rejection_reason=reason,
                        mode=mode_name,
                    )
                    idx += 1

        # 2. Single-flag variations: activate each argument individually
        for target_arg in mode_args:
            sample_val = _get_sample_value_for_argument(target_arg)
            params = dict(base_params)
            params[target_arg.name] = sample_val
            explicit = set(base_explicit) | {target_arg.name}

            sig = f"{mode_name}:single:{target_arg.name}"
            if sig in yielded_signatures:
                continue
            yielded_signatures.add(sig)

            is_valid, reason = evaluate_combination_constraints(node, params, explicit)
            if is_valid or include_invalid:
                if not filter_flag or filter_flag in explicit:
                    yield ArgumentCombination(
                        index=idx,
                        command_name=node.name,
                        command_path=node.path,
                        parameters=params,
                        explicit_flags=explicit,
                        is_valid=is_valid,
                        rejection_reason=reason,
                        mode=mode_name,
                    )
                    idx += 1

        # 3. Pairwise interactions between important flags
        flag_names = [a.name for a in mode_args if not a.is_positional]
        for a1, a2 in itertools.combinations(flag_names[:12], 2):
            params = dict(base_params)
            arg1 = node.get_argument(a1)
            arg2 = node.get_argument(a2)
            if not arg1 or not arg2:
                continue

            params[a1] = _get_sample_value_for_argument(arg1)
            params[a2] = _get_sample_value_for_argument(arg2)
            explicit = set(base_explicit) | {a1, a2}

            sig = f"{mode_name}:pair:{a1}+{a2}"
            if sig in yielded_signatures:
                continue
            yielded_signatures.add(sig)

            is_valid, reason = evaluate_combination_constraints(node, params, explicit)
            if is_valid or include_invalid:
                if not filter_flag or filter_flag in explicit:
                    yield ArgumentCombination(
                        index=idx,
                        command_name=node.name,
                        command_path=node.path,
                        parameters=params,
                        explicit_flags=explicit,
                        is_valid=is_valid,
                        rejection_reason=reason,
                        mode=mode_name,
                    )
                    idx += 1

    def get_paginated_combinations(
        self,
        node: CommandNode,
        page: int = 1,
        page_size: int = 20,
        filter_query: Optional[str] = None,
        include_invalid: bool = False,
    ) -> Tuple[List[ArgumentCombination], int, int]:
        """Return (items_on_page, total_items, total_pages)."""
        all_combos: List[ArgumentCombination] = []
        filter_flag = filter_query.lstrip("-").replace("-", "_") if filter_query else None

        for combo in self.enumerate_combinations(
            node,
            max_limit=1000,
            include_invalid=include_invalid,
            filter_flag=filter_flag,
        ):
            if filter_query:
                q = filter_query.lower()
                cmd_line = combo.to_command_line().lower()
                summary = combo.summary_description().lower()
                if q not in cmd_line and q not in summary:
                    continue
            all_combos.append(combo)

        total_items = len(all_combos)
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        page = max(1, min(page, total_pages))

        start = (page - 1) * page_size
        end = start + page_size
        return all_combos[start:end], total_items, total_pages


# Cached singleton
_GLOBAL_ENGINE: Optional[CombinationEngine] = None


def get_combination_engine() -> CombinationEngine:
    """Retrieve or build the cached singleton CombinationEngine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = CombinationEngine()
    return _GLOBAL_ENGINE
