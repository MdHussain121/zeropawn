"""Dynamic Parser Introspection, Metadata Extraction, and Command Catalog for ZeroPawn."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from python.cli.operations import CATEGORIES


@dataclass
class ArgumentMetadata:
    """Complete specification of a CLI argument or option extracted from argparse."""

    name: str  # internal dest name, e.g. "batch_size", "supervised"
    flags: List[str]  # e.g. ["--batch-size"], ["-s", "--supervised"], [] for positional
    is_positional: bool
    type: str  # "bool", "int", "float", "str", "choice", "path"
    default: Any
    required: bool
    choices: Optional[List[str]] = None
    help: str = ""
    action_type: str = "store"  # "store", "store_true", "store_false", "append", etc.
    metavar: Optional[str] = None
    nargs: Optional[Union[int, str]] = None
    is_inherited: bool = False  # True for global flags inherited from root parser
    min_val: Optional[Union[int, float]] = None
    max_val: Optional[Union[int, float]] = None

    @property
    def is_flag_only(self) -> bool:
        """Return True if this argument is a boolean flag taking no value."""
        return self.type == "bool" or self.action_type in ("store_true", "store_false", "count")

    @property
    def display_name(self) -> str:
        """Canonical display name for UI."""
        if self.flags:
            return "/".join(self.flags)
        return f"<{self.name}>"

    def format_help(self) -> str:
        """Formatted help string with default and constraints."""
        parts = [self.help.strip()] if self.help else []
        if self.choices:
            parts.append(f"Choices: [{', '.join(str(c) for c in self.choices)}]")
        if self.default is not None and self.default != argparse.SUPPRESS:
            parts.append(f"Default: {self.default}")
        if self.required:
            parts.append("(REQUIRED)")
        return " | ".join(parts)


@dataclass
class ConstraintRule:
    """A semantic or parser-level constraint between arguments."""

    rule_type: str  # "MUTUALLY_EXCLUSIVE", "REQUIRES", "MODE_ONLY", "RECOMMENDED_WITH"
    source_arg: str  # Argument that triggers or is subject to constraint
    target_args: List[str]  # Related arguments
    description: str  # Human-readable explanation of why this combination is valid/invalid
    mode: Optional[str] = None  # If applicable only in a specific mode (e.g. "supervised", "rl")


@dataclass
class CommandNode:
    """A command or subcommand node in the ZeroPawn CLI hierarchy."""

    name: str  # e.g. "download", "selfplay", "train"
    path: List[str]  # e.g. ["data", "download"], ["train"]
    full_command: str  # e.g. "zeropawn data download"
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    help: str = ""
    category_id: int = 10
    category_name: str = "Maintenance & Utilities"
    classification: str = (
        "standard"  # "standard", "advanced", "deprecated", "experimental", "internal"
    )
    requires_model: str = "none"  # "none", "optional", "required"
    arguments: List[ArgumentMetadata] = field(default_factory=list)
    subcommands: Dict[str, "CommandNode"] = field(default_factory=dict)
    is_leaf: bool = False
    func: Optional[Callable] = None
    constraints: List[ConstraintRule] = field(default_factory=list)

    @property
    def positional_arguments(self) -> List[ArgumentMetadata]:
        return [a for a in self.arguments if a.is_positional and not a.is_inherited]

    @property
    def required_arguments(self) -> List[ArgumentMetadata]:
        return [a for a in self.arguments if a.required and not a.is_inherited]

    @property
    def optional_arguments(self) -> List[ArgumentMetadata]:
        return [
            a
            for a in self.arguments
            if not a.required and not a.is_positional and not a.is_inherited
        ]

    @property
    def inherited_arguments(self) -> List[ArgumentMetadata]:
        return [a for a in self.arguments if a.is_inherited]

    @property
    def finite_choice_arguments(self) -> List[ArgumentMetadata]:
        return [a for a in self.arguments if a.choices and not a.is_inherited]

    @property
    def boolean_flags(self) -> List[ArgumentMetadata]:
        return [a for a in self.arguments if a.is_flag_only and not a.is_inherited]

    def get_argument(self, name_or_flag: str) -> Optional[ArgumentMetadata]:
        """Find an argument by its dest name or flag string."""
        clean = name_or_flag.lstrip("-").replace("-", "_")
        for a in self.arguments:
            if a.name == clean or a.name == name_or_flag:
                return a
            if name_or_flag in a.flags:
                return a
        return None


# Canonical mapping from top-level command paths to ZeroPawn 10 functional domains
CATEGORY_MAPPINGS: Dict[str, int] = {
    "train": 1,
    "train connect4": 1,
    "supervised": 1,
    "finetune": 1,
    "selfplay": 1,
    "data": 2,
    "add-puzzles": 2,
    "models": 3,
    "match": 4,
    "eval": 4,
    "bench": 5,
    "profile": 5,
    "puzzles": 6,
    "tablebase": 6,
    "book": 6,
    "search": 6,
    "test": 7,
    "presets": 8,
    "config": 8,
    "status": 9,
    "diagnose": 9,
    "runs": 9,
    "logs": 9,
    "hardware": 9,
    "cleanup": 10,
    "uci": 10,
    "build": 10,
    "interactive": 10,
}

# Known model requirements
MODEL_REQUIREMENTS: Dict[str, str] = {
    "train": "optional",
    "supervised": "optional",
    "finetune": "required",
    "match run": "optional",
    "match live": "optional",
    "match human": "optional",
    "eval baseline": "optional",
    "eval heldout": "optional",
    "eval reliability": "optional",
    "eval connect4": "optional",
    "bench selfplay": "optional",
    "bench batch-sizes": "optional",
    "puzzles eval": "required",
    "puzzles solve": "required",
    "puzzles learn": "required",
    "models inspect": "optional",
    "models compare": "required",
    "models validate": "required",
    "models export": "required",
    "models check-compat": "required",
    "search": "optional",
    "uci": "optional",
    "book evaluate": "optional",
    "book train": "optional",
}

# Known command classifications
COMMAND_CLASSIFICATIONS: Dict[str, str] = {
    "chess": "deprecated",
    "diagnose lag": "experimental",
    "diagnose cache": "experimental",
    "diagnose value-signs": "experimental",
    "diagnose resign": "experimental",
    "bench sweep-concurrency": "advanced",
    "bench sweep-timeout": "advanced",
    "bench batch-sweep": "advanced",
    "bench concurrency-sweep": "advanced",
    "profile trace": "advanced",
    "profile memory": "advanced",
    "profile pacing": "advanced",
    "profile cuda": "advanced",
    "data check-leakage": "advanced",
    "data clean-tmp": "internal",
    "build compat": "internal",
}


def _infer_argument_type(action: argparse.Action) -> str:
    """Infer high-level argument type from argparse action attributes."""
    if action.choices:
        return "choice"
    if isinstance(
        action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._CountAction)
    ):
        return "bool"
    if action.type is int:
        return "int"
    if action.type is float:
        return "float"
    if action.type is not None:
        type_name = getattr(action.type, "__name__", str(action.type)).lower()
        if "int" in type_name:
            return "int"
        if "float" in type_name:
            return "float"
        if "path" in type_name or "file" in type_name:
            return "path"

    # Infer from dest name
    dest_lower = action.dest.lower()
    if any(k in dest_lower for k in ("path", "dir", "file", "csv", "pgn", "checkpoint", "weights")):
        return "path"
    return "str"


def _extract_constraints_for_command(
    cmd_path: str, args: List[ArgumentMetadata]
) -> List[ConstraintRule]:
    """Derive semantic constraint rules for a specific command."""
    rules: List[ConstraintRule] = []
    arg_names = {a.name for a in args}

    # 1. Train command constraints
    if cmd_path in ("train", "supervised"):
        if "checkpoint" in arg_names and "random_init" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="MUTUALLY_EXCLUSIVE",
                    source_arg="random_init",
                    target_args=["checkpoint", "resume"],
                    description="--random-init initializes new random weights and cannot be combined with --checkpoint or --resume",
                )
            )
        if "resume" in arg_names and "random_init" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="MUTUALLY_EXCLUSIVE",
                    source_arg="resume",
                    target_args=["random_init"],
                    description="--resume cannot be used with --random-init",
                )
            )
        if "device" in arg_names and "train_device" in arg_names and "infer_device" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="MUTUALLY_EXCLUSIVE",
                    source_arg="device",
                    target_args=["train_device", "infer_device"],
                    description="--device overrides both train and infer devices; do not specify both generic and specific device flags",
                )
            )
        if "tactics_ratio" in arg_names and "tactics_dir" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="REQUIRES",
                    source_arg="tactics_ratio",
                    target_args=["tactics_dir"],
                    description="--tactics-ratio requires --tactics-dir pointing to puzzle dataset chunks",
                )
            )
        if "mode" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="MODE_ONLY",
                    source_arg="concurrent_games",
                    target_args=["mode"],
                    mode="rl",
                    description="--concurrent-games is only active during RL self-play mode",
                )
            )
            rules.append(
                ConstraintRule(
                    rule_type="MODE_ONLY",
                    source_arg="train_dir",
                    target_args=["mode"],
                    mode="supervised",
                    description="--train-dir is only active during supervised training mode",
                )
            )

    # 2. Match command constraints
    if cmd_path.startswith("match"):
        if "engine_color" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="RECOMMENDED_WITH",
                    source_arg="engine_color",
                    target_args=["elo"],
                    description="In Stockfish matches, engine color determines AI's side (white/black/random)",
                )
            )

    # 3. Models export constraints
    if cmd_path == "models export":
        if "output" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="REQUIRES",
                    source_arg="format",
                    target_args=["output"],
                    description="Export format requires specifying target --output path",
                )
            )

    # 4. Data merge / split constraints
    if cmd_path == "data merge":
        if "inputs" in arg_names and "output" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="REQUIRES",
                    source_arg="inputs",
                    target_args=["output"],
                    description="Merging datasets requires both --inputs and --output directories",
                )
            )
    if cmd_path == "data split":
        if "input" in arg_names and "output" in arg_names:
            rules.append(
                ConstraintRule(
                    rule_type="REQUIRES",
                    source_arg="input",
                    target_args=["output"],
                    description="Splitting dataset requires both source --input and destination --output directory",
                )
            )

    return rules


class CommandCatalog:
    """Master catalog of all discoverable ZeroPawn CLI commands, options, and constraints."""

    def __init__(self) -> None:
        self.commands: Dict[str, CommandNode] = {}
        self.root_arguments: List[ArgumentMetadata] = []
        self._initialized: bool = False

    def build_from_parser(self, parser: Optional[argparse.ArgumentParser] = None) -> None:
        """Introspect ArgumentParser and populate the complete command catalog."""
        if parser is None:
            from python.cli.main import build_parser

            parser = build_parser()

        self.commands.clear()
        self.root_arguments.clear()

        # Extract root inherited arguments
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                continue
            if action.dest == "help":
                continue
            meta = ArgumentMetadata(
                name=action.dest,
                flags=list(action.option_strings),
                is_positional=len(action.option_strings) == 0,
                type=_infer_argument_type(action),
                default=action.default if action.default != argparse.SUPPRESS else None,
                required=action.required,
                choices=list(action.choices) if action.choices else None,
                help=action.help or "",
                action_type=type(action).__name__.lower().replace("_", "").replace("action", ""),
                metavar=action.metavar if isinstance(action.metavar, str) else None,
                nargs=action.nargs,
                is_inherited=True,
            )
            self.root_arguments.append(meta)

        # Traverse parser hierarchy recursively
        self._traverse_subparsers(parser, parent_path=[])
        self._initialized = True

    def _traverse_subparsers(self, parser: argparse.ArgumentParser, parent_path: List[str]) -> None:
        sub_actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        is_leaf = (hasattr(parser, "_defaults") and "func" in parser._defaults) or (
            len(parent_path) > 0 and len(sub_actions) == 0
        )
        cmd_path_str = " ".join(parent_path)

        if parent_path and is_leaf:
            # Extract arguments for this parser
            args: List[ArgumentMetadata] = []

            for action in parser._actions:
                if isinstance(action, argparse._SubParsersAction) or action.dest == "help":
                    continue
                meta = ArgumentMetadata(
                    name=action.dest,
                    flags=list(action.option_strings),
                    is_positional=len(action.option_strings) == 0,
                    type=_infer_argument_type(action),
                    default=action.default if action.default != argparse.SUPPRESS else None,
                    required=action.required,
                    choices=list(action.choices) if action.choices else None,
                    help=action.help or "",
                    action_type=type(action)
                    .__name__.lower()
                    .replace("_", "")
                    .replace("action", ""),
                    metavar=action.metavar if isinstance(action.metavar, str) else None,
                    nargs=action.nargs,
                    is_inherited=False,
                )
                args.append(meta)

            # Include inherited root flags
            for r_arg in self.root_arguments:
                args.append(r_arg)

            # Determine category
            cat_id = 10
            for prefix in sorted(CATEGORY_MAPPINGS.keys(), key=len, reverse=True):
                if cmd_path_str == prefix or cmd_path_str.startswith(f"{prefix} "):
                    cat_id = CATEGORY_MAPPINGS[prefix]
                    break
            cat_name = CATEGORIES.get(cat_id, "Maintenance & Utilities")

            # Determine model requirement
            req_model = "none"
            for k, req in MODEL_REQUIREMENTS.items():
                if cmd_path_str == k or cmd_path_str.startswith(f"{k} "):
                    req_model = req
                    break

            # Classification
            classification = COMMAND_CLASSIFICATIONS.get(cmd_path_str, "standard")

            # Constraints
            constraints = _extract_constraints_for_command(cmd_path_str, args)

            # Build CommandNode
            node = CommandNode(
                name=parent_path[-1],
                path=list(parent_path),
                full_command=f"zeropawn {cmd_path_str}",
                aliases=[],  # Will be populated from parent subparsers
                description=parser.description or "",
                help=parser.description or "",
                category_id=cat_id,
                category_name=cat_name,
                classification=classification,
                requires_model=req_model,
                arguments=args,
                is_leaf=True,
                func=parser._defaults.get("func"),
                constraints=constraints,
            )
            self.commands[cmd_path_str] = node

        # Now recurse into subparser actions
        for sa in sub_actions:
            seen_parsers: Dict[int, str] = {}
            for choice_name, sub_p in sa.choices.items():
                p_id = id(sub_p)
                current_path = parent_path + [choice_name]
                if p_id in seen_parsers:
                    # This is an alias for an already processed subparser
                    primary_name = seen_parsers[p_id]
                    primary_path_str = " ".join(parent_path + [primary_name])
                    if primary_path_str in self.commands:
                        if choice_name not in self.commands[primary_path_str].aliases:
                            self.commands[primary_path_str].aliases.append(choice_name)
                    continue

                seen_parsers[p_id] = choice_name
                self._traverse_subparsers(sub_p, current_path)

    def get_command(self, name_or_path: Union[str, List[str]]) -> Optional[CommandNode]:
        """Find a command by name, space-separated string, alias, or token list."""
        if not self._initialized:
            self.build_from_parser()

        if isinstance(name_or_path, list):
            path_str = " ".join(name_or_path)
        else:
            path_str = name_or_path.strip()
            if path_str.startswith(("zeropawn ", "chess ")):
                path_str = path_str.split(" ", 1)[1]

        # Exact path match
        if path_str in self.commands:
            return self.commands[path_str]

        # Check aliases
        for node in self.commands.values():
            if path_str in node.aliases:
                return node
            # Check prefix + alias
            parts = path_str.split()
            if len(parts) == len(node.path):
                if parts[:-1] == node.path[:-1] and parts[-1] in node.aliases:
                    return node

        return None

    def search_commands(self, query: str) -> List[CommandNode]:
        """Search commands by keyword across name, path, aliases, description, and arguments."""
        if not self._initialized:
            self.build_from_parser()

        q = query.lower().strip()
        matches: List[CommandNode] = []
        for node in self.commands.values():
            if q in node.full_command.lower() or any(q in a.lower() for a in node.aliases):
                matches.append(node)
                continue
            if q in node.description.lower() or q in node.help.lower():
                matches.append(node)
                continue
            if any(
                q in arg.name.lower() or any(q in f.lower() for f in arg.flags)
                for arg in node.arguments
            ):
                matches.append(node)
                continue
        return matches

    def filter_by_category(self, category_id: int) -> List[CommandNode]:
        """Return all leaf commands belonging to a specific functional category."""
        if not self._initialized:
            self.build_from_parser()
        return [node for node in self.commands.values() if node.category_id == category_id]

    def all_leaf_commands(self) -> List[CommandNode]:
        """Return all executable leaf commands."""
        if not self._initialized:
            self.build_from_parser()
        return list(self.commands.values())

    def total_arguments_count(self) -> int:
        """Count total unique argument definitions across all commands."""
        if not self._initialized:
            self.build_from_parser()
        return sum(len(node.arguments) for node in self.commands.values())


# Singleton instance for quick access
_GLOBAL_CATALOG: Optional[CommandCatalog] = None


def get_command_catalog() -> CommandCatalog:
    """Retrieve or build the cached singleton CommandCatalog."""
    global _GLOBAL_CATALOG
    if _GLOBAL_CATALOG is None:
        _GLOBAL_CATALOG = CommandCatalog()
        _GLOBAL_CATALOG.build_from_parser()
    return _GLOBAL_CATALOG
