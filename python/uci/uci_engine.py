"""Universal Chess Interface (UCI) Protocol Engine Implementation.

Provides standard UCI protocol compliance for ZeroPawn to play against chess GUIs
such as Arena, Lucas Chess, cutechess-cli, and standard chess tools.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from python.cli.common import (
    get_default_checkpoint_path,
    load_engine_model,
    resolve_device,
    setup_environment,
)
from python.model.chess_net import ChessNet

logger = logging.getLogger(__name__)


class UCIEngine:
    """Implements Universal Chess Interface (UCI) protocol for ZeroPawn."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        default_sims: int = 400,
        c_puct: float = 2.0,
        search_mode: str = "serial",
        batch_size: Union[int, str] = 1,
        use_book: bool = True,
        book_path: Optional[Union[str, Path]] = None,
        book_policy: str = "weighted_blend",
        book_max_ply: int = 30,
        fpu_reduction: float = 0.2,
        reuse_tree: bool = False,
    ) -> None:
        setup_environment()
        import zeropawn_cpp
        from python.eval.match_runner import check_search_mode, require_serial_search
        from python.hardware.profile import resolve_batch_size

        self.zeropawn_cpp = zeropawn_cpp
        self.device = resolve_device(device)
        self.checkpoint_path = checkpoint_path or get_default_checkpoint_path()
        self.default_sims = default_sims
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.reuse_tree = reuse_tree
        self.search_mode = check_search_mode(search_mode)
        if isinstance(batch_size, str) and batch_size.lower() == "auto":
            self.batch_size = resolve_batch_size("auto", workload="uci", device=self.device)
        else:
            self.batch_size = max(1, int(batch_size))
        self._require_serial_search = require_serial_search
        self.use_book = use_book
        self.book_path = str(book_path) if book_path else "data/openings/curated_book.json"
        self.book_policy = book_policy
        self.book_max_ply = book_max_ply
        self.opening_book: Optional[Any] = None
        self._init_opening_book()
        logger.info(
            "UCIEngine search_mode=%s (batch_size=%d), use_book=%s, reuse_tree=%s",
            self.search_mode,
            self.batch_size,
            self.use_book,
            self.reuse_tree,
        )
        self.model: Optional[ChessNet] = None
        self.model_info: dict[str, Any] = {}

        # MCTS Config
        self.mcts_config = self.zeropawn_cpp.MCTSConfig()
        self.mcts_config.c_puct = self.c_puct
        self.mcts_config.fpu_reduction = self.fpu_reduction
        self.mcts_config.reuse_tree = self.reuse_tree
        self.mcts_config.batch_size = self.batch_size
        self.mcts_config.virtual_loss = 1
        self.mcts_config.add_noise = False  # Deterministic / no root noise for play mode per PRD
        self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)

        # Game state and subtree reuse tracking
        self.game = self.zeropawn_cpp.ChessGame()
        self._lock = threading.Lock()
        self.last_engine_action: Optional[int] = None
        self.last_engine_move: Optional[str] = None
        self.prev_moves_at_search: Optional[List[str]] = None
        self.prev_moves: List[str] = []
        self.prev_fen: str = "startpos"

        # Load model if checkpoint exists
        if self.checkpoint_path and Path(self.checkpoint_path).is_file():
            self._load_model(self.checkpoint_path)

    def _load_model(self, path: Union[str, Path]) -> bool:
        try:
            new_model, new_info = load_engine_model(path, device=self.device)
            new_model.eval()
            self.model = new_model
            self.model_info = new_info
            self.checkpoint_path = path
            if hasattr(self, "mcts") and self.mcts is not None:
                self.mcts.clear()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            logger.info("Loaded checkpoint %s on %s", path, self.device)
            return True
        except Exception as e:
            logger.error("Failed to load model from %s: %s", path, e)
            return False

    def _eval_callback(self, game: Any) -> Tuple[List[float], float]:
        if self.model is None:
            num_actions = game.num_actions()
            return ([1.0 / num_actions] * num_actions, 0.0)

        chess_g = self.zeropawn_cpp.ChessGame(game.get_fen())
        planes = self.zeropawn_cpp.encode_state(chess_g)
        t = torch.from_numpy(planes).unsqueeze(0).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).squeeze(0).cpu().numpy().tolist()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).squeeze(0).cpu().numpy()
            scalar_val = float(v_probs[0] - v_probs[2])

        return (p_probs, scalar_val)

    def _batched_eval_callback(self, batch_planes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n = int(batch_planes.shape[0])
        if n == 0 or self.model is None:
            return (np.zeros((n, 4672), dtype=np.float32), np.zeros((n,), dtype=np.float32))

        t = torch.from_numpy(np.ascontiguousarray(batch_planes, dtype=np.float32))
        t = t.view(n, 19, 8, 8).to(self.device, non_blocking=True)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            policy = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            wdl = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            values = (wdl[:, 0] - wdl[:, 2]).astype(np.float32)

        return (policy.astype(np.float32), values)

    def _eval_batch_callback(self, games: List[Any]) -> List[Tuple[List[float], float]]:
        if not games:
            return []
        if self.model is None:
            num_actions = games[0].num_actions()
            uniform_policy = [1.0 / num_actions] * num_actions
            return [(uniform_policy, 0.0) for _ in games]

        batch_planes = []
        for g in games:
            if hasattr(g, "get_fen"):
                cg = self.zeropawn_cpp.ChessGame(g.get_fen())
                batch_planes.append(self.zeropawn_cpp.encode_state(cg))
            else:
                batch_planes.append(self.zeropawn_cpp.encode_state(g))

        batch_arr = np.stack(batch_planes)
        t = torch.from_numpy(batch_arr).to(self.device)
        if self.device == "cuda":
            t = t.to(memory_format=torch.channels_last)

        with torch.no_grad():
            with self._lock:
                if self.device == "cuda":
                    with torch.amp.autocast(self.device, dtype=torch.bfloat16):
                        p_logits, v_wdl = self.model(t)
                else:
                    p_logits, v_wdl = self.model(t)
            p_probs = torch.softmax(p_logits.float(), dim=-1).cpu().numpy()
            v_probs = torch.softmax(v_wdl.float(), dim=-1).cpu().numpy()
            scalar_vals = (v_probs[:, 0] - v_probs[:, 2]).astype(float)

        results = []
        for i in range(len(games)):
            results.append((p_probs[i].tolist(), float(scalar_vals[i])))
        return results

    def _init_opening_book(self) -> None:
        try:
            from python.data.opening_book import OpeningBook

            self.opening_book = OpeningBook(
                book_path=self.book_path,
                policy=self.book_policy,
                max_depth_plies=self.book_max_ply,
                enabled=self.use_book,
            )
        except Exception as e:
            logger.debug("Opening book initialization skipped: %s", e)
            self.opening_book = None

    def search_and_respond(self, sims: int) -> str:
        """Run MCTS search on current game state and output bestmove."""
        self._require_serial_search(self.search_mode, "UCIEngine.search_and_respond")
        if self.game.is_game_over():
            self.mcts.reset_root()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
            return "0000"

        legal_moves = self.game.legal_moves_uci()
        if not legal_moves:
            self.mcts.reset_root()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            sys.stdout.write("bestmove 0000\n")
            sys.stdout.flush()
            return "0000"

        # Check opening book first if enabled
        if self.use_book and self.opening_book is not None:
            book_res = self.opening_book.lookup(self.game)
            if book_res is not None:
                book_move, _ = book_res
                if book_move in legal_moves:
                    self.mcts.reset_root()
                    self.last_engine_action = None
                    self.last_engine_move = None
                    self.prev_moves_at_search = None
                    sys.stdout.write(f"info depth 1 nodes 1 score cp 0 pv {book_move}\n")
                    sys.stdout.write(f"bestmove {book_move}\n")
                    sys.stdout.flush()
                    return book_move

        # Search with MCTS
        self.prev_moves_at_search = list(self.prev_moves)
        res = self.mcts.search(self.game, sims, self._eval_callback, self._eval_batch_callback)
        best_action = res.best_action
        best_move = self.game.action_to_string(best_action)
        root_val = res.root_value
        self.last_engine_action = best_action
        self.last_engine_move = best_move

        # Score in centipawns (AlphaZero approx: score_cp = int(root_val * 400))
        score_cp = int(root_val * 400)
        sys.stdout.write(f"info depth 10 nodes {sims} score cp {score_cp} pv {best_move}\n")
        sys.stdout.write(f"bestmove {best_move}\n")
        sys.stdout.flush()
        return best_move

    def handle_line(self, line: str) -> bool:
        """Handle a single UCI protocol command line. Returns False if 'quit'."""
        line = line.strip()
        if not line:
            return True

        tokens = line.split()
        cmd = tokens[0].lower()

        if cmd == "uci":
            sys.stdout.write("id name ZeroPawn\n")
            sys.stdout.write("id author ZeroPawn Team\n")
            sys.stdout.write("option name Hash type spin default 64 min 1 max 2048\n")
            ckpt_str = str(self.checkpoint_path) if self.checkpoint_path else "none"
            sys.stdout.write(f"option name WeightsFile type string default {ckpt_str}\n")
            sys.stdout.write(
                f"option name Simulations type spin default {self.default_sims} min 10 max 10000\n"
            )
            sys.stdout.write(f"option name CP_UCT type string default {self.c_puct}\n")
            sys.stdout.write(f"option name FPU type string default {self.fpu_reduction}\n")
            sys.stdout.write(
                f"option name ReuseTree type check default {'true' if self.reuse_tree else 'false'}\n"
            )
            sys.stdout.write(f"option name Device type string default {self.device}\n")
            sys.stdout.write(
                f"option name SearchMode type combo default {self.search_mode} var serial var batched\n"
            )
            sys.stdout.write(
                f"option name BatchSize type spin default {self.batch_size} min 1 max 64\n"
            )
            sys.stdout.write(
                f"option name OwnBook type check default {'true' if self.use_book else 'false'}\n"
            )
            sys.stdout.write(f"option name BookFile type string default {self.book_path}\n")
            sys.stdout.write(
                f"option name BookPolicy type combo default {self.book_policy} var frequency var score var weighted_blend var uniform var best\n"
            )
            sys.stdout.write(
                f"option name BookMaxPly type spin default {self.book_max_ply} min 2 max 100\n"
            )
            sys.stdout.write("uciok\n")
            sys.stdout.flush()

        elif cmd == "isready":
            if self.model is None and self.checkpoint_path:
                self._load_model(self.checkpoint_path)
            sys.stdout.write("readyok\n")
            sys.stdout.flush()

        elif cmd == "setoption":
            try:
                name_idx = tokens.index("name") + 1
                val_idx = tokens.index("value") + 1 if "value" in tokens else None
                if val_idx is not None:
                    opt_name = " ".join(tokens[name_idx : val_idx - 1]).lower()
                    opt_val = " ".join(tokens[val_idx:])
                else:
                    opt_name = " ".join(tokens[name_idx:]).lower()
                    opt_val = ""

                if opt_name == "weightsfile":
                    self._load_model(opt_val)
                elif opt_name == "simulations":
                    self.default_sims = int(opt_val)
                elif opt_name in ("cp_uct", "cpuct", "c_puct"):
                    self.c_puct = float(opt_val)
                    self.mcts_config.c_puct = self.c_puct
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name in ("fpu", "fpu_reduction", "fpureduction"):
                    self.fpu_reduction = float(opt_val)
                    self.mcts_config.fpu_reduction = self.fpu_reduction
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name in ("reusetree", "reuse_tree"):
                    self.reuse_tree = opt_val.lower() in ("true", "1", "yes", "on")
                    self.mcts_config.reuse_tree = self.reuse_tree
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                    self.last_engine_action = None
                    self.last_engine_move = None
                    self.prev_moves_at_search = None
                elif opt_name in (
                    "batchsize",
                    "batch_size",
                    "inferencebatchsize",
                    "inference_batch_size",
                ):
                    if opt_val.strip().lower() == "auto":
                        from python.hardware.profile import resolve_batch_size

                        self.batch_size = resolve_batch_size(
                            "auto", workload="uci", device=self.device
                        )
                    else:
                        self.batch_size = max(1, min(64, int(opt_val)))
                    self.mcts_config.batch_size = self.batch_size
                    self.mcts = self.zeropawn_cpp.MCTS(self.mcts_config)
                elif opt_name == "device":
                    self.device = resolve_device(opt_val)
                    if self.checkpoint_path:
                        self._load_model(self.checkpoint_path)
                elif opt_name == "searchmode":
                    from python.eval.match_runner import check_search_mode

                    self.search_mode = check_search_mode(opt_val.lower())
                elif opt_name == "ownbook":
                    self.use_book = opt_val.lower() in ("true", "1", "yes", "on")
                    if self.opening_book:
                        self.opening_book.enabled = self.use_book
                    else:
                        self._init_opening_book()
                elif opt_name == "bookfile":
                    self.book_path = opt_val
                    self._init_opening_book()
                elif opt_name == "bookpolicy":
                    self.book_policy = opt_val.lower()
                    if self.opening_book:
                        self.opening_book.policy = self.book_policy
                elif opt_name == "bookmaxply":
                    try:
                        self.book_max_ply = max(2, min(100, int(opt_val)))
                        if self.opening_book:
                            self.opening_book.max_depth_plies = self.book_max_ply
                        else:
                            self._init_opening_book()
                    except ValueError:
                        logger.debug("Invalid BookMaxPly value: %s", opt_val)
            except Exception as e:
                logger.debug("Failed parsing setoption: %s", e)

        elif cmd == "ucinewgame":
            self.mcts.clear()
            self.mcts.reset_root()
            self.game = self.zeropawn_cpp.ChessGame()
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None
            self.prev_moves = []
            self.prev_fen = "startpos"

        elif cmd == "position":
            idx = 1
            moves_idx = None
            if "moves" in tokens:
                moves_idx = tokens.index("moves")

            current_fen = "startpos"
            if idx < len(tokens) and tokens[idx] == "startpos":
                new_game = self.zeropawn_cpp.ChessGame()
                current_fen = "startpos"
            elif idx < len(tokens) and tokens[idx] == "fen":
                end_fen = moves_idx if moves_idx is not None else len(tokens)
                fen_str = " ".join(tokens[idx + 1 : end_fen])
                new_game = self.zeropawn_cpp.ChessGame(fen_str)
                current_fen = fen_str
            else:
                new_game = self.zeropawn_cpp.ChessGame()

            new_moves: List[str] = []
            if moves_idx is not None:
                for move_uci in tokens[moves_idx + 1 :]:
                    if new_game.is_legal(move_uci):
                        new_game.make_move_uci(move_uci)
                        new_moves.append(move_uci)
                    else:
                        logger.warning(
                            "Illegal move in position command: '%s', stopping move replay", move_uci
                        )
                        break

            # Check if subtree reuse can advance root to resulting position
            reused = False
            if (
                self.reuse_tree
                and self.last_engine_action is not None
                and self.last_engine_move is not None
                and self.prev_moves_at_search is not None
                and current_fen == self.prev_fen
                and len(new_moves) == len(self.prev_moves_at_search) + 2
                and new_moves[: len(self.prev_moves_at_search)] == self.prev_moves_at_search
                and new_moves[-2] == self.last_engine_move
            ):
                opp_move_str = new_moves[-1]
                temp_g = (
                    self.zeropawn_cpp.ChessGame()
                    if current_fen == "startpos"
                    else self.zeropawn_cpp.ChessGame(current_fen)
                )
                for m in self.prev_moves_at_search:
                    temp_g.make_move_uci(m)
                temp_g.make_move_uci(self.last_engine_move)
                opp_action = temp_g.string_to_action(opp_move_str)
                if opp_action >= 0:
                    reused = self.mcts.advance_root(self.last_engine_action, opp_action, new_game)

            if not reused:
                self.mcts.reset_root()

            self.game = new_game
            self.prev_fen = current_fen
            self.prev_moves = new_moves
            self.last_engine_action = None
            self.last_engine_move = None
            self.prev_moves_at_search = None

        elif cmd == "go":
            sims = self.default_sims
            side = self.game.side_to_move()

            args = tokens[1:]
            arg_dict = {}
            i = 0
            while i < len(args):
                k = args[i]
                if (
                    i + 1 < len(args)
                    and not args[i + 1].startswith("-")
                    and args[i + 1] not in ["infinite"]
                ):
                    arg_dict[k] = args[i + 1]
                    i += 2
                else:
                    arg_dict[k] = True
                    i += 1

            if "nodes" in arg_dict:
                try:
                    sims = max(10, int(arg_dict["nodes"]))
                except ValueError:
                    pass
            elif "movetime" in arg_dict:
                try:
                    ms = float(arg_dict["movetime"])
                    sims = max(10, min(10000, int((ms / 1000.0) * 400)))
                except ValueError:
                    pass
            else:
                # NOTE: side_to_move() returns bool (True = White to move).
                time_key = "wtime" if side else "btime"
                inc_key = "winc" if side else "binc"
                if time_key in arg_dict:
                    try:
                        total_time_ms = float(arg_dict[time_key])
                        inc_ms = float(arg_dict.get(inc_key, 0.0))
                        allocated_sec = (total_time_ms / 30000.0) + (inc_ms / 1000.0)
                        allocated_sec = max(0.1, min(10.0, allocated_sec))
                        sims = max(20, min(2000, int(allocated_sec * 400)))
                    except ValueError:
                        pass

            self.search_and_respond(sims)

        elif cmd == "stop":
            pass

        elif cmd == "quit":
            return False

        return True

    def run_stdio_loop(self) -> None:
        """Main stdio loop reading UCI protocol commands."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                if not self.handle_line(line):
                    break
            except (EOFError, KeyboardInterrupt):
                break


def main() -> int:
    """Entry point for standalone UCI engine executable."""
    engine = UCIEngine()
    engine.run_stdio_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
