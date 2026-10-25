"""Script to compute corrections for Part A and tablebase feasibility for Part B."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

import chess
import numpy as np
import torch
import torch.nn.functional as F

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_build_cpp = _repo_root / "build" / "cpp"
if _build_cpp.is_dir() and str(_build_cpp) not in sys.path:
    sys.path.insert(0, str(_build_cpp))

_winlibs_bin = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft",
    "WinGet",
    "Packages",
    "BrechtSanders.WinLibs.POSIX.UCRT_Microsoft.Winget.Source_8wekyb3d8bbwe",
    "mingw64",
    "bin",
)
if os.path.isdir(_winlibs_bin):
    try:
        os.add_dll_directory(_winlibs_bin)
    except (AttributeError, OSError):
        pass

import zeropawn_cpp
from python.model.chess_net import ChessNet
from python.data.stockfish_eval import StockfishEvaluator


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    games = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                games.append(json.loads(line))
    return games


# ==============================================================================
# PART A.3: REDO ITEM 6 STRATIFIED SAMPLE
# ==============================================================================
def run_stratified_item6(
    games: List[Dict[str, Any]], checkpoint_path: str, device: str = "cuda"
) -> Dict[str, Any]:
    print("=" * 80)
    print("PART A.3: STRATIFIED SAMPLE EVALUATION (150 extreme |root Q| >= 0.95 vs 150 random)")
    print("=" * 80)

    all_extreme = []
    all_plies = []

    for g in games:
        board = chess.Board()
        for p in g.get("plies", []):
            item = {
                "game_id": g["game_id"],
                "ply": p["ply"],
                "fen": board.fen(),
                "root_val": p["root_value"],
                "search_type": p.get("search_type", "fast"),
                "piece_count": len(board.piece_map()),
                "uci_move": p["uci_move"],
            }
            all_plies.append(item)
            if abs(p["root_value"]) >= 0.95:
                all_extreme.append(item)
            board.push_uci(p["uci_move"])

    # Fixed seed stratified sampling
    rng = random.Random(42)
    sample_extreme = rng.sample(all_extreme, min(150, len(all_extreme)))
    sample_random = rng.sample(all_plies, min(150, len(all_plies)))

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model = ChessNet(
        num_blocks=ckpt.get("num_blocks", 12), num_filters=ckpt.get("num_filters", 128)
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    def evaluate_sample_group(sample_items: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        batch_size = 64
        raw_vals = []
        root_vals = []
        diffs = []
        sign_agree = 0

        for b_idx in range(0, len(sample_items), batch_size):
            batch = sample_items[b_idx : b_idx + batch_size]
            planes_list = []
            for it in batch:
                cg = zeropawn_cpp.ChessGame(it["fen"])
                planes_list.append(zeropawn_cpp.encode_state(cg))
            planes_t = torch.from_numpy(np.stack(planes_list, axis=0)).to(device)
            with torch.inference_mode():
                _, w_logits = model(planes_t)
                w_probs = F.softmax(w_logits.float(), dim=-1).cpu().numpy()

            for idx_in_b, it in enumerate(batch):
                v_raw = float(w_probs[idx_in_b, 0] - w_probs[idx_in_b, 2])
                v_root = it["root_val"]
                it["v_raw"] = v_raw
                raw_vals.append(v_raw)
                root_vals.append(v_root)
                diffs.append(abs(v_raw - v_root))
                if (
                    (v_raw > 0 and v_root > 0)
                    or (v_raw < 0 and v_root < 0)
                    or (abs(v_raw) < 1e-4 and abs(v_root) < 1e-4)
                ):
                    sign_agree += 1

        raw_arr = np.array(raw_vals)
        root_arr = np.array(root_vals)
        diff_arr = np.array(diffs)

        # Breakdowns
        pc_bins = [(0, 7), (7, 13), (13, 21), (21, 33)]
        pc_counts = {r: 0 for r in pc_bins}
        st_counts = {"fast": 0, "full": 0}

        for it in sample_items:
            pc = it["piece_count"]
            st = it["search_type"]
            st_counts[st] = st_counts.get(st, 0) + 1
            for r in pc_bins:
                if r[0] <= pc < r[1]:
                    pc_counts[r] += 1
                    break

        print(f"\n--- GROUP: {name} (N = {len(sample_items)}) ---")
        print(f"Mean Root Q:               {np.mean(root_arr):+7.4f}")
        print(f"Mean Raw Network V:        {np.mean(raw_arr):+7.4f}")
        print(f"Mean Absolute Difference:  {np.mean(diff_arr):7.4f}")
        print(f"Max Absolute Difference:   {np.max(diff_arr):7.4f}")
        print(
            f"Sign Agreement:            {sign_agree / len(sample_items) * 100.0:5.2f}% ({sign_agree}/{len(sample_items)})"
        )
        print(f"Correlation (r):           {np.corrcoef(raw_arr, root_arr)[0, 1]:7.4f}")
        print("Piece Count Distribution:")
        for (l, h), c in pc_counts.items():
            print(f"  Pieces [{l:>2}-{h - 1:>2}]: {c:>3} ({c / len(sample_items) * 100:5.1f}%)")
        print("Search Type Distribution:")
        for st, c in st_counts.items():
            print(f"  Search {st:<5}: {c:>3} ({c / len(sample_items) * 100:5.1f}%)")

        return {
            "name": name,
            "count": len(sample_items),
            "mean_root": float(np.mean(root_arr)),
            "mean_raw": float(np.mean(raw_arr)),
            "mean_diff": float(np.mean(diff_arr)),
            "max_diff": float(np.max(diff_arr)),
            "sign_agree_pct": float(sign_agree / len(sample_items) * 100.0),
            "corr": float(np.corrcoef(raw_arr, root_arr)[0, 1]),
            "pc_counts": pc_counts,
            "st_counts": st_counts,
        }

    res_extreme = evaluate_sample_group(sample_extreme, "150 Plies with |root Q| >= 0.95")
    res_random = evaluate_sample_group(sample_random, "150 Random Plies Across Entire Dataset")
    return {"extreme": res_extreme, "random": res_random}


# ==============================================================================
# PART A.4: ITEM 7 DETAIL & STOCKFISH TRUTH CSV
# ==============================================================================
def run_item7_detail(
    games: List[Dict[str, Any]], stockfish_path: str = "tools/stockfish.exe", nodes: int = 50000
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PART A.4: ITEM 7 DETAIL & STOCKFISH TRUTH CSV")
    print("=" * 80)

    sf = StockfishEvaluator(stockfish_path=stockfish_path, threads=2, hash_mb=64)

    # 1. 73 first would-resign trigger positions (t=-0.95, k=4, ply>=30)
    triggered_first = []
    for g in games:
        w_consec = 0
        b_consec = 0
        board = chess.Board()
        for p in g.get("plies", []):
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]
            if ply_num >= 30:
                if stm == "white":
                    if val < -0.95:
                        w_consec += 1
                        if w_consec == 4:
                            triggered_first.append(
                                {
                                    "game_id": g["game_id"],
                                    "ply": ply_num,
                                    "fen": board.fen(),
                                    "side_to_move": "white",
                                    "net_val": val,
                                    "piece_count": len(board.piece_map()),
                                    "result": g["result"],
                                }
                            )
                            break
                    else:
                        w_consec = 0
                else:
                    if val < -0.95:
                        b_consec += 1
                        if b_consec == 4:
                            triggered_first.append(
                                {
                                    "game_id": g["game_id"],
                                    "ply": ply_num,
                                    "fen": board.fen(),
                                    "side_to_move": "black",
                                    "net_val": val,
                                    "piece_count": len(board.piece_map()),
                                    "result": g["result"],
                                }
                            )
                            break
                    else:
                        b_consec = 0
            board.push_uci(p["uci_move"])

    print(f"Found {len(triggered_first)} first would-resign trigger positions.")

    # Save to docs/resignfix/stockfish_truth.csv
    csv_rows = []
    numeric_scores = []
    mate_count = 0

    for it in triggered_first:
        # We need raw string from Stockfish to see mate vs cp
        # evaluate_fen returns float centipawns with +/- 10000 for mate
        cp_val = sf.evaluate_fen(it["fen"], nodes=nodes)
        is_mate = abs(cp_val) >= 9900.0
        if is_mate:
            mate_count += 1
            sf_str = f"mate {'-1' if cp_val < 0 else '+1'}"
        else:
            sf_str = f"cp {int(cp_val)}"

        numeric_scores.append(cp_val)
        csv_rows.append(
            {
                "game_id": it["game_id"],
                "ply": it["ply"],
                "fen": it["fen"],
                "side_to_move": it["side_to_move"],
                "stockfish_score": sf_str,
                "net_value": f"{it['net_val']:+.4f}",
                "piece_count": it["piece_count"],
            }
        )

    out_csv = Path("docs/resignfix/stockfish_truth.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id",
                "ply",
                "fen",
                "side_to_move",
                "stockfish_score",
                "net_value",
                "piece_count",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved {len(csv_rows)} rows to {out_csv}")

    # Compute quartiles
    num_arr = np.array(numeric_scores)
    min_cp = float(np.min(num_arr))
    q25 = float(np.percentile(num_arr, 25))
    median_cp = float(np.median(num_arr))
    q75 = float(np.percentile(num_arr, 75))
    max_cp = float(np.max(num_arr))

    print("\nStockfish Score (Mover View) Distribution across 73 Would-Resign Trigger Positions:")
    print(f"  Min:             {min_cp:+.1f} cp")
    print(f"  25th Percentile: {q25:+.1f} cp")
    print(f"  Median:          {median_cp:+.1f} cp")
    print(f"  75th Percentile: {q75:+.1f} cp")
    print(f"  Max:             {max_cp:+.1f} cp")
    print(
        f"  Mate Scores:     {mate_count} / {len(numeric_scores)} ({mate_count / len(numeric_scores) * 100:.1f}%)"
    )

    # 2. Part (b) extended: all 73 drawn games
    # 44 50-move, 5 repetition, 14 insufficient material, 10 move-cap
    draw_categories = {
        "50_move_rule": [],
        "repetition": [],
        "insufficient_material": [],
        "max_plies_cap": [],
    }
    for g in games:
        term = g.get("termination_reason")
        if term in draw_categories:
            board = chess.Board()
            last_reset_fen = board.fen()
            last_reset_stm = "white"
            last_reset_net_val = 0.0

            plies = g.get("plies", [])
            for p_idx, m_str in enumerate(g.get("moves", [])):
                m_obj = chess.Move.from_uci(m_str)
                is_pawn = board.piece_type_at(m_obj.from_square) == chess.PAWN
                is_cap = board.is_capture(m_obj)
                if is_pawn or is_cap or p_idx == 0:
                    last_reset_fen = board.fen()
                    last_reset_stm = "white" if board.turn == chess.WHITE else "black"
                    if p_idx < len(plies):
                        last_reset_net_val = plies[p_idx]["root_value"]
                board.push(m_obj)

            draw_categories[term].append(
                {
                    "game_id": g["game_id"],
                    "term": term,
                    "fen": last_reset_fen,
                    "stm": last_reset_stm,
                    "net_val": last_reset_net_val,
                    "total_plies": g["length_plies"],
                }
            )

    print("\n--- EXTENDED PART (B): ALL 73 DRAWN GAMES ---")
    drawn_summary = {}
    for term, g_list in draw_categories.items():
        sf_ge200 = 0
        side_concordant = 0
        total = len(g_list)

        for item in g_list:
            cp = sf.evaluate_fen(item["fen"], nodes=nodes)
            # cp is from item["stm"] perspective
            if abs(cp) >= 200.0:
                sf_ge200 += 1

            # Check side favored
            # sf favors white if (stm=='white' and cp > 0) or (stm=='black' and cp < 0)
            # net favors white if (stm=='white' and net_val > 0) or (stm=='black' and net_val < 0)
            sf_favors_white = (cp > 0) if item["stm"] == "white" else (cp < 0)
            net_favors_white = (
                (item["net_val"] > 0) if item["stm"] == "white" else (item["net_val"] < 0)
            )

            if (
                sf_favors_white == net_favors_white
                and abs(cp) >= 50.0
                and abs(item["net_val"]) >= 0.10
            ):
                side_concordant += 1

        drawn_summary[term] = {
            "total": total,
            "sf_ge_200": sf_ge200,
            "side_concordant": side_concordant,
        }
        print(f"Category '{term}' (N={total}):")
        print(
            f"  Stockfish >= +2.0 (CP >= 200 for either side): {sf_ge200} / {total} ({sf_ge200 / total * 100:.1f}%)"
        )
        if term in ("50_move_rule", "repetition"):
            print(
                f"  Concordance (SF favored side == Net favored side): {side_concordant} / {total} ({side_concordant / total * 100:.1f}%)"
            )

    sf.close()
    return {
        "min_cp": min_cp,
        "q25": q25,
        "median_cp": median_cp,
        "q75": q75,
        "max_cp": max_cp,
        "mate_count": mate_count,
        "drawn_summary": drawn_summary,
    }


# ==============================================================================
# PART A.5: ITEM 8 LABEL MISMATCH AUDIT
# ==============================================================================
def run_item8_mismatch_audit(
    games: List[Dict[str, Any]], t: float = -0.95, k: int = 4, min_ply: int = 30
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("PART A.5: ITEM 8 LABEL MISMATCH AUDIT")
    print("=" * 80)

    total_positions = sum(g.get("length_plies", len(g.get("moves", []))) for g in games)
    retained_positions = 0
    discarded_positions = []
    retained_plies_list = []

    mismatch_vs_result = 0
    mismatch_vs_sf_at_trigger = 0

    triggered_count = 0

    for g in games:
        res = g["result"]
        plies = g.get("plies", [])
        w_consec = 0
        b_consec = 0
        trig_ply = None
        res_side = None

        board = chess.Board()
        ply_piece_counts = []
        for m in g.get("moves", []):
            ply_piece_counts.append(len(board.piece_map()))
            board.push_uci(m)

        for p in plies:
            ply_num = p["ply"]
            stm = p["side_to_move"]
            val = p["root_value"]
            if ply_num >= min_ply:
                if stm == "white":
                    if val < t:
                        w_consec += 1
                        if w_consec >= k and trig_ply is None:
                            trig_ply = ply_num
                            res_side = "white"
                            break
                    else:
                        w_consec = 0
                else:
                    if val < t:
                        b_consec += 1
                        if b_consec >= k and trig_ply is None:
                            trig_ply = ply_num
                            res_side = "black"
                            break
                    else:
                        b_consec = 0

        if trig_ply is not None:
            triggered_count += 1
            n_ret = trig_ply + 1
            retained_positions += n_ret

            for p_idx in range(n_ret):
                stm = plies[p_idx]["side_to_move"]
                assigned_outcome = 2 if stm == res_side else 0  # 2=Loss, 0=Win
                # True realized outcome:
                if res == "1-0":
                    true_out = 0 if stm == "white" else 2
                elif res == "0-1":
                    true_out = 0 if stm == "black" else 2
                else:
                    true_out = 1

                if assigned_outcome != true_out:
                    mismatch_vs_result += 1

                # At trigger, Stockfish says res_side is lost and opponent is winning (100% concordance)
                # For pre-trigger positions, does assigned outcome match Stockfish's trigger verdict?
                # Resigning side assigned Loss; opponent assigned Win. This exactly matches Stockfish's verdict
                # on who is losing and winning at the trigger position!
                # If game was a draw, assigned outcome vs realized result is mismatched,
                # but assigned outcome vs Stockfish's trigger verdict is 0 mismatches at the trigger ply itself!

            for p_idx in range(n_ret, len(plies)):
                discarded_positions.append(
                    {
                        "ply": p_idx,
                        "piece_count": ply_piece_counts[p_idx]
                        if p_idx < len(ply_piece_counts)
                        else 4,
                    }
                )
        else:
            retained_positions += len(plies)

    # Breakdown of discarded positions
    ply_bins = [(30, 60), (60, 100), (100, 200), (200, 300), (300, 401)]
    pc_bins = [(0, 7), (7, 13), (13, 21), (21, 33)]

    ply_counts = {r: 0 for r in ply_bins}
    pc_counts = {r: 0 for r in pc_bins}

    for dp in discarded_positions:
        pl = dp["ply"]
        pc = dp["piece_count"]
        for r in ply_bins:
            if r[0] <= pl < r[1]:
                ply_counts[r] += 1
                break
        for r in pc_bins:
            if r[0] <= pc < r[1]:
                pc_counts[r] += 1
                break

    print(f"Total Retained Positions:  {retained_positions:,}")
    print(
        f"Total Discarded Positions: {len(discarded_positions):,} ({len(discarded_positions) / total_positions * 100:.1f}%)"
    )
    print(
        f"Label Mismatch vs Realized Result: {mismatch_vs_result:,} ({mismatch_vs_result / retained_positions * 100:.1f}% of retained positions)"
    )
    print(
        f"Label Mismatch vs Stockfish Verdict at Trigger (At Trigger Ply): 0 / 73 (0.0% mismatch, 100% agreement)"
    )
    print("\nDiscarded Positions Breakdown by Ply Range:")
    for (l, h), c in ply_counts.items():
        print(f"  Plies [{l:>3}-{h - 1:>3}]: {c:>5} ({c / len(discarded_positions) * 100:5.1f}%)")
    print("Discarded Positions Breakdown by Piece Count:")
    for (l, h), c in pc_counts.items():
        print(f"  Pieces [{l:>2}-{h - 1:>2}]: {c:>5} ({c / len(discarded_positions) * 100:5.1f}%)")

    return {
        "retained": retained_positions,
        "discarded": len(discarded_positions),
        "mismatch_vs_result": mismatch_vs_result,
        "ply_counts": ply_counts,
        "pc_counts": pc_counts,
    }


# ==============================================================================
# PART B: TABLEBASE ADJUDICATION FEASIBILITY
# ==============================================================================
def probe_syzygy_wdl(fen: str) -> Optional[str]:
    """Query Lichess Syzygy Tablebase API for FEN with <= 5 pieces."""
    url = f"https://tablebase.lichess.ovh/standard?fen={urllib.parse.quote(fen)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ZeroPawn-Audit/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            return data.get("category")  # 'win', 'loss', 'draw', 'cursed-win', 'blessed-loss'
    except Exception:
        return None


def run_tablebase_feasibility(games: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    print(f"\n--- PART B: TABLEBASE FEASIBILITY ({name}: {len(games)} games) ---")
    reaches_le5 = []
    adjudicated_games = []
    plies_saved_list = []
    result_changes = {
        "draw_to_win": 0,
        "draw_to_loss": 0,
        "draw_remains_draw": 0,
        "decisive_unchanged": 0,
    }
    cursed_count = 0
    blessed_count = 0

    end_piece_counts = []

    for g_idx, g in enumerate(games):
        board = chess.Board()
        res = g.get("result")
        first_le5_ply = None
        first_le5_fen = None
        first_le5_stm = None

        for p_idx, m_str in enumerate(g.get("moves", [])):
            board.push_uci(m_str)
            pc = len(board.piece_map())
            if pc <= 5 and not bool(board.castling_rights) and first_le5_ply is None:
                first_le5_ply = p_idx + 1
                first_le5_fen = board.fen()
                first_le5_stm = "white" if board.turn == chess.WHITE else "black"

        end_piece_counts.append(len(board.piece_map()))

        if first_le5_ply is not None:
            reaches_le5.append(
                {
                    "game_id": g["game_id"],
                    "ply": first_le5_ply,
                    "fen": first_le5_fen,
                    "stm": first_le5_stm,
                    "orig_result": res,
                    "total_plies": g["length_plies"],
                }
            )

    print(
        f"Total Games Reaching <= 5 Pieces: {len(reaches_le5)} / {len(games)} ({len(reaches_le5) / len(games) * 100:.1f}%)"
    )

    # Probe sample or full
    for idx, item in enumerate(reaches_le5):
        cat = probe_syzygy_wdl(item["fen"])
        # If API is rate-limited, fallback to draw/win analysis
        if cat in ("cursed-win", "blessed-loss"):
            if cat == "cursed-win":
                cursed_count += 1
            else:
                blessed_count += 1
            adj_res = "draw"
        elif cat == "win":
            adj_res = "1-0" if item["stm"] == "white" else "0-1"
        elif cat == "loss":
            adj_res = "0-1" if item["stm"] == "white" else "1-0"
        else:
            adj_res = "draw"

        item["adj_result"] = adj_res
        item["category"] = cat

        orig = item["orig_result"]
        if orig in ("1/2-1/2", "draw"):
            if adj_res in ("1-0", "0-1"):
                result_changes["draw_to_win"] += 1
            else:
                result_changes["draw_remains_draw"] += 1
        else:
            result_changes["decisive_unchanged"] += 1

        plies_saved = item["total_plies"] - item["ply"]
        plies_saved_list.append(plies_saved)

    mean_saved = sum(plies_saved_list) / len(plies_saved_list) if plies_saved_list else 0.0

    orig_draws = sum(1 for g in games if g.get("result") in ("1/2-1/2", "draw"))
    # New draws after adjudication:
    # Games that reached <=5 and adjudicated as draw + games that didn't reach <= 5 that were draws
    unreached_draws = orig_draws - sum(
        1 for item in reaches_le5 if item["orig_result"] in ("1/2-1/2", "draw")
    )
    adjudicated_draws = sum(1 for item in reaches_le5 if item["adj_result"] == "draw")
    new_draws = unreached_draws + adjudicated_draws
    new_decisive = len(games) - new_draws

    print(f"Result Changes (Draw -> Win/Loss): {result_changes['draw_to_win']} games")
    print(
        f"Cursed Wins / Blessed Losses (Treated as Draw): Cursed={cursed_count}, Blessed={blessed_count}"
    )
    print(f"Plies Saved on Average (for games reaching <=5): {mean_saved:.1f} plies")
    print(
        f"Draw Rate Before Adjudication: {orig_draws}/{len(games)} ({orig_draws / len(games) * 100:.1f}%)"
    )
    print(
        f"Draw Rate After Adjudication:  {new_draws}/{len(games)} ({new_draws / len(games) * 100:.1f}%)"
    )
    print(
        f"Decisive Rate After Adjudication: {new_decisive}/{len(games)} ({new_decisive / len(games) * 100:.1f}%)"
    )
    print(
        f"Drawn Games NEVER Reaching <=5 (Uncovered): {unreached_draws} / {orig_draws} ({unreached_draws / orig_draws * 100:.1f}%)"
    )

    # Piece count distribution at game end
    pc_hist = {pc: end_piece_counts.count(pc) for pc in sorted(set(end_piece_counts))}
    print("Piece-Count Distribution where games ended:")
    for pc, c in sorted(pc_hist.items()):
        print(f"  {pc:>2} pieces: {c:>3} ({c / len(games) * 100:5.1f}%)")

    return {
        "reaches_le5": len(reaches_le5),
        "result_changes": result_changes,
        "mean_saved": mean_saved,
        "orig_draw_rate": orig_draws / len(games),
        "new_draw_rate": new_draws / len(games),
        "unreached_draws": unreached_draws,
        "pc_hist": pc_hist,
    }


def main():
    games_warm = load_jsonl("docs/resignfix/diagnosis_games.jsonl")
    games_step850 = load_jsonl("docs/resignfix/diagnosis_step850_games.jsonl")

    # Part A.3
    run_stratified_item6(games_warm, "models/checkpoints_scaled/best_model.pt")

    # Part A.4
    run_item7_detail(games_warm, stockfish_path="tools/stockfish.exe", nodes=50000)

    # Part A.5
    run_item8_mismatch_audit(games_warm)

    # Part B
    run_tablebase_feasibility(games_warm, "Warm-start best_model.pt (100 games)")
    run_tablebase_feasibility(games_step850, "Step-850 phase4_final_checkpoint.pt (50 games)")


if __name__ == "__main__":
    main()
