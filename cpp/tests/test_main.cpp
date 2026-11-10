#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include "zeropawn/stub.h"
#include "zeropawn/chess_game.hpp"
#include "zeropawn/encoder.hpp"
#include "zeropawn/mcts.hpp"
#include "zeropawn/connect4.hpp"

#define TEST_CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[FAIL] " << msg << " (" << #cond << ") at line " << __LINE__ << "\n"; \
            std::exit(1); \
        } \
    } while (0)

void test_stub() {
    const char* v = zeropawn::version();
    TEST_CHECK(v != nullptr, "version string not null");
    std::cout << "[PASS] version() = " << v << "\n";
}

void test_board_basics() {
    zeropawn::ChessGame game;
    TEST_CHECK(game.side_to_move() == true, "White should move first");
    TEST_CHECK(game.full_move_number() == 1, "Full move 1");
    TEST_CHECK(game.half_move_clock() == 0, "Half move clock 0");
    TEST_CHECK(!game.in_check(), "Not in check at start");

    auto moves = game.legal_moves_uci();
    TEST_CHECK(moves.size() == 20, "Startpos should have 20 moves");

    uint64_t h0 = game.get_hash();
    bool ok = game.make_move_uci("e2e4");
    TEST_CHECK(ok, "e2e4 should be legal");
    TEST_CHECK(game.side_to_move() == false, "Black should move after e2e4");
    uint64_t h1 = game.get_hash();
    TEST_CHECK(h0 != h1, "Hash must change after move");

    bool unmake_ok = game.unmake_move();
    TEST_CHECK(unmake_ok, "Unmake should succeed");
    TEST_CHECK(game.side_to_move() == true, "White to move after unmake");
    TEST_CHECK(game.get_hash() == h0, "Hash must restore after unmake");

    // Clone test
    auto clone_game = game.clone();
    TEST_CHECK(clone_game != nullptr, "Clone should not be null");
    TEST_CHECK(clone_game->get_hash() == game.get_hash(), "Clone hash matches original");
    auto* chess_clone = dynamic_cast<zeropawn::ChessGame*>(clone_game.get());
    TEST_CHECK(chess_clone != nullptr, "Clone should cast to ChessGame");
    TEST_CHECK(chess_clone->side_to_move() == game.side_to_move(), "Clone side to move matches");

    // Safe set_fen: invalid FEN must return false and not corrupt board state
    bool bad_fen_res = game.set_fen("invalid fen syntax");
    TEST_CHECK(!bad_fen_res, "Invalid FEN must return false");
    TEST_CHECK(game.side_to_move() == true, "Board must not be corrupted by invalid FEN");
    TEST_CHECK(game.get_hash() == h0, "Hash must be intact after rejected invalid FEN");

    std::cout << "[PASS] test_board_basics\n";
}

void test_perft() {
    zeropawn::ChessGame game;
    std::vector<uint64_t> startpos_expected = {20, 400, 8902, 197281, 4865609};
    for (int d = 1; d <= 5; ++d) {
        uint64_t nodes = game.perft(d);
        TEST_CHECK(nodes == startpos_expected[d-1], "Startpos perft mismatch");
    }

    std::string kiwi_fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
    game.set_fen(kiwi_fen);
    std::vector<uint64_t> kiwi_expected = {48, 2039, 97862, 4085603};
    for (int d = 1; d <= 4; ++d) {
        uint64_t nodes = game.perft(d);
        TEST_CHECK(nodes == kiwi_expected[d-1], "Kiwipete perft mismatch");
    }

    std::string pos3_fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1";
    game.set_fen(pos3_fen);
    std::vector<uint64_t> pos3_expected = {14, 191, 2812, 43238};
    for (int d = 1; d <= 4; ++d) {
        uint64_t nodes = game.perft(d);
        TEST_CHECK(nodes == pos3_expected[d-1], "Position 3 perft mismatch");
    }

    std::cout << "[PASS] test_perft\n";
}

void test_terminal() {
    zeropawn::ChessGame game;
    game.set_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2");
    TEST_CHECK(game.in_check(), "White must be in check");
    TEST_CHECK(game.is_checkmate(), "Fool's mate must be checkmate");
    TEST_CHECK(!game.is_stalemate(), "Checkmate is not stalemate");
    TEST_CHECK(game.is_game_over(), "Checkmate must be game over");

    game.set_fen("k7/8/1Q6/8/8/8/8/7K b - - 0 1");
    TEST_CHECK(!game.in_check(), "Black is not in check");
    TEST_CHECK(game.is_stalemate(), "Must be stalemate");

    std::cout << "[PASS] test_terminal\n";
}

void test_encoder() {
    zeropawn::ChessGame game;
    std::vector<float> state = zeropawn::encode_state_vec(game);
    TEST_CHECK(state.size() == 1216, "State size must be 1216");

    auto moves = game.legal_moves_uci();
    for (const auto& m : moves) {
        int idx = zeropawn::move_to_index(game, m);
        TEST_CHECK(idx >= 0 && idx < zeropawn::NUM_ACTIONS, "Action index in bounds");
        std::string back = zeropawn::index_to_move(game, idx);
        TEST_CHECK(back == m, "Move round trip failed for " + m);
    }

    std::cout << "[PASS] test_encoder\n";
}

void test_mcts() {
    auto dummy_eval = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        int n = g.num_actions();
        std::vector<float> policy(n, 1.0f / static_cast<float>(n));
        return {policy, 0.0f};
    };

    struct MateCase {
        std::string fen;
        std::string mate_move;
    };

    std::vector<MateCase> mate_cases = {
        {"r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "f3f7"},
        {"rnbqkbnr/ppppp2p/8/5pp1/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", "d1h5"},
        {"6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", "e1e8"},
        {"8/8/8/8/8/5K2/4Q3/7k w - - 0 1", "e2g2"},
        {"7k/5K2/8/8/8/8/8/R7 w - - 0 1", "a1h1"},
        // Black to move mate-in-1 cases
        {"rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2", "d8h4"},
        {"8/8/8/8/8/5k2/4q3/7K b - - 0 1", "e2g2"},
        {"4r1k1/5ppp/8/8/8/8/5PPP/4R1K1 b - - 0 1", "e8e1"}
    };

    zeropawn::MCTSConfig config;
    config.c_puct = 2.0f;
    config.add_noise = false;
    config.seed = 12345;
    zeropawn::MCTS mcts(config);

    for (size_t i = 0; i < mate_cases.size(); ++i) {
        zeropawn::ChessGame game(mate_cases[i].fen);
        auto res = mcts.search(game, 200, dummy_eval);
        std::string found_move = game.action_to_string(res.best_action);
        if (i == 3) {
            TEST_CHECK(found_move == "e2g2" || found_move == "e2f1", "Mate 4 not found");
        } else {
            TEST_CHECK(found_move == mate_cases[i].mate_move, "Mate move mismatch");
        }
    }

    zeropawn::ChessGame startpos;
    mcts.set_seed(42);
    auto res1 = mcts.search(startpos, 100, dummy_eval);
    float sum_visits = 0.0f;
    for (float v : res1.action_visits) sum_visits += v;
    TEST_CHECK(std::abs(sum_visits - 1.0f) < 1e-4f, "Action visits must sum to 1.0");

    mcts.set_seed(42);
    auto res2 = mcts.search(startpos, 100, dummy_eval);
    TEST_CHECK(res1.best_action == res2.best_action, "Deterministic search with same seed");

    std::cout << "[PASS] test_mcts\n";
}

void test_connect4() {
    zeropawn::Connect4 c4;
    TEST_CHECK(c4.legal_actions().size() == 7, "Initial legal actions 7");
    TEST_CHECK(!c4.is_terminal(), "Not terminal at start");
    TEST_CHECK(c4.current_player() == 1, "Player 1 starts");

    // Vertical win for Player 1:
    // P1: col 0, P2: col 1, P1: col 0, P2: col 1, P1: col 0, P2: col 1, P1: col 0 (win!)
    c4.apply_action(0); // P1
    c4.apply_action(1); // P2
    c4.apply_action(0); // P1
    c4.apply_action(1); // P2
    c4.apply_action(0); // P1
    c4.apply_action(1); // P2
    TEST_CHECK(!c4.is_terminal(), "Still ongoing before 4th move");
    c4.apply_action(0); // P1 connects 4!
    TEST_CHECK(c4.is_terminal(), "Terminal after 4 in a col");
    TEST_CHECK(c4.winner() == 1, "Player 1 should win");
    TEST_CHECK(c4.terminal_value() == -1.0f, "Terminal value from loser perspective is -1.0");

    // Connect 4 with MCTS:
    // With 3 in a row on col 0, MCTS should immediately pick col 0 to win in 1!
    zeropawn::Connect4 c4_mcts;
    c4_mcts.apply_action(0); // P1
    c4_mcts.apply_action(1); // P2
    c4_mcts.apply_action(0); // P1
    c4_mcts.apply_action(1); // P2
    c4_mcts.apply_action(0); // P1
    c4_mcts.apply_action(1); // P2
    // Now P1 to move, col 0 completes 4!
    auto dummy_c4_eval = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        return {std::vector<float>(7, 1.0f / 7.0f), 0.0f};
    };
    zeropawn::MCTS mcts;
    auto res = mcts.search(c4_mcts, 50, dummy_c4_eval);
    TEST_CHECK(res.best_action == 0, "MCTS must find winning column 0 in Connect 4");

    std::cout << "[PASS] test_connect4\n";
}

void test_subtree_reuse() {
    auto dummy_eval = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        int n = g.num_actions();
        std::vector<float> policy(n, 1.0f / static_cast<float>(n));
        return {policy, 0.0f};
    };

    // 1. Config defaults verification
    zeropawn::MCTSConfig cfg_default;
    TEST_CHECK(!cfg_default.reuse_tree, "reuse_tree must be false by default");
    TEST_CHECK(cfg_default.max_nodes == 25000, "max_nodes must default to 25000");

    // 2. Disabled-mode search lifecycle: root must be cleared after search
    {
        zeropawn::MCTS mcts_off(cfg_default);
        zeropawn::ChessGame game;
        auto res = mcts_off.search(game, 50, dummy_eval);
        TEST_CHECK(res.retained_visits == 0, "Disabled mode must have 0 retained visits");
        TEST_CHECK(res.new_simulations == 50, "Disabled mode must execute 50 simulations");
        TEST_CHECK(!mcts_off.has_retained_root(), "Disabled mode must not retain root");
        TEST_CHECK(mcts_off.live_node_count() == 0, "Disabled mode live_node_count must be 0");
    }

    // 3. Enabled-mode retained root and advance_root
    zeropawn::MCTSConfig cfg_reuse;
    cfg_reuse.reuse_tree = true;
    cfg_reuse.seed = 42;
    zeropawn::MCTS mcts(cfg_reuse);
    zeropawn::ChessGame game;

    auto res1 = mcts.search(game, 200, dummy_eval);
    TEST_CHECK(res1.retained_visits == 0, "First search has 0 retained visits");
    TEST_CHECK(res1.new_simulations == 200, "First search executes 200 new simulations");
    TEST_CHECK(mcts.has_retained_root(), "Enabled mode must retain root");
    size_t live_nodes_1 = mcts.live_node_count();
    TEST_CHECK(live_nodes_1 > 1, "Live node count must be > 1");

    int my_action = res1.best_action;
    std::string my_move = game.action_to_string(my_action);
    zeropawn::ChessGame game_after_my_move;
    game_after_my_move.make_move_uci(my_move);

    // Find an opponent move explored under my_action
    auto opp_legal = game_after_my_move.legal_moves_uci();
    int chosen_opp_act = -1;
    std::string chosen_opp_move = "";
    zeropawn::ChessGame game_after_both;

    for (const auto& om : opp_legal) {
        int oact = game_after_my_move.string_to_action(om);
        zeropawn::ChessGame cand_game = game_after_my_move;
        cand_game.make_move_uci(om);

        // Try single action advance or 2-action advance
        // We test advancing from root with (my_action, oact, cand_game)
        zeropawn::MCTS probe_mcts(cfg_reuse);
        probe_mcts.search(game, 200, dummy_eval);
        if (probe_mcts.advance_root(my_action, oact, cand_game)) {
            chosen_opp_act = oact;
            chosen_opp_move = om;
            game_after_both = cand_game;
            break;
        }
    }

    TEST_CHECK(chosen_opp_act >= 0, "At least one opponent reply must be expanded in 200 sims");

    // Advance real MCTS root
    bool adv_ok = mcts.advance_root(my_action, chosen_opp_act, game_after_both);
    TEST_CHECK(adv_ok, "Advance root must succeed for valid move sequence");
    TEST_CHECK(mcts.has_retained_root(), "Root must be retained after successful advance");
    size_t live_nodes_2 = mcts.live_node_count();
    TEST_CHECK(live_nodes_2 > 0 && live_nodes_2 < live_nodes_1, "Discarded siblings must be reclaimed");

    // Subsequent search preserves retained visits and executes exactly M new simulations
    auto res2 = mcts.search(game_after_both, 100, dummy_eval);
    TEST_CHECK(res2.retained_visits > 0, "Retained visits must be > 0");
    TEST_CHECK(res2.new_simulations == 100, "Exactly M=100 new simulations executed");
    TEST_CHECK(res2.best_action >= 0, "Valid best action returned");

    // 4. Missing child fallback
    bool bad_child = mcts.advance_root(my_action, 9999, game_after_both);
    TEST_CHECK(!bad_child, "Missing child must fail advance");
    TEST_CHECK(!mcts.has_retained_root(), "Failed advance must reset root to nullptr");
    TEST_CHECK(mcts.live_node_count() == 0, "Live node count must be 0 after reset");

    // 5. Board state mismatch fallback
    mcts.search(game, 50, dummy_eval);
    TEST_CHECK(mcts.has_retained_root(), "Root retained before mismatch test");
    zeropawn::ChessGame mismatch_state("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    bool mismatch_res = mcts.advance_root(my_action, chosen_opp_act, mismatch_state);
    TEST_CHECK(!mismatch_res, "State mismatch must reject advance");
    TEST_CHECK(!mcts.has_retained_root(), "State mismatch must reset root to nullptr");

    // 6. Node limit enforcement
    zeropawn::MCTSConfig cfg_cap;
    cfg_cap.reuse_tree = true;
    cfg_cap.max_nodes = 5; // Very small cap
    zeropawn::MCTS mcts_cap(cfg_cap);
    auto res_cap = mcts_cap.search(game, 20, dummy_eval);
    TEST_CHECK(!mcts_cap.has_retained_root(), "Tree exceeding max_nodes must be reset after search");
    TEST_CHECK(mcts_cap.live_node_count() == 0, "Live nodes 0 after cap reset");

    // 7. Terminal position handling
    zeropawn::ChessGame fools_mate("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 2");
    zeropawn::MCTS mcts_term(cfg_reuse);
    auto res_term = mcts_term.search(fools_mate, 50, dummy_eval);
    TEST_CHECK(res_term.new_simulations == 0, "Terminal position must execute 0 simulations");
    TEST_CHECK(res_term.best_action == -1, "Terminal position best_action is -1");
    TEST_CHECK(res_term.root_value == -1.0f, "Terminal checkmate value must be -1.0");

    // 8. Memory leak & node count reconciliation
    zeropawn::MCTS::reset_node_alloc_stats();
    {
        zeropawn::MCTS temp_mcts(cfg_reuse);
        temp_mcts.search(game, 150, dummy_eval);
        temp_mcts.reset_root();
    }
    TEST_CHECK(zeropawn::MCTS::total_allocated_nodes() > 0, "Allocations occurred");
    TEST_CHECK(zeropawn::MCTS::total_allocated_nodes() == zeropawn::MCTS::total_freed_nodes(),
               "Total allocated nodes must match total freed nodes after teardown");

    std::cout << "[PASS] test_subtree_reuse\n";
}

void test_mcts_exception_safety() {
    auto dummy_eval = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        int n = g.num_actions();
        std::vector<float> policy(n, 1.0f / static_cast<float>(n));
        return {policy, 0.0f};
    };

    zeropawn::ChessGame game;
    uint64_t root_hash = game.get_hash();

    // 1. Test evaluator exception after virtual losses have been applied to root
    zeropawn::MCTSConfig cfg_reuse;
    cfg_reuse.reuse_tree = true;
    cfg_reuse.seed = 42;
    zeropawn::MCTS mcts(cfg_reuse);

    int eval_calls = 0;
    auto throwing_eval = [&](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        eval_calls++;
        if (g.get_hash() == root_hash) {
            // Root node succeeds
            return dummy_eval(g);
        }
        // First child expansion throws
        throw std::runtime_error("Simulated neural net evaluator failure");
    };

    bool caught = false;
    std::string err_msg;
    try {
        mcts.search(game, 10, throwing_eval);
    } catch (const std::runtime_error& e) {
        caught = true;
        err_msg = e.what();
    } catch (...) {
        caught = true;
        err_msg = "Unknown exception";
    }

    TEST_CHECK(caught, "Original evaluator exception must propagate to caller");
    TEST_CHECK(err_msg == "Simulated neural net evaluator failure", "Original exception message preserved");
    TEST_CHECK(mcts.total_active_virtual_losses() == 0,
               "All virtual losses must be cleanly undone after exception");
    TEST_CHECK(mcts.has_retained_root(), "Retained root preserved with clean stats on reuse_tree=true");

    // 2. Subsequent search on the same MCTS instance does not inherit dirty virtual losses
    auto res = mcts.search(game, 20, dummy_eval);
    TEST_CHECK(res.new_simulations == 20, "Subsequent search executes full budget of new simulations");
    TEST_CHECK(res.best_action >= 0, "Subsequent search produces valid best action");
    TEST_CHECK(mcts.total_active_virtual_losses() == 0, "No active virtual losses after successful search");

    // 3. Disabled-mode exception safety: root must be reset on exception
    zeropawn::MCTSConfig cfg_off;
    cfg_off.reuse_tree = false;
    zeropawn::MCTS mcts_off(cfg_off);
    caught = false;
    try {
        mcts_off.search(game, 10, throwing_eval);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    TEST_CHECK(caught, "Exception propagates in disabled mode");
    TEST_CHECK(!mcts_off.has_retained_root(), "Disabled mode must reset root on exception");
    TEST_CHECK(mcts_off.live_node_count() == 0, "Live node count 0 in disabled mode after exception");

    // 4. Memory reconciliation: verify no node leaks during exception unwinding
    zeropawn::MCTS::reset_node_alloc_stats();
    {
        zeropawn::MCTS temp_mcts(cfg_reuse);
        try {
            temp_mcts.search(game, 5, throwing_eval);
        } catch (...) {}
        temp_mcts.reset_root();
    }
    TEST_CHECK(zeropawn::MCTS::total_allocated_nodes() == zeropawn::MCTS::total_freed_nodes(),
               "Allocated nodes must equal freed nodes after exception recovery");

    std::cout << "[PASS] test_mcts_exception_safety\n";
}

void test_mcts_batching() {
    auto dummy_eval = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        int n = g.num_actions();
        std::vector<float> policy(n, 1.0f / static_cast<float>(n));
        return {policy, 0.0f};
    };

    // 1. Config default verification
    zeropawn::MCTSConfig cfg_default;
    TEST_CHECK(cfg_default.batch_size == 1, "batch_size must default to 1");

    // 2. Budget verification across batch sizes 1, 2, 4, 8 with non-multiple budget (47)
    for (int bs : {1, 2, 4, 8}) {
        zeropawn::MCTSConfig cfg;
        cfg.batch_size = bs;
        cfg.seed = 12345;
        zeropawn::MCTS mcts(cfg);
        zeropawn::ChessGame game;

        auto res = mcts.search(game, 47, dummy_eval);
        TEST_CHECK(res.new_simulations == 47, "Search budget must be exactly 47 regardless of batch size");
        TEST_CHECK(res.batch_size == bs, "SearchResult batch_size must match config batch_size");
        TEST_CHECK(res.best_action >= 0, "Valid best action must be selected");
        TEST_CHECK(mcts.total_active_virtual_losses() == 0, "No active virtual losses after search");
    }

    // 3. Batched evaluation callback invocation check
    {
        zeropawn::MCTSConfig cfg;
        cfg.batch_size = 8;
        cfg.seed = 42;
        zeropawn::MCTS mcts(cfg);
        zeropawn::ChessGame game;

        int total_batch_calls = 0;
        int max_observed_batch = 0;
        zeropawn::BatchEvaluator batch_eval = [&](const std::vector<const zeropawn::Game*>& states)
            -> std::vector<std::pair<std::vector<float>, float>> {
            total_batch_calls++;
            int sz = static_cast<int>(states.size());
            if (sz > max_observed_batch) max_observed_batch = sz;

            std::vector<std::pair<std::vector<float>, float>> out;
            out.reserve(sz);
            for (const auto* s : states) {
                out.push_back(dummy_eval(*s));
            }
            return out;
        };

        auto res = mcts.search(game, 64, dummy_eval, batch_eval);
        TEST_CHECK(res.new_simulations == 64, "Batched search must execute exactly 64 simulations");
        TEST_CHECK(total_batch_calls > 0, "Batch evaluator must be invoked");
        TEST_CHECK(max_observed_batch > 1, "Batch evaluator must receive batches of size > 1");
        TEST_CHECK(mcts.total_active_virtual_losses() == 0, "Virtual losses clean after batch search");
    }

    // 4. Subtree reuse compatibility with batching
    {
        zeropawn::MCTSConfig cfg;
        cfg.reuse_tree = true;
        cfg.batch_size = 4;
        cfg.seed = 999;
        zeropawn::MCTS mcts(cfg);
        zeropawn::ChessGame game;

        auto res1 = mcts.search(game, 60, dummy_eval);
        TEST_CHECK(res1.new_simulations == 60, "First search executes 60 simulations");
        TEST_CHECK(res1.retained_visits == 0, "First search has 0 retained visits");
        TEST_CHECK(mcts.has_retained_root(), "Retained root present with reuse_tree=true");

        int my_act = res1.best_action;
        std::string my_move = game.action_to_string(my_act);
        zeropawn::ChessGame g2;
        g2.make_move_uci(my_move);
        auto opp_moves = g2.legal_moves_uci();
        std::string opp_move = opp_moves[0];
        int opp_act = g2.string_to_action(opp_move);
        g2.make_move_uci(opp_move);

        bool advanced = mcts.advance_root(my_act, opp_act, g2);
        TEST_CHECK(advanced, "advance_root must succeed with subtree reuse under batching");

        auto res2 = mcts.search(g2, 50, dummy_eval);
        TEST_CHECK(res2.new_simulations == 50, "Second search executes exactly 50 new simulations");
        TEST_CHECK(res2.retained_visits > 0, "Second search has > 0 retained visits");
        TEST_CHECK(mcts.total_active_virtual_losses() == 0, "No active virtual losses after reuse search");
    }

    // 5. Exception safety during batched leaf evaluation
    {
        zeropawn::MCTSConfig cfg;
        cfg.reuse_tree = true;
        cfg.batch_size = 4;
        cfg.seed = 42;
        zeropawn::MCTS mcts(cfg);
        zeropawn::ChessGame game;

        int call_count = 0;
        zeropawn::BatchEvaluator throwing_batch_eval = [&](const std::vector<const zeropawn::Game*>& states)
            -> std::vector<std::pair<std::vector<float>, float>> {
            call_count++;
            if (call_count >= 2) {
                throw std::runtime_error("Simulated batched GPU OOM error");
            }
            std::vector<std::pair<std::vector<float>, float>> out;
            for (const auto* s : states) {
                out.push_back(dummy_eval(*s));
            }
            return out;
        };

        bool caught = false;
        try {
            mcts.search(game, 20, dummy_eval, throwing_batch_eval);
        } catch (const std::runtime_error& e) {
            caught = true;
            TEST_CHECK(std::string(e.what()) == "Simulated batched GPU OOM error", "Exception message intact");
        }
        TEST_CHECK(caught, "Batched evaluator exception must propagate to caller");
        TEST_CHECK(mcts.total_active_virtual_losses() == 0,
                   "All virtual losses in batch must be undone on exception");

        // Memory leak check on exception recovery
        zeropawn::MCTS::reset_node_alloc_stats();
        {
            zeropawn::MCTS temp_mcts(cfg);
            int cc = 0;
            zeropawn::BatchEvaluator err_eval = [&](const std::vector<const zeropawn::Game*>& states)
                -> std::vector<std::pair<std::vector<float>, float>> {
                cc++;
                if (cc >= 1) throw std::runtime_error("Immediate failure");
                return {};
            };
            try {
                temp_mcts.search(game, 20, dummy_eval, err_eval);
            } catch (...) {}
            temp_mcts.reset_root();
        }
        TEST_CHECK(zeropawn::MCTS::total_allocated_nodes() == zeropawn::MCTS::total_freed_nodes(),
                   "Allocated nodes must equal freed nodes after batched exception recovery");
    }

    std::cout << "[PASS] test_mcts_batching\n";
}
void test_search_batched() {
    zeropawn::ChessGame game;
    zeropawn::MCTSConfig config;
    config.seed = 42;
    zeropawn::MCTS mcts(config);

    auto batch_eval = [](const std::vector<float>& flat, int n)
        -> std::pair<std::vector<float>, std::vector<float>> {
        std::vector<float> policy(static_cast<size_t>(n) * 4672, 1.0f / 4672.0f);
        std::vector<float> values(static_cast<size_t>(n), 0.0f);
        return {std::move(policy), std::move(values)};
    };

    // 1. Batch size 1
    mcts.set_seed(42);
    auto res_b1 = mcts.search_batched(game, 100, batch_eval, 1);
    float sum1 = 0.0f;
    for (float v : res_b1.action_visits) sum1 += v;
    TEST_CHECK(std::abs(sum1 - 1.0f) < 1e-4f, "Batched B=1 action visits must sum to 1.0");

    // 2. Batch size 16
    mcts.set_seed(42);
    auto res_b16 = mcts.search_batched(game, 100, batch_eval, 16);
    float sum16 = 0.0f;
    for (float v : res_b16.action_visits) sum16 += v;
    TEST_CHECK(std::abs(sum16 - 1.0f) < 1e-4f, "Batched B=16 action visits must sum to 1.0");

    // 3. Mate in 1 with batched search (B=8)
    // Scholar's mate: 1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7#
    zeropawn::ChessGame mate_game("r1bqkb1r/pppp1ppp/2n5/4p3/2B1n3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1");
    mcts.set_seed(12345);
    auto mate_res = mcts.search_batched(mate_game, 100, batch_eval, 8);
    std::string mate_move = mate_game.action_to_string(mate_res.best_action);
    TEST_CHECK(mate_move == "f3f7", "Batched MCTS must find mate in 1 (f3f7)");

    std::cout << "[PASS] test_search_batched\n";
}

void test_apply_action_invalid_raises() {
    // F-4: an undecodable action index must throw (never silently no-op).
    zeropawn::ChessGame game;
    bool raised = false;
    try {
        game.apply_action(-1);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action(-1) must raise std::invalid_argument");

    raised = false;
    try {
        game.apply_action(999999);
    } catch (const std::exception&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action(999999) must raise");

    // State must be untouched and the game still fully usable.
    TEST_CHECK(game.legal_actions().size() == 20, "startpos still has 20 legal actions");
    game.apply_action(game.legal_actions()[0]);
    TEST_CHECK(game.history_len() == 1, "valid action still applies");
    std::cout << "[PASS] test_apply_action_invalid_raises\n";
}

void test_abort_fn_stops_search() {
    // F-3: the abort predicate stops the search and reports actual sims.
    zeropawn::ChessGame game;
    zeropawn::MCTSConfig cfg;
    zeropawn::MCTS mcts(cfg);
    auto eval_fn = [](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
        return {std::vector<float>(g.num_actions(), 1.0f / g.num_actions()), 0.0f};
    };

    int polls = 0;
    zeropawn::AbortFn abort = [&polls]() {
        polls++;
        return polls > 2;  // abort on the 3rd poll (sim 64)
    };
    auto res = mcts.search(game, 10000, eval_fn, nullptr, abort);
    TEST_CHECK(res.new_simulations > 0 && res.new_simulations <= 64, "aborted search reports actual sims (<= 64)");
    TEST_CHECK(polls > 2, "abort predicate was triggered");

    auto full = mcts.search(game, 50, eval_fn);
    TEST_CHECK(full.new_simulations == 50, "unaborted search reports full budget");
    TEST_CHECK(mcts.total_active_virtual_losses() == 0, "no residual virtual loss after abort");
    std::cout << "[PASS] test_abort_fn_stops_search\n";
}

void test_perft_depth_limits() {
    // Remediation F-02: negative and excessive depths must raise instead of
    // recursing (perft(20000) previously overflowed the stack and killed the
    // process; negative depths silently returned 1).
    zeropawn::ChessGame game;
    bool raised = false;
    try {
        game.perft(-1);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "perft(-1) must raise std::invalid_argument");

    raised = false;
    try {
        game.perft(zeropawn::ChessGame::MAX_PERFT_DEPTH + 1);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "perft beyond MAX_PERFT_DEPTH must raise");

    raised = false;
    try {
        game.perft(20000);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "perft(20000) must raise instead of overflowing the stack");

    // Depth 0 is the standard convention (1 node) and the reachable workload
    // is unaffected.
    TEST_CHECK(game.perft(0) == 1ULL, "perft(0) == 1");
    TEST_CHECK(game.perft(4) == 197281ULL, "perft(4) still exact after validation");
    std::cout << "[PASS] test_perft_depth_limits\n";
}

void test_apply_action_illegal_raises() {
    // Remediation F-05: an in-range but ILLEGAL action previously mutated the
    // game silently (pawn teleported a2->a5). It must raise and leave the
    // state untouched; legal actions must still apply.
    zeropawn::ChessGame game;
    int idx = game.string_to_action("a2a5");
    TEST_CHECK(idx >= 0, "a2a5 decodes to an in-range index");
    std::string fen_before = game.get_fen();
    uint64_t hash_before = game.get_hash();
    bool raised = false;
    try {
        game.apply_action(idx);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action(a2a5) must raise on an illegal action");
    TEST_CHECK(game.get_fen() == fen_before, "FEN unchanged after illegal apply_action");
    TEST_CHECK(game.get_hash() == hash_before, "hash unchanged after illegal apply_action");
    TEST_CHECK(game.history_len() == 0, "history unchanged after illegal apply_action");

    // A legal action still applies (hot path preserved for valid callers).
    int legal_idx = game.legal_actions()[0];
    game.apply_action(legal_idx);
    TEST_CHECK(game.history_len() == 1, "legal action still applies");
    std::cout << "[PASS] test_apply_action_illegal_raises\n";
}

void test_connect4_bounds_and_validation() {
    // Remediation F-01: out-of-range cell() indices must raise, not read OOB.
    zeropawn::Connect4 c4;
    bool raised = false;
    try {
        c4.cell(6, 0);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "cell(6,0) must raise");
    raised = false;
    try {
        c4.cell(-1, 0);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "cell(-1,0) must raise");
    raised = false;
    try {
        c4.cell(0, 7);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "cell(0,7) must raise");
    TEST_CHECK(c4.cell(0, 0) == 0, "valid cell access still works");

    // Remediation REM-02: illegal apply_action must raise instead of silently
    // ignoring (out-of-range, full column, finished game).
    raised = false;
    try {
        c4.apply_action(-1);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action(-1) must raise");
    raised = false;
    try {
        c4.apply_action(7);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action(7) must raise");

    zeropawn::Connect4 full;
    // Alternating owners fill column 0 to the top without a vertical win.
    for (int i = 0; i < zeropawn::Connect4::ROWS; ++i) {
        full.apply_action(0);
    }
    TEST_CHECK(!full.is_terminal(), "alternating column fill does not end the game");
    // Column 0 now holds ROWS pieces: applying again must raise.
    raised = false;
    try {
        full.apply_action(0);
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    TEST_CHECK(raised, "apply_action on a full column must raise");
    TEST_CHECK(full.cell(zeropawn::Connect4::ROWS - 1, 0) == 2, "full column top cell unchanged");

    // Remediation REM-01: string_to_action accepts only column digits.
    TEST_CHECK(full.string_to_action("0") == 0, "\"0\" decodes to 0");
    TEST_CHECK(full.string_to_action("6") == 6, "\"6\" decodes to 6");
    TEST_CHECK(full.string_to_action("9") == -1, "\"9\" is undecodable");
    TEST_CHECK(full.string_to_action("abc") == -1, "\"abc\" is undecodable");
    TEST_CHECK(full.string_to_action("") == -1, "\"\" is undecodable");
    TEST_CHECK(full.string_to_action("10") == -1, "multi-char strings are undecodable");
    std::cout << "[PASS] test_connect4_bounds_and_validation\n";
}

void test_mcts_config_validation() {
    // Remediation F-11: semantically invalid configurations must be rejected
    // at MCTS construction instead of silently degrading search.
    using Mutator = void (*)(zeropawn::MCTSConfig&);
    auto expect_raise = [](const char* what, Mutator mutate) {
        zeropawn::MCTSConfig cfg;
        mutate(cfg);
        bool raised = false;
        try {
            zeropawn::MCTS mcts(cfg);
        } catch (const std::invalid_argument&) {
            raised = true;
        }
        TEST_CHECK(raised, what);
    };
    expect_raise("negative c_puct must raise", [](zeropawn::MCTSConfig& c) { c.c_puct = -1.0f; });
    expect_raise("zero dirichlet_alpha must raise", [](zeropawn::MCTSConfig& c) { c.dirichlet_alpha = 0.0f; });
    expect_raise("negative dirichlet_weight must raise", [](zeropawn::MCTSConfig& c) { c.dirichlet_weight = -0.1f; });
    expect_raise("negative fpu_reduction must raise", [](zeropawn::MCTSConfig& c) { c.fpu_reduction = -1.0f; });
    expect_raise("negative virtual_loss must raise", [](zeropawn::MCTSConfig& c) { c.virtual_loss = -2; });
    expect_raise("batch_size 0 must raise", [](zeropawn::MCTSConfig& c) { c.batch_size = 0; });
    expect_raise("max_nodes 0 must raise", [](zeropawn::MCTSConfig& c) { c.max_nodes = 0; });

    // The documented defaults and every legitimate configuration still work.
    zeropawn::MCTSConfig valid;
    valid.c_puct = 0.0f;      // 0 is allowed (pure exploitation)
    valid.batch_size = 64;
    valid.max_nodes = 1;
    zeropawn::MCTS mcts(valid);
    TEST_CHECK(mcts.get_config().batch_size == 64, "valid config constructed");
    std::cout << "[PASS] test_mcts_config_validation\n";
}

void test_chess_rules_remediation() {
    using zeropawn::ChessGame;

    // ------------------------------------------------------------------
    // CR-01: en-passant position identity must not depend on the path.
    // The audit repro: after 1.d4 the black e4 pawn's EP capture is
    // pseudo-legal but illegal (exd3 e.p. opens the Qh4-a4 rank against the
    // a4 king). Play path and FEN path must produce identical state.
    // ------------------------------------------------------------------
    ChessGame play("8/8/8/8/k3p2Q/8/3P4/3K4 w - - 0 1");
    TEST_CHECK(play.make_move_uci("d2d4"), "d2d4 legal");
    ChessGame load(play.get_fen());
    TEST_CHECK(play.get_hash() == load.get_hash(), "CR-01: play/FEN hashes must match (white)");
    TEST_CHECK(play.get_fen() == load.get_fen(), "CR-01: play/FEN FENs must match (white)");
    TEST_CHECK(load.get_fen().find(" d3 ") == std::string::npos,
               "CR-01: illegal EP square must not be serialized");

    // Black-side mirror of the same horizontal-pin family.
    ChessGame play_b("3k4/3p4/8/8/K2pP2q/8/8/8 b - - 0 1");
    TEST_CHECK(play_b.make_move_uci("d7d5"), "d7d5 legal");
    ChessGame load_b(play_b.get_fen());
    TEST_CHECK(play_b.get_hash() == load_b.get_hash(), "CR-01: play/FEN hashes must match (black)");
    TEST_CHECK(play_b.get_fen() == load_b.get_fen(), "CR-01: play/FEN FENs must match (black)");

    // Fully legal EP must still be recorded after a double push.
    ChessGame legal_ep("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1");
    TEST_CHECK(legal_ep.make_move_uci("e2e4"), "e2e4 legal");
    TEST_CHECK(legal_ep.get_fen().find(" e3 ") != std::string::npos,
               "legal EP square is serialized");
    ChessGame legal_ep_load(legal_ep.get_fen());
    TEST_CHECK(legal_ep.get_hash() == legal_ep_load.get_hash(), "legal EP identity across paths");

    // EP state must remain hash-distinguishable when it is legal.
    ChessGame with_ep("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1");
    ChessGame without_ep("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1");
    TEST_CHECK(with_ep.get_hash() != without_ep.get_hash(),
               "legal-EP and no-EP positions must hash differently");

    // FEN round-trip idempotence on the repro family.
    for (const char* fen : {"8/8/8/8/k2Pp2Q/8/8/3K4 b - - 0 1",
                            "3k4/8/8/K2pP2q/8/8/8/8 w - - 0 1",
                            "4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1",
                            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"}) {
        ChessGame g(fen);
        TEST_CHECK(g.get_fen() == fen, "loaded FEN serializes unchanged");
    }

    // ------------------------------------------------------------------
    // CR-02: structurally malformed board layouts are rejected instead of
    // being silently transformed; every rejection preserves prior state.
    // ------------------------------------------------------------------
    ChessGame g;
    g.make_move_uci("e2e4");
    const std::string pre_fen = g.get_fen();
    const uint64_t pre_hash = g.get_hash();
    const char* bad_fens[] = {
        "4k3/8/8/8/8/8/8/qqqqqqqqK w - - 0 1",  // the audit's rank-overflow repro
        "4k3/8/8/8/8/8/8/9q w - - 0 1",          // compression digit 9
        "4k3/8/8/8/8/8/8/0q w - - 0 1",          // compression digit 0
        "4k3/8/8/8/8/8/8/4K2 w - - 0 1",         // short final rank
        "4k3/8/8/8/8/8/4K/8 w - - 0 1",          // short rank mid-board (7 squares)
        "4k3/8/8/8/8/8/8/4K3/8 w - - 0 1",       // 9 ranks
        "4k3/8/8/8/8/8/8 w - - 0 1",             // 7 ranks
        "4k3/8/8/8/8/8/8//4K3 w - - 0 1",        // empty rank
        "4k3/8/8/8/8/8/8/4K3/ w - - 0 1",        // trailing separator
        "4k3/8/8/8/8/8/8/4K3X w - - 0 1",        // invalid piece char
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra",   // trailing garbage field
        "4k3  8/8/8/8/8/8/8/4K3 w - - 0 1",      // consecutive separators
    };
    for (const char* bad : bad_fens) {
        TEST_CHECK(!g.set_fen(bad), "malformed FEN must be rejected");
        TEST_CHECK(g.get_fen() == pre_fen && g.get_hash() == pre_hash,
                   "rejected FEN must not mutate state");
    }
    // Valid structural variants still load (44 == 8 squares of compression).
    TEST_CHECK(g.set_fen("4k3/8/8/8/8/8/44/4K3 w - - 0 1"), "valid compression accepted");
    TEST_CHECK(g.set_fen(pre_fen), "valid FEN re-loads after rejections");

    // ------------------------------------------------------------------
    // REM-CR-08 (found during this remediation): a king-less FEN carrying
    // castling rights for the king-less side made the vendored setFen
    // dereference a missing king (undefined lsb of an empty bitboard) while
    // initializing the castling-path table -- flaky out-of-bounds access
    // and segfault (~70% per attempt on the pre-remediation build). The
    // structural gate must reject king-less and multi-king placements
    // BEFORE the vendored parser runs. The first FEN is the fuzz killer:
    // Kiwipete with the black king replaced by a white king.
    // ------------------------------------------------------------------
    const char* king_faults[] = {
        "r3K2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "4k3/8/8/8/8/8/8/8 w KQkq - 0 1",
        "8/8/8/8/8/8/8/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/K3K3 w - - 0 1",
        "8/8/8/8/8/8/8/8 w - - 0 1",
    };
    for (const char* bad : king_faults) {
        ChessGame k;
        TEST_CHECK(!k.set_fen(bad), "king-less/multi-king FEN must be rejected pre-parse");
        TEST_CHECK(k.get_hash() == ChessGame().get_hash(), "rejected FEN must not mutate state");
    }

    // ------------------------------------------------------------------
    // CR-03: negative, malformed, or out-of-range counters are rejected;
    // representable boundaries are accepted.
    // ------------------------------------------------------------------
    const char* bad_counters[] = {
        "4k3/8/8/8/8/8/8/4K3 w - - -5 1",
        "4k3/8/8/8/8/8/8/4K3 b - - 0 -3",
        "4k3/8/8/8/8/8/8/4K3 w - - 256 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 32769",
        "4k3/8/8/8/8/8/8/4K3 w - - 1x 1",
        "4k3/8/8/8/8/8/8/4K3 w - - x1 1",
        "4k3/8/8/8/8/8/8/4K3 w - - +5 1",
        "4k3/8/8/8/8/8/8/4K3 w - -  1",  // empty halfmove field (double space)
    };
    TEST_CHECK(g.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "clean FEN loads");
    const std::string counter_pre_fen = g.get_fen();
    for (const char* bad : bad_counters) {
        TEST_CHECK(!g.set_fen(bad), "invalid counter must be rejected");
        TEST_CHECK(g.get_fen() == counter_pre_fen, "rejected counter must not mutate state");
    }
    TEST_CHECK(g.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), "halfmove 0 accepted");
    TEST_CHECK(g.set_fen("4k3/8/8/8/8/8/8/4K3 w - - 255 32768"), "counter upper bounds accepted");
    TEST_CHECK(g.half_move_clock() == 255 && g.full_move_number() == 32768,
               "boundary counters stored exactly");

    // ------------------------------------------------------------------
    // CR-04: insufficient material must classify the multi-bishop dead
    // positions correctly without false positives on playable material.
    // ------------------------------------------------------------------
    const struct {
        const char* fen;
        bool insufficient;
    } material_cases[] = {
        {"8/8/8/4k3/2b5/8/8/4KB1B w - - 0 1", true},   // audit repro: K+2 light B vs K+light B
        {"8/8/8/4k3/8/8/8/4KB1B w - - 0 1", true},     // K+2 light bishops vs K
        {"8/8/8/4k3/2b5/8/8/4KB2 w - - 0 1", true},    // KB vs KB same color
        {"8/8/8/4k3/2b5/8/8/4K1B1 w - - 0 1", false},  // KB vs KB opposite color
        {"8/8/8/4k3/2n5/8/8/4K1N1 w - - 0 1", false},  // KN vs KN (selfmate material)
        {"8/8/8/4k3/8/8/8/4KN2 w - - 0 1", true},      // KN vs K
        {"8/8/8/4k3/8/8/8/4KNN1 w - - 0 1", false},    // true KNN vs K (helpmate exists)
        {"8/8/8/4k3/2n5/8/8/4KB1B w - - 0 1", false},  // KBB vs KN (knight can selfmate)
        {"8/8/8/4k3/8/8/8/4KBB1 w - - 0 1", false},    // opposite-colored bishop pair
        {"8/8/8/4k3/2q5/8/8/4K1N1 w - - 0 1", false},  // KN vs KQ
        {"8/8/8/4k3/2r5/8/8/4K1N1 w - - 0 1", false},  // KN vs KR
        {"8/8/8/4k3/8/8/4P3/4K3 w - - 0 1", false},    // KP vs K
        {"8/8/8/4k3/8/8/8/4KR2 w - - 0 1", false},     // KR vs K
        {"8/8/8/4k3/8/8/8/4KBN1 w - - 0 1", false},    // KBN vs K
        {"8/8/8/4k3/8/8/8/4K3 w - - 0 1", true},       // K vs K
    };
    for (const auto& mc : material_cases) {
        ChessGame m(mc.fen);
        TEST_CHECK(m.is_insufficient_material() == mc.insufficient,
                   mc.fen);
    }
    std::cout << "[PASS] test_chess_rules_remediation\n";
}

int main() {
    test_stub();
    test_board_basics();
    test_perft();
    test_perft_depth_limits();
    test_terminal();
    test_encoder();
    test_mcts();
    test_subtree_reuse();
    test_mcts_exception_safety();
    test_mcts_batching();
    test_connect4();
    test_connect4_bounds_and_validation();
    test_search_batched();
    test_apply_action_invalid_raises();
    test_apply_action_illegal_raises();
    test_abort_fn_stops_search();
    test_mcts_config_validation();
    test_chess_rules_remediation();
    std::cout << "\nAll C++ tests passed successfully.\n";
    return 0;
}
