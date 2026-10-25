#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <numeric>
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

int main() {
    test_stub();
    test_board_basics();
    test_perft();
    test_terminal();
    test_encoder();
    test_mcts();
    test_subtree_reuse();
    test_mcts_exception_safety();
    test_mcts_batching();
    test_connect4();
    test_search_batched();
    std::cout << "\nAll C++ tests passed successfully.\n";
    return 0;
}
