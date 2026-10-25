#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <cstring>
#include <stdexcept>
#include "zeropawn/stub.h"
#include "zeropawn/game.hpp"
#include "zeropawn/chess_game.hpp"
#include "zeropawn/encoder.hpp"
#include "zeropawn/mcts.hpp"
#include "zeropawn/connect4.hpp"

namespace py = pybind11;

py::array_t<float> py_encode_state(const zeropawn::ChessGame& game) {
    py::array_t<float> result({zeropawn::NUM_PLANES, zeropawn::BOARD_SIZE, zeropawn::BOARD_SIZE});
    py::buffer_info buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    zeropawn::encode_state(game, ptr);
    return result;
}

PYBIND11_MODULE(zeropawn_cpp, m) {
    m.doc() = "ZeroPawn C++ core — board, move gen, MCTS";
    m.def("version", &zeropawn::version, "Return library version string");

    // Game base class
    py::class_<zeropawn::Game, std::unique_ptr<zeropawn::Game>>(m, "Game")
        .def("clone", &zeropawn::Game::clone)
        .def("legal_actions", &zeropawn::Game::legal_actions)
        .def("apply_action", &zeropawn::Game::apply_action, py::arg("action"))
        .def("is_terminal", &zeropawn::Game::is_terminal)
        .def("terminal_value", &zeropawn::Game::terminal_value)
        .def("get_hash", &zeropawn::Game::get_hash)
        .def("num_planes", &zeropawn::Game::num_planes)
        .def("state_size", &zeropawn::Game::state_size)
        .def("num_actions", &zeropawn::Game::num_actions)
        .def("action_to_string", &zeropawn::Game::action_to_string, py::arg("action"))
        .def("string_to_action", &zeropawn::Game::string_to_action, py::arg("uci"));

    // ChessGame class
    py::class_<zeropawn::ChessGame, zeropawn::Game>(m, "ChessGame")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("fen"))
        .def("set_fen", &zeropawn::ChessGame::set_fen, py::arg("fen"))
        .def("get_fen", &zeropawn::ChessGame::get_fen)
        .def("legal_moves_uci", &zeropawn::ChessGame::legal_moves_uci)
        .def("is_legal", &zeropawn::ChessGame::is_legal, py::arg("uci"))
        .def("make_move_uci", &zeropawn::ChessGame::make_move_uci, py::arg("uci"))
        .def("unmake_move", &zeropawn::ChessGame::unmake_move)
        .def("get_hash", &zeropawn::ChessGame::get_hash)
        .def("side_to_move", &zeropawn::ChessGame::side_to_move)
        .def("in_check", &zeropawn::ChessGame::in_check)
        .def("is_game_over", &zeropawn::ChessGame::is_game_over)
        .def("is_checkmate", &zeropawn::ChessGame::is_checkmate)
        .def("is_stalemate", &zeropawn::ChessGame::is_stalemate)
        .def("is_repetition", &zeropawn::ChessGame::is_repetition, py::arg("count") = 2)
        .def("is_50_move_rule", &zeropawn::ChessGame::is_50_move_rule)
        .def("is_insufficient_material", &zeropawn::ChessGame::is_insufficient_material)
        .def("half_move_clock", &zeropawn::ChessGame::half_move_clock)
        .def("full_move_number", &zeropawn::ChessGame::full_move_number)
        .def("perft", &zeropawn::ChessGame::perft, py::arg("depth"));

    // Connect4 class
    py::class_<zeropawn::Connect4, zeropawn::Game>(m, "Connect4")
        .def(py::init<>())
        .def("current_player", &zeropawn::Connect4::current_player)
        .def("cell", &zeropawn::Connect4::cell, py::arg("r"), py::arg("c"))
        .def("winner", &zeropawn::Connect4::winner)
        .def("encode_state", [](const zeropawn::Connect4& c4) {
            py::array_t<float> res({c4.num_planes(), zeropawn::Connect4::ROWS, zeropawn::Connect4::COLS});
            py::buffer_info buf = res.request();
            c4.encode_state(static_cast<float*>(buf.ptr));
            return res;
        });

    // Encoders
    m.def("encode_state", &py_encode_state, py::arg("game"), "Encode board state to 19x8x8 numpy array");
    m.def("move_to_index", &zeropawn::move_to_index, py::arg("game"), py::arg("uci"), "Encode move to 0..4671 index");
    m.def("index_to_move", &zeropawn::index_to_move, py::arg("game"), py::arg("index"), "Decode index to UCI move");

    // MCTS Config
    py::class_<zeropawn::MCTSConfig>(m, "MCTSConfig")
        .def(py::init<>())
        .def_readwrite("c_puct", &zeropawn::MCTSConfig::c_puct)
        .def_readwrite("dirichlet_alpha", &zeropawn::MCTSConfig::dirichlet_alpha)
        .def_readwrite("dirichlet_weight", &zeropawn::MCTSConfig::dirichlet_weight)
        .def_readwrite("fpu_reduction", &zeropawn::MCTSConfig::fpu_reduction)
        .def_readwrite("virtual_loss", &zeropawn::MCTSConfig::virtual_loss)
        .def_readwrite("add_noise", &zeropawn::MCTSConfig::add_noise)
        .def_readwrite("seed", &zeropawn::MCTSConfig::seed)
        .def_readwrite("reuse_tree", &zeropawn::MCTSConfig::reuse_tree)
        .def_readwrite("max_nodes", &zeropawn::MCTSConfig::max_nodes)
        .def_readwrite("batch_size", &zeropawn::MCTSConfig::batch_size);

    // Search Result
    py::class_<zeropawn::SearchResult>(m, "SearchResult")
        .def_readonly("best_action", &zeropawn::SearchResult::best_action)
        .def_property_readonly("action_visits", [](const zeropawn::SearchResult& r) {
            // Return an owned copy (memcpy) — never a view into C++ memory.
            py::array_t<float> arr(r.action_visits.size());
            if (!r.action_visits.empty()) {
                std::memcpy(arr.mutable_data(), r.action_visits.data(),
                            r.action_visits.size() * sizeof(float));
            }
            return arr;
        })
        .def_readonly("root_value", &zeropawn::SearchResult::root_value)
        .def_readonly("retained_visits", &zeropawn::SearchResult::retained_visits)
        .def_readonly("new_simulations", &zeropawn::SearchResult::new_simulations)
        .def_readonly("batch_size", &zeropawn::SearchResult::batch_size);

    // MCTS Class
    py::class_<zeropawn::MCTS>(m, "MCTS")
        .def(py::init<const zeropawn::MCTSConfig&>(), py::arg("config") = zeropawn::MCTSConfig{})
        .def("clear", &zeropawn::MCTS::clear)
        .def("set_seed", &zeropawn::MCTS::set_seed, py::arg("seed"))
        .def("cache_size", &zeropawn::MCTS::cache_size)
        .def("advance_root", py::overload_cast<int, int, const zeropawn::Game&>(&zeropawn::MCTS::advance_root),
             py::arg("my_action"), py::arg("opp_action"), py::arg("new_state"))
        .def("advance_root", py::overload_cast<int, const zeropawn::Game&>(&zeropawn::MCTS::advance_root),
             py::arg("action"), py::arg("new_state"))
        .def("reset_root", &zeropawn::MCTS::reset_root)
        .def("live_node_count", &zeropawn::MCTS::live_node_count)
        .def("has_retained_root", &zeropawn::MCTS::has_retained_root)
        .def("retained_visits", &zeropawn::MCTS::retained_visits)
        .def("total_active_virtual_losses", &zeropawn::MCTS::total_active_virtual_losses)
        .def_static("total_allocated_nodes", &zeropawn::MCTS::total_allocated_nodes)
        .def_static("total_freed_nodes", &zeropawn::MCTS::total_freed_nodes)
        .def_static("reset_node_alloc_stats", &zeropawn::MCTS::reset_node_alloc_stats)
        .def("search", [](zeropawn::MCTS& mcts, const zeropawn::Game& game, int num_simulations,
                          py::function eval_fn, py::object batch_eval_fn = py::none()) {
            zeropawn::Evaluator evaluator = [&eval_fn, &game](const zeropawn::Game& g) -> std::pair<std::vector<float>, float> {
                py::gil_scoped_acquire acquire;
                py::object res = eval_fn(&g);
                auto tuple = res.cast<py::tuple>();
                if (tuple.size() != 2) {
                    throw std::runtime_error("evaluator must return a (policy, value) tuple of len 2");
                }
                std::vector<float> policy;
                if (py::isinstance<py::array_t<float>>(tuple[0])) {
                    auto arr = tuple[0].cast<py::array_t<float, py::array::c_style>>();
                    if (static_cast<int>(arr.size()) != g.num_actions()) {
                        throw std::runtime_error("evaluator policy size must equal game.num_actions()");
                    }
                    const float* ptr = arr.data();
                    policy.assign(ptr, ptr + arr.size());
                } else {
                    policy = tuple[0].cast<std::vector<float>>();
                    if (static_cast<int>(policy.size()) != g.num_actions()) {
                        throw std::runtime_error("evaluator policy size must equal game.num_actions()");
                    }
                }
                float val = tuple[1].cast<float>();
                return {std::move(policy), val};
            };

            zeropawn::BatchEvaluator batch_evaluator = nullptr;
            if (!batch_eval_fn.is_none()) {
                batch_evaluator = [&batch_eval_fn](const std::vector<const zeropawn::Game*>& states)
                    -> std::vector<std::pair<std::vector<float>, float>> {
                    py::gil_scoped_acquire acquire;
                    py::list py_games;
                    for (const auto* s : states) {
                        py_games.append(py::cast(const_cast<zeropawn::Game*>(s), py::return_value_policy::reference));
                    }
                    py::object res = batch_eval_fn(py_games);
                    auto list_res = res.cast<py::list>();
                    if (list_res.size() != states.size()) {
                        throw std::runtime_error("batch_evaluator must return list of same length as input states");
                    }
                    std::vector<std::pair<std::vector<float>, float>> out;
                    out.reserve(list_res.size());
                    for (size_t i = 0; i < list_res.size(); ++i) {
                        auto tuple = list_res[i].cast<py::tuple>();
                        if (tuple.size() != 2) {
                            throw std::runtime_error("batch_evaluator element must be (policy, value) tuple");
                        }
                        std::vector<float> policy;
                        const auto* g = states[i];
                        if (py::isinstance<py::array_t<float>>(tuple[0])) {
                            auto arr = tuple[0].cast<py::array_t<float, py::array::c_style>>();
                            if (static_cast<int>(arr.size()) != g->num_actions()) {
                                throw std::runtime_error("evaluator policy size must equal game.num_actions()");
                            }
                            const float* ptr = arr.data();
                            policy.assign(ptr, ptr + arr.size());
                        } else {
                            policy = tuple[0].cast<std::vector<float>>();
                            if (static_cast<int>(policy.size()) != g->num_actions()) {
                                throw std::runtime_error("evaluator policy size must equal game.num_actions()");
                            }
                        }
                        float val = tuple[1].cast<float>();
                        out.emplace_back(std::move(policy), val);
                    }
                    return out;
                };
            }

            return mcts.search(game, num_simulations, evaluator, batch_evaluator);
        }, py::arg("game"), py::arg("num_simulations"), py::arg("evaluator"),
           py::arg("batch_evaluator") = py::none(), py::call_guard<py::gil_scoped_release>())
        .def("search_batched", [](zeropawn::MCTS& mcts, const zeropawn::Game& game, int num_simulations, py::object arg3, py::object arg4) {
            py::function batch_fn;
            int batch_size = 16;
            if (py::isinstance<py::function>(arg3) && py::isinstance<py::int_>(arg4)) {
                batch_fn = arg3.cast<py::function>();
                batch_size = arg4.cast<int>();
            } else if (py::isinstance<py::int_>(arg3) && py::isinstance<py::function>(arg4)) {
                batch_size = arg3.cast<int>();
                batch_fn = arg4.cast<py::function>();
            } else {
                throw py::type_error("search_batched expects (game, num_simulations, batch_size, batch_evaluator) or (game, num_simulations, batch_evaluator, batch_size)");
            }

            int state_size = game.state_size();
            int num_actions = game.num_actions();
            zeropawn::SearchResult result;
            {
                zeropawn::RawBatchEvaluator batch_eval =
                    [&batch_fn, state_size, num_actions](const std::vector<float>& flat, int n)
                    -> std::pair<std::vector<float>, std::vector<float>> {
                    py::gil_scoped_acquire acquire;
                    py::array_t<float> py_planes({n, state_size});
                    std::memcpy(py_planes.mutable_data(), flat.data(),
                                static_cast<size_t>(n) * static_cast<size_t>(state_size) * sizeof(float));

                    py::object py_res = batch_fn(py_planes);
                    auto tuple = py_res.cast<py::tuple>();
                    if (tuple.size() != 2) {
                        throw std::runtime_error("batch_evaluator must return (policy_array, value_array)");
                    }

                    auto py_policy = tuple[0].cast<py::array_t<float, py::array::c_style>>();
                    auto py_vals = tuple[1].cast<py::array_t<float, py::array::c_style>>();

                    std::vector<float> policy(static_cast<size_t>(n) * static_cast<size_t>(num_actions));
                    size_t p_size = std::min(policy.size(), static_cast<size_t>(py_policy.size()));
                    std::memcpy(policy.data(), py_policy.data(), p_size * sizeof(float));

                    std::vector<float> values(static_cast<size_t>(n), 0.0f);
                    size_t v_size = std::min(values.size(), static_cast<size_t>(py_vals.size()));
                    std::memcpy(values.data(), py_vals.data(), v_size * sizeof(float));

                    return {std::move(policy), std::move(values)};
                };
                py::gil_scoped_release release;
                result = mcts.search_batched(game, num_simulations, batch_eval, batch_size);
            }
            return result;
        }, py::arg("game"), py::arg("num_simulations"), py::arg("arg3"), py::arg("arg4"));
}
