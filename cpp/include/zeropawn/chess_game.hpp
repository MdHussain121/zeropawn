#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include "chess.hpp"
#include "zeropawn/game.hpp"

namespace zeropawn {

class ChessGame : public Game {
public:
    ChessGame();
    explicit ChessGame(const std::string& fen);

    bool set_fen(const std::string& fen);
    std::string get_fen() const;

    std::vector<std::string> legal_moves_uci() const;
    bool is_legal(const std::string& uci) const;
    bool make_move_uci(const std::string& uci);
    bool unmake_move();

    uint64_t get_hash() const override;
    bool side_to_move() const; // true: White, false: Black
    bool in_check() const;

    // Terminal state detection
    bool is_game_over() const;
    bool is_checkmate() const;
    bool is_stalemate() const;
    bool is_repetition(int count = 2) const;
    bool is_50_move_rule() const;
    bool is_insufficient_material() const;
    int half_move_clock() const;
    int full_move_number() const;
    // Number of moves applied in this game object (exact ply count, unlike the
    // halfmove-clock estimate derived from a FEN).
    int history_len() const { return static_cast<int>(move_history_.size()); }

    // Game interface overrides
    std::unique_ptr<Game> clone() const override;
    std::vector<int> legal_actions() const override;
    void apply_action(int action) override;
    bool is_terminal() const override;
    float terminal_value() const override;
    int num_planes() const override { return 19; }
    int state_size() const override { return 1216; }
    int num_actions() const override { return 4672; }
    void encode_state(float* out) const override;
    std::string action_to_string(int action) const override;
    int string_to_action(const std::string& str) const override;

    // Direct access to chess::Board
    chess::Board& raw_board() { return board_; }
    const chess::Board& raw_board() const { return board_; }

    uint64_t perft(int depth);

private:
    chess::Board board_;
    std::vector<chess::Move> move_history_;
};

uint64_t compute_perft(chess::Board& board, int depth);

} // namespace zeropawn
