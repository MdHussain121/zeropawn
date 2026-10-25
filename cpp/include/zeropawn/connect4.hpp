#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <array>
#include "zeropawn/game.hpp"

namespace zeropawn {

class Connect4 : public Game {
public:
    static constexpr int ROWS = 6;
    static constexpr int COLS = 7;
    static constexpr int NUM_CELLS = ROWS * COLS; // 42

    Connect4();

    std::unique_ptr<Game> clone() const override {
        return std::make_unique<Connect4>(*this);
    }

    std::vector<int> legal_actions() const override;
    void apply_action(int col) override;
    bool is_terminal() const override;
    float terminal_value() const override;
    uint64_t get_hash() const override;
    int num_planes() const override { return 3; }
    int state_size() const override { return 3 * ROWS * COLS; }
    int num_actions() const override { return COLS; } // 7
    void encode_state(float* out) const override;
    std::string action_to_string(int action) const override;
    int string_to_action(const std::string& str) const override;

    int current_player() const { return current_player_; }
    int cell(int r, int c) const { return grid_[r][c]; }
    int winner() const { return winner_; } // 0: none, 1: player 1, 2: player 2, -1: draw

private:
    std::array<std::array<int8_t, COLS>, ROWS> grid_; // 0: empty, 1: P1, 2: P2
    std::array<int8_t, COLS> heights_;
    int8_t current_player_ = 1; // 1 or 2
    int8_t winner_ = 0; // 0: ongoing, 1: P1 won, 2: P2 won, -1: draw
    int moves_count_ = 0;
    uint64_t hash_ = 0;

    bool check_win_at(int r, int c, int player) const;
    void recompute_hash();
};

} // namespace zeropawn
