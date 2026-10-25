#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "zeropawn/chess_game.hpp"

namespace zeropawn {

constexpr int NUM_PLANES = 19;
constexpr int BOARD_SIZE = 8;
constexpr int NUM_SQUARES = 64;
constexpr int STATE_SIZE = NUM_PLANES * NUM_SQUARES; // 1216 floats
constexpr int NUM_ACTIONS = 4672; // 64 * 73

// State encoding (19 planes, 8x8)
void encode_state(const ChessGame& game, float* out);
std::vector<float> encode_state_vec(const ChessGame& game);

// Action encoding (4672 actions)
int move_to_index(const ChessGame& game, const std::string& uci);
std::string index_to_move(const ChessGame& game, int index);

} // namespace zeropawn
