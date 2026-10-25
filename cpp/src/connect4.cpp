#include "zeropawn/connect4.hpp"
#include <cstring>
#include <algorithm>

namespace zeropawn {

// Zobrist-like random table for Connect 4 hashing
static const uint64_t C4_ZOBRIST[2][6][7] = {
    { // Player 1
        {0x9d39247e33776d41ULL, 0x2af7398005aaa5c7ULL, 0x44db015024623547ULL, 0x9c15f73e62a76ae2ULL, 0x75834465489c0c89ULL, 0x3290ac3a203001bfULL, 0x0fbbad1f61042279ULL},
        {0xe83a908ff2fb60caULL, 0x0d7e765d58755c10ULL, 0x1a083822ceafe02dULL, 0x9605d5f0e25ec3b0ULL, 0xd021ff5cd13a2ed5ULL, 0x40bdf15d4a672e32ULL, 0x011355146fd56395ULL},
        {0x5db4832046f3d9e5ULL, 0x239f8b2d7ff719ccULL, 0x05d1a1ae85b49aa1ULL, 0x679f848f6e8fc971ULL, 0x7449bbff801fed0bULL, 0x7d11cdb1c3b7adf0ULL, 0x82c7709e781eb7ccULL},
        {0xf3218f1c9510786cULL, 0x331478f3af51bbe6ULL, 0x4bb38de5e7219443ULL, 0xaa649c6ebcfd50fcULL, 0xd1eb3eb555df3296ULL, 0xee21d12bcf3166d4ULL, 0x5a58cbe7a01d51a6ULL},
        {0x5ca9c687e83dbbfaULL, 0x7f113a36279f532aULL, 0x338f9b96eb96ff76ULL, 0x9a365dfb119d6daeULL, 0xee8e8608d0df6288ULL, 0x4aa2ec96ef743cfcULL, 0x2e35a3ab11cae5e0ULL},
        {0xa4d416b251239c4fULL, 0x48bb8fe91f65bb1bULL, 0x08be14f9d0c641f9ULL, 0x88db09c8646b5a32ULL, 0xd19f8623ad94e8a1ULL, 0x794f8373ee8555e0ULL, 0x6d9f7535ad019ab7ULL}
    },
    { // Player 2
        {0x7c73ff0552b4859aULL, 0x11029e96e62ba132ULL, 0x4f016e4511daec13ULL, 0x18db7421cb45aa71ULL, 0x651c68676d6f228bULL, 0x78a1ee9e2178ffcaULL, 0x39a113d09a259c77ULL},
        {0x92c28646f90547aaULL, 0xbb0302b11e2f754cULL, 0xcb9c66e22fbb95a7ULL, 0xaa9e02c5f111894bULL, 0x56a6ec15df61a9ecULL, 0x33b1e32dca91ddc2ULL, 0x16b046039ea8ba5cULL},
        {0x08779944e832049eULL, 0x7508dae76cba9332ULL, 0xbb732009ddb2f56bULL, 0x11cb93df45ecba01ULL, 0xaa2755530188b488ULL, 0xd88bb3e5271a084cULL, 0x38ef55913ca01288ULL},
        {0x44bf67e335552a44ULL, 0x933011ee67bfa392ULL, 0x96efca9411ea2b61ULL, 0x10d0577bf610368bULL, 0x88bb37e108cbff44ULL, 0x01daee387fa054eeULL, 0x2610bb4e11cb02bbULL},
        {0x83e200bc54a0f443ULL, 0x17db4422e171bca4ULL, 0x47bca8817a22ea67ULL, 0x77bca22e1188dca1ULL, 0x892015fa7661ee20ULL, 0xbb5589c336bba197ULL, 0x52ca913076ba012bULL},
        {0x13dfb9315577ca22ULL, 0x68ecaa29015ba677ULL, 0x82bca88310baee19ULL, 0x228fae9811bb470aULL, 0x44a29e46a188f401ULL, 0xbb509a2123880455ULL, 0x992fa68b77054ae3ULL}
    }
};

static const uint64_t C4_TURN_KEY = 0x89abcdef01234567ULL;

Connect4::Connect4() {
    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            grid_[r][c] = 0;
        }
    }
    heights_.fill(0);
    current_player_ = 1;
    winner_ = 0;
    moves_count_ = 0;
    hash_ = 0;
}

std::vector<int> Connect4::legal_actions() const {
    std::vector<int> actions;
    if (winner_ != 0) return actions;
    for (int c = 0; c < COLS; ++c) {
        if (heights_[c] < ROWS) {
            actions.push_back(c);
        }
    }
    return actions;
}

bool Connect4::check_win_at(int r, int c, int p) const {
    static const int dirs[4][2] = {
        {0, 1},  // Horizontal
        {1, 0},  // Vertical
        {1, 1},  // Diagonal up-right
        {1, -1}  // Diagonal down-right
    };

    for (int d = 0; d < 4; ++d) {
        int dr = dirs[d][0];
        int dc = dirs[d][1];
        int count = 1;

        // Forward
        int nr = r + dr;
        int nc = c + dc;
        while (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid_[nr][nc] == p) {
            count++;
            nr += dr;
            nc += dc;
        }

        // Backward
        nr = r - dr;
        nc = c - dc;
        while (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && grid_[nr][nc] == p) {
            count++;
            nr -= dr;
            nc -= dc;
        }

        if (count >= 4) return true;
    }
    return false;
}

void Connect4::apply_action(int col) {
    if (col < 0 || col >= COLS || heights_[col] >= ROWS || winner_ != 0) return;

    int r = heights_[col]++;
    grid_[r][col] = current_player_;
    hash_ ^= C4_ZOBRIST[current_player_ - 1][r][col];
    moves_count_++;

    if (check_win_at(r, col, current_player_)) {
        winner_ = current_player_;
    } else if (moves_count_ == NUM_CELLS) {
        winner_ = -1; // Draw
    }

    current_player_ = 3 - current_player_;
    hash_ ^= C4_TURN_KEY;
}

bool Connect4::is_terminal() const {
    return winner_ != 0;
}

float Connect4::terminal_value() const {
    if (winner_ == -1) return 0.0f; // Draw
    if (winner_ == 1 || winner_ == 2) {
        // Player who just moved won.
        // Current player to move is the loser!
        return -1.0f;
    }
    return 0.0f;
}

uint64_t Connect4::get_hash() const {
    return hash_;
}

void Connect4::encode_state(float* out) const {
    int plane_size = ROWS * COLS;
    std::memset(out, 0, 3 * plane_size * sizeof(float));

    int own = current_player_;
    int opp = 3 - current_player_;

    for (int r = 0; r < ROWS; ++r) {
        for (int c = 0; c < COLS; ++c) {
            int idx = r * COLS + c;
            if (grid_[r][c] == own) {
                out[0 * plane_size + idx] = 1.0f;
            } else if (grid_[r][c] == opp) {
                out[1 * plane_size + idx] = 1.0f;
            }
            out[2 * plane_size + idx] = 1.0f; // Turn plane
        }
    }
}

std::string Connect4::action_to_string(int action) const {
    return std::to_string(action);
}

int Connect4::string_to_action(const std::string& str) const {
    if (str.empty()) return -1;
    return str[0] - '0';
}

} // namespace zeropawn
