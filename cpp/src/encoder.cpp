#include "zeropawn/encoder.hpp"
#include <algorithm>
#include <cstring>
#include <utility>

namespace zeropawn {

void encode_state(const ChessGame& game, float* out) {
    std::memset(out, 0, STATE_SIZE * sizeof(float));

    const auto& board = game.raw_board();
    chess::Color us = board.sideToMove();
    chess::Color them = ~us;
    bool white_to_move = (us == chess::Color::WHITE);

    // Planes 0-11: Pieces
    for (int sq = 0; sq < 64; ++sq) {
        auto p = board.at(chess::Square(sq));
        if (p != chess::Piece::NONE) {
            int rank = sq / 8;
            int file = sq % 8;
            int r = white_to_move ? rank : (7 - rank);
            int c = file;
            int target_plane = -1;
            if (p.color() == us) {
                target_plane = static_cast<int>(p.type()); // 0-5
            } else {
                target_plane = 6 + static_cast<int>(p.type()); // 6-11
            }
            out[target_plane * 64 + r * 8 + c] = 1.0f;
        }
    }

    // Planes 12-15: Castling rights
    auto cr = board.castlingRights();
    bool own_ks = cr.has(us, chess::Board::CastlingRights::Side::KING_SIDE);
    bool own_qs = cr.has(us, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    bool opp_ks = cr.has(them, chess::Board::CastlingRights::Side::KING_SIDE);
    bool opp_qs = cr.has(them, chess::Board::CastlingRights::Side::QUEEN_SIDE);

    if (own_ks) std::fill_n(&out[12 * 64], 64, 1.0f);
    if (own_qs) std::fill_n(&out[13 * 64], 64, 1.0f);
    if (opp_ks) std::fill_n(&out[14 * 64], 64, 1.0f);
    if (opp_qs) std::fill_n(&out[15 * 64], 64, 1.0f);

    // Plane 16: En passant square
    auto ep = board.enpassantSq();
    if (ep != chess::Square::NO_SQ && ep.is_valid()) {
        int sq = ep.index();
        int rank = sq / 8;
        int file = sq % 8;
        int r = white_to_move ? rank : (7 - rank);
        int c = file;
        out[16 * 64 + r * 8 + c] = 1.0f;
    }

    // Plane 17: Halfmove clock / 100.0
    float hf = static_cast<float>(board.halfMoveClock()) / 100.0f;
    std::fill_n(&out[17 * 64], 64, hf);

    // Plane 18: Repetition flag
    if (game.is_repetition(1)) {
        std::fill_n(&out[18 * 64], 64, 1.0f);
    }
}

std::vector<float> encode_state_vec(const ChessGame& game) {
    std::vector<float> vec(STATE_SIZE);
    encode_state(game, vec.data());
    return vec;
}

int move_to_index(const ChessGame& game, const std::string& uci) {
    if (uci.length() < 4) return -1;
    bool white_to_move = game.side_to_move();

    int from_file = uci[0] - 'a';
    int from_rank = uci[1] - '1';
    int to_file = uci[2] - 'a';
    int to_rank = uci[3] - '1';
    char promo = (uci.length() > 4) ? uci[4] : '\0';

    int from_r = white_to_move ? from_rank : (7 - from_rank);
    int from_c = from_file;
    int to_r = white_to_move ? to_rank : (7 - to_rank);
    int to_c = to_file;

    int dy = to_r - from_r;
    int dx = to_c - from_c;
    int from_s = from_r * 8 + from_c;

    int plane = -1;

    // Check underpromotion first (n, b, r)
    if (promo == 'n' || promo == 'b' || promo == 'r') {
        int base = 64;
        if (promo == 'b') base = 67;
        else if (promo == 'r') base = 70;
        int dir_offset = dx + 1; // dx can be -1 (left), 0 (center), +1 (right)
        if (dir_offset >= 0 && dir_offset <= 2 && dy == 1) {
            plane = base + dir_offset;
        }
    } else {
        // Check knight moves (8 offsets)
        static const std::pair<int, int> knight_deltas[8] = {
            {+2, +1}, {+1, +2}, {-1, +2}, {-2, +1},
            {-2, -1}, {-1, -2}, {+1, -2}, {+2, -1}
        };
        for (int k = 0; k < 8; ++k) {
            if (dy == knight_deltas[k].first && dx == knight_deltas[k].second) {
                plane = 56 + k;
                break;
            }
        }

        // If not a knight move, check queen-style ray moves (8 directions)
        if (plane == -1) {
            int dir_id = -1;
            int dist = 0;

            if (dx == 0 && dy > 0) { dir_id = 0; dist = dy; }
            else if (dx > 0 && dy > 0 && dx == dy) { dir_id = 1; dist = dy; }
            else if (dy == 0 && dx > 0) { dir_id = 2; dist = dx; }
            else if (dx > 0 && dy < 0 && dx == -dy) { dir_id = 3; dist = dx; }
            else if (dx == 0 && dy < 0) { dir_id = 4; dist = -dy; }
            else if (dx < 0 && dy < 0 && dx == dy) { dir_id = 5; dist = -dx; }
            else if (dy == 0 && dx < 0) { dir_id = 6; dist = -dx; }
            else if (dx < 0 && dy > 0 && -dx == dy) { dir_id = 7; dist = dy; }

            if (dir_id != -1 && dist >= 1 && dist <= 7) {
                plane = dir_id * 7 + (dist - 1);
            }
        }
    }

    if (plane == -1 || from_s < 0 || from_s >= 64) return -1;
    return from_s * 73 + plane;
}

std::string index_to_move(const ChessGame& game, int index) {
    if (index < 0 || index >= NUM_ACTIONS) return "";
    bool white_to_move = game.side_to_move();

    int from_s = index / 73;
    int plane = index % 73;

    int from_r = from_s / 8;
    int from_c = from_s % 8;

    int dx = 0;
    int dy = 0;
    char promo = '\0';

    if (plane < 56) {
        int dir_id = plane / 7;
        int dist = (plane % 7) + 1;
        switch (dir_id) {
            case 0: dy = dist; dx = 0; break;
            case 1: dy = dist; dx = dist; break;
            case 2: dy = 0; dx = dist; break;
            case 3: dy = -dist; dx = dist; break;
            case 4: dy = -dist; dx = 0; break;
            case 5: dy = -dist; dx = -dist; break;
            case 6: dy = 0; dx = -dist; break;
            case 7: dy = dist; dx = -dist; break;
        }
        // Check if pawn promotion to queen
        if (from_r == 6 && (from_r + dy) == 7) {
            int orig_sq = white_to_move ? (from_r * 8 + from_c) : ((7 - from_r) * 8 + from_c);
            auto p = game.raw_board().at(chess::Square(orig_sq));
            if (p.type() == chess::PieceType::PAWN) {
                promo = 'q';
            }
        }
    } else if (plane < 64) {
        int k = plane - 56;
        static const std::pair<int, int> knight_deltas[8] = {
            {+2, +1}, {+1, +2}, {-1, +2}, {-2, +1},
            {-2, -1}, {-1, -2}, {+1, -2}, {+2, -1}
        };
        dy = knight_deltas[k].first;
        dx = knight_deltas[k].second;
    } else {
        int up_idx = plane - 64;
        int piece_type = up_idx / 3;
        int dir_type = up_idx % 3;
        if (piece_type == 0) promo = 'n';
        else if (piece_type == 1) promo = 'b';
        else promo = 'r';

        dy = 1;
        dx = dir_type - 1;
    }

    int to_r = from_r + dy;
    int to_c = from_c + dx;

    if (to_r < 0 || to_r >= 8 || to_c < 0 || to_c >= 8) return "";

    int orig_from_rank = white_to_move ? from_r : (7 - from_r);
    int orig_from_file = from_c;
    int orig_to_rank = white_to_move ? to_r : (7 - to_r);
    int orig_to_file = to_c;

    std::string uci = "";
    uci += static_cast<char>('a' + orig_from_file);
    uci += static_cast<char>('1' + orig_from_rank);
    uci += static_cast<char>('a' + orig_to_file);
    uci += static_cast<char>('1' + orig_to_rank);
    if (promo != '\0') uci += promo;

    return uci;
}

} // namespace zeropawn
