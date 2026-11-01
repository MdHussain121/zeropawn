#include "zeropawn/chess_game.hpp"
#include "zeropawn/encoder.hpp"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace zeropawn {

uint64_t compute_perft(chess::Board& board, int depth) {
    if (depth <= 0) return 1ULL;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const auto& move : moves) {
        // EXACT en-passant policy (remediation CR-01): the vendored default
        // makeMove<false> records the en-passant square whenever an enemy
        // pawn pseudo-legally attacks it, while setFen validates it with
        // full legality -- the same physical position then carried different
        // hashes depending on whether it was reached by play or by FEN load.
        // makeMove<true> is the vendor's own template that applies the same
        // isEpSquareValid legality test at play time, unifying position
        // identity across both paths. EP legality itself (and therefore
        // perft counts) is unaffected: an EP capture the square would have
        // enabled was already rejected by move generation.
        board.makeMove<true>(move);
        nodes += compute_perft(board, depth - 1);
        board.unmakeMove(move);
    }
    return nodes;
}

ChessGame::ChessGame() : board_(chess::constants::STARTPOS) {}

ChessGame::ChessGame(const std::string& fen) : board_(chess::constants::STARTPOS) {
    set_fen(fen);
}

namespace {

// Structural FEN gate (remediation CR-02/CR-03). The vendored parser walks a
// single square cursor over the piece-placement field with no per-rank
// boundary check, so a rank with more than eight squares silently rolls into
// earlier ranks and transforms the input into a different position; its
// counter parsing wraps negative integers into the unsigned storage fields
// (halfmove "-5" became 251, instantly reporting the 50-move draw state; a
// negative fullmove wrapped to an absurd number). This gate runs BEFORE the
// vendored parse so that ChessGame::set_fen either loads the intended
// position or rejects the input without touching any existing state
// (fail-closed, DEF-02 contract). Lenient normalization of the castling and
// en-passant fields is intentionally left to the vendored parser (documented
// CR-05 policy).
//
// Returns an empty string when the FEN is structurally valid, otherwise a
// diagnostic describing the first violation.
std::string validate_fen_structure(std::string_view fen) {
    // Tokenize on exactly one separating space. The vendored splitter turns
    // consecutive spaces into empty tokens (an empty counter field silently
    // defaulted to 0), which is now rejected as ambiguous input instead.
    std::vector<std::string_view> fields;
    size_t start = fen.find_first_not_of(' ');
    if (start == std::string_view::npos) return "empty FEN";
    while (start < fen.size()) {
        size_t end = fen.find(' ', start);
        if (end == std::string_view::npos) end = fen.size();
        fields.push_back(fen.substr(start, end - start));
        if (end == fen.size()) break;
        if (end + 1 < fen.size() && fen[end + 1] == ' ') {
            return "consecutive spaces in FEN";
        }
        start = end + 1;
    }
    if (fields.size() > 6) return "more than 6 FEN fields (trailing garbage)";

    // Piece placement: exactly 8 ranks, each expanding to exactly 8 squares.
    // Kings are counted during the same scan: the vendored setFenCommon
    // initializes its castling-path table with kingSq(c) for BOTH colors
    // unconditionally (before any king validation), and lsb() of an empty
    // king bitboard is undefined -- a FEN without exactly one king per side
    // crashes with an out-of-bounds table access there (flaky segfault,
    // reproduced at ~70% on the pre-remediation build). Gating king counts
    // here means such input never reaches the vendored parser (REM-CR-08).
    const std::string_view placement = fields[0];
    int rank = 0;
    size_t i = 0;
    int white_kings = 0;
    int black_kings = 0;
    while (true) {
        if (rank >= 8) return "more than 8 ranks in piece placement";
        int width = 0;
        while (i < placement.size() && placement[i] != '/') {
            const char c = placement[i];
            if (c >= '1' && c <= '8') {
                width += c - '0';
            } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
                switch (c) {
                    case 'p': case 'n': case 'b': case 'r': case 'q': case 'k':
                    case 'P': case 'N': case 'B': case 'R': case 'Q': case 'K':
                        ++width;
                        if (c == 'K') ++white_kings;
                        if (c == 'k') ++black_kings;
                        break;
                    default:
                        return "invalid piece character in placement";
                }
            } else {
                if (c == '0' || c == '9') {
                    return "invalid rank compression digit in placement";
                }
                return "invalid character in piece placement";
            }
            if (width > 8) return "rank expands past 8 squares";
            ++i;
        }
        if (width != 8) return "rank does not contain exactly 8 squares";
        ++rank;
        if (i == placement.size()) break;
        ++i;  // skip '/'
        if (i == placement.size()) return "empty final rank";
    }
    if (rank != 8) return "fewer than 8 ranks in piece placement";
    if (white_kings != 1 || black_kings != 1) {
        return "piece placement must contain exactly one king per side";
    }

    // Counters (fields 5/6): digits only, within the ranges representable by
    // the vendored storage (halfmove clock uint8, ply counter uint16). A
    // negative or overflowing value would wrap silently and corrupt
    // draw-state behavior.
    const auto valid_counter = [](std::string_view field, long long min_value,
                                  long long max_value) {
        if (field.empty()) return false;
        long long value = 0;
        for (const char c : field) {
            if (c < '0' || c > '9') return false;
            value = value * 10 + (c - '0');
            if (value > max_value) return false;
        }
        return value >= min_value;
    };
    if (fields.size() >= 5 && !valid_counter(fields[4], 0, 255)) {
        return "invalid halfmove clock (must be an integer in [0, 255])";
    }
    if (fields.size() >= 6 && !valid_counter(fields[5], 1, 32768)) {
        return "invalid fullmove number (must be an integer in [1, 32768])";
    }
    return "";
}

}  // namespace

bool ChessGame::set_fen(const std::string& fen) {
    // Structural gate (remediation CR-02/CR-03): reject malformed board
    // layouts and out-of-range counters before the vendored parser runs.
    // Every rejection path below leaves the existing game state untouched.
    if (!validate_fen_structure(fen).empty()) return false;
    chess::Board temp;
    if (!temp.setFen(fen)) return false;
    // Required king invariants: the vendored parser accepts king-less and
    // multi-king FENs (require_kings defaults to false), while every later
    // code path (movegen, search) dereferences exactly one king per side.
    if (temp.pieces(chess::PieceType::KING, chess::Color::WHITE).count() != 1 ||
        temp.pieces(chess::PieceType::KING, chess::Color::BLACK).count() != 1) {
        return false;
    }
    // Rule-illegal position guard (audit C-1): if the side NOT to move is
    // already in check, move generation for the side to move emits king
    // captures; applying one removes a king and the following move generation
    // crashes (access violation). Such positions cannot arise from legal play,
    // so reject them at the boundary instead of letting them reach search.
    const chess::Color stm = temp.sideToMove();
    const chess::Color other = ~stm;
    if (temp.isAttacked(temp.kingSq(other), stm)) return false;
    board_ = temp;
    move_history_.clear();
    return true;
}

std::string ChessGame::get_fen() const {
    return board_.getFen();
}

std::vector<std::string> ChessGame::legal_moves_uci() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    std::vector<std::string> result;
    result.reserve(moves.size());
    for (const auto& m : moves) {
        result.push_back(chess::uci::moveToUci(m));
    }
    return result;
}

static bool is_valid_uci_format(std::string_view uci) {
    if (uci.size() < 4 || uci.size() > 5) return false;
    if (uci[0] < 'a' || uci[0] > 'h') return false;
    if (uci[1] < '1' || uci[1] > '8') return false;
    if (uci[2] < 'a' || uci[2] > 'h') return false;
    if (uci[3] < '1' || uci[3] > '8') return false;
    if (uci.size() == 5) {
        char p = uci[4];
        if (p != 'q' && p != 'r' && p != 'b' && p != 'n' &&
            p != 'k' && p != 'Q' && p != 'R' && p != 'B' && p != 'N' && p != 'K') {
            return false;
        }
    }
    return true;
}

bool ChessGame::is_legal(const std::string& uci) const {
    if (!is_valid_uci_format(uci)) return false;
    chess::Move m = chess::uci::uciToMove(board_, uci);
    if (m == chess::Move::NO_MOVE) return false;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    for (const auto& lm : moves) {
        if (lm == m) return true;
    }
    return false;
}

bool ChessGame::make_move_uci(const std::string& uci) {
    if (!is_valid_uci_format(uci)) return false;
    chess::Move m = chess::uci::uciToMove(board_, uci);
    if (m == chess::Move::NO_MOVE) return false;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    bool legal = false;
    for (const auto& lm : moves) {
        if (lm == m) {
            legal = true;
            break;
        }
    }
    if (!legal) return false;
    // Reserve before mutating the board (remediation F-13): if the history
    // push had to reallocate after makeMove, a bad_alloc would desync
    // board_ and move_history_. With capacity guaranteed up front and
    // chess-library makeMove being non-throwing, either both mutate or
    // neither does.
    move_history_.reserve(move_history_.size() + 1);
    // EXACT en-passant policy, see compute_perft (remediation CR-01).
    board_.makeMove<true>(m);
    move_history_.push_back(m);
    return true;
}

bool ChessGame::unmake_move() {
    if (move_history_.empty()) return false;
    chess::Move m = move_history_.back();
    move_history_.pop_back();
    board_.unmakeMove(m);
    return true;
}

uint64_t ChessGame::get_hash() const {
    return board_.hash();
}

bool ChessGame::side_to_move() const {
    return board_.sideToMove() == chess::Color::WHITE;
}

bool ChessGame::in_check() const {
    return board_.inCheck();
}

bool ChessGame::is_checkmate() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    return moves.empty() && board_.inCheck();
}

bool ChessGame::is_stalemate() const {
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    return moves.empty() && !board_.inCheck();
}

bool ChessGame::is_repetition(int count) const {
    return board_.isRepetition(count);
}

bool ChessGame::is_50_move_rule() const {
    return board_.isHalfMoveDraw();
}

bool ChessGame::is_insufficient_material() const {
    // Remediation CR-04: the vendored isInsufficientMaterial() only
    // recognizes the <=4-piece basic cases and misses dead positions with
    // more pieces, e.g. K + two same-colored bishops vs K + bishop (six
    // pieces), where no checkmate exists under any series of legal moves.
    // This mirrors python-chess's has_insufficient_material() for both
    // colors -- the project's reference oracle -- which reduces to the
    // vendored verdicts for every <=4-piece configuration and extends the
    // same-color-bishop rule to arbitrary piece counts. The classification
    // stays conservative: any pawn, rook or queen; king + two knights; a
    // knight facing non-king material; or opposite-colored bishops anywhere
    // keeps the game classified as playable.
    using chess::Bitboard;
    using chess::Color;
    using chess::PieceType;

    const Bitboard pawns   = board_.pieces(PieceType::PAWN);
    const Bitboard knights = board_.pieces(PieceType::KNIGHT);
    const Bitboard bishops = board_.pieces(PieceType::BISHOP);
    const Bitboard rooks   = board_.pieces(PieceType::ROOK);
    const Bitboard queens  = board_.pieces(PieceType::QUEEN);
    const Bitboard kings   = board_.pieces(PieceType::KING);

    // True when every bishop on the board stands on the same square color.
    const auto bishops_all_same_color = [&]() {
        if (bishops.count() <= 1) return true;
        const int ref = bishops.lsb();
        Bitboard rest = bishops;
        rest.clear(ref);
        while (rest) {
            if (!chess::Square::same_color(ref, rest.lsb())) return false;
            rest.clear(rest.lsb());
        }
        return true;
    };

    const auto has_insufficient = [&](Color color) {
        const Bitboard own = board_.us(color);
        const Bitboard opp = board_.us(~color);
        if (own & (pawns | rooks | queens)) return false;
        if (own & knights) {
            // King + exactly one knight; the opponent may hold nothing but
            // kings and queens (any pawn/knight/bishop/rook to capture would
            // allow a selfmate construction).
            return own.count() <= 2 && !(opp & ~kings & ~queens);
        }
        if (own & bishops) {
            // Any number of bishops, but all on one square color, and no
            // pawns or knights anywhere on the board.
            return bishops_all_same_color() && !pawns && !knights;
        }
        return true;
    };

    return has_insufficient(Color::WHITE) && has_insufficient(Color::BLACK);
}

bool ChessGame::is_game_over() const {
    return is_checkmate() || is_stalemate() || is_50_move_rule() || is_repetition(2) || is_insufficient_material();
}

int ChessGame::half_move_clock() const {
    return static_cast<int>(board_.halfMoveClock());
}

int ChessGame::full_move_number() const {
    return static_cast<int>(board_.fullMoveNumber());
}

uint64_t ChessGame::perft(int depth) {
    // Remediation F-02: validate depth before any recursion. Negative depths
    // were silently treated as 0 (returning 1) and excessive depths overflowed
    // the stack (perft(20000) killed the process). perft(0) == 1 remains the
    // standard convention.
    if (depth < 0) {
        throw std::invalid_argument(
            "ChessGame::perft: depth must be >= 0 (got " + std::to_string(depth) + ")");
    }
    if (depth > MAX_PERFT_DEPTH) {
        throw std::invalid_argument(
            "ChessGame::perft: depth " + std::to_string(depth) +
            " exceeds the supported maximum of " + std::to_string(MAX_PERFT_DEPTH));
    }
    return compute_perft(board_, depth);
}

// Game interface overrides
std::unique_ptr<Game> ChessGame::clone() const {
    return std::make_unique<ChessGame>(*this);
}

std::vector<int> ChessGame::legal_actions() const {
    auto uci_moves = legal_moves_uci();
    std::vector<int> actions;
    actions.reserve(uci_moves.size());
    for (const auto& uci : uci_moves) {
        int idx = move_to_index(*this, uci);
        if (idx >= 0) actions.push_back(idx);
    }
    return actions;
}

void ChessGame::apply_action(int action) {
    // Hot path for MCTS/self-play: the action index is produced by
    // index_to_move()/legal_actions(), so legitimate callers always pass
    // legal actions. The index is still validated for legality before any
    // mutation (remediation F-05): the previous implementation applied any
    // decodable action, and an in-range-but-illegal index silently corrupted
    // the game (repro: a pawn teleported a2->a5). The cost is one legal-move
    // generation per application — negligible next to a real evaluator call,
    // and callers that need the unvalidated fast path own the legality of
    // their inputs.
    std::string uci = index_to_move(*this, action);
    if (uci.empty() || action < 0) {
        // An undecodable action must never silently leave the game unchanged:
        // search would continue as if the move had been played and expand child
        // nodes from a state that diverges from the tree path. Threading the
        // error out (pybind maps std::invalid_argument to ValueError) lets the
        // RAII virtual-loss guards unwind the search cleanly.
        throw std::invalid_argument(
            "ChessGame::apply_action: cannot decode action index " + std::to_string(action));
    }
    chess::Move m = chess::uci::uciToMove(board_, uci);
    if (m == chess::Move::NO_MOVE) {
        throw std::invalid_argument(
            "ChessGame::apply_action: action index " + std::to_string(action) +
            " decoded to NO_MOVE");
    }
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board_);
    bool legal = false;
    for (const auto& lm : moves) {
        if (lm == m) {
            legal = true;
            break;
        }
    }
    if (!legal) {
        throw std::invalid_argument(
            "ChessGame::apply_action: action index " + std::to_string(action) + " (" + uci +
            ") is not legal in the current position");
    }
    // Reserve before mutating the board (see make_move_uci, remediation F-13).
    move_history_.reserve(move_history_.size() + 1);
    // EXACT en-passant policy, see compute_perft (remediation CR-01).
    board_.makeMove<true>(m);
    move_history_.push_back(m);
}

bool ChessGame::is_terminal() const {
    return is_game_over();
}

float ChessGame::terminal_value() const {
    if (is_checkmate()) return -1.0f;
    return 0.0f;
}

void ChessGame::encode_state(float* out) const {
    zeropawn::encode_state(*this, out);
}

std::string ChessGame::action_to_string(int action) const {
    return index_to_move(*this, action);
}

int ChessGame::string_to_action(const std::string& str) const {
    return move_to_index(*this, str);
}

} // namespace zeropawn
