#include "zeropawn/chess_game.hpp"
#include "zeropawn/encoder.hpp"

namespace zeropawn {

uint64_t compute_perft(chess::Board& board, int depth) {
    if (depth <= 0) return 1ULL;
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const auto& move : moves) {
        board.makeMove(move);
        nodes += compute_perft(board, depth - 1);
        board.unmakeMove(move);
    }
    return nodes;
}

ChessGame::ChessGame() : board_(chess::constants::STARTPOS) {}

ChessGame::ChessGame(const std::string& fen) : board_(chess::constants::STARTPOS) {
    set_fen(fen);
}

bool ChessGame::set_fen(const std::string& fen) {
    chess::Board temp;
    if (!temp.setFen(fen)) return false;
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
    board_.makeMove(m);
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
    return board_.isInsufficientMaterial();
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
    std::string uci = index_to_move(*this, action);
    make_move_uci(uci);
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
