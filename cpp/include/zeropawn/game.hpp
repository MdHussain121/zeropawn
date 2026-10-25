#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace zeropawn {

class Game {
public:
    virtual ~Game() = default;

    virtual std::unique_ptr<Game> clone() const = 0;
    virtual std::vector<int> legal_actions() const = 0;
    virtual void apply_action(int action) = 0;
    virtual bool is_terminal() const = 0;
    virtual float terminal_value() const = 0; // Perspective of side to move
    virtual uint64_t get_hash() const = 0;
    virtual int num_planes() const = 0;
    virtual int state_size() const = 0;
    virtual int num_actions() const = 0;
    virtual void encode_state(float* out) const = 0;
    virtual std::string action_to_string(int action) const = 0;
    virtual int string_to_action(const std::string& str) const = 0;
};

} // namespace zeropawn
