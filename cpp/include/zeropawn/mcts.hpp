#pragma once

#include <vector>
#include <string>
#include <list>
#include <memory>
#include <unordered_map>
#include <functional>
#include <random>
#include <utility>
#include <atomic>
#include <cstdint>
#include "zeropawn/game.hpp"

namespace zeropawn {

struct MCTSConfig {
    float c_puct = 2.0f;
    float dirichlet_alpha = 0.3f;
    float dirichlet_weight = 0.25f;
    float fpu_reduction = 0.2f;
    int virtual_loss = 1;
    bool add_noise = false;
    uint64_t seed = 42;
    bool reuse_tree = false;
    size_t max_nodes = 25000;
    int batch_size = 1;
};

struct SearchResult {
    int best_action = -1;
    std::vector<float> action_visits; // full policy distribution (size = game.num_actions())
    float root_value = 0.0f;
    int retained_visits = 0;
    int new_simulations = 0;
    int batch_size = 1;
};

// Evaluator function: returns (policy_distribution, scalar_value in [-1, 1])
using Evaluator = std::function<std::pair<std::vector<float>, float>(const Game&)>;

// Batch evaluator function: returns vector of (policy_distribution, scalar_value) for a batch of games
using BatchEvaluator = std::function<std::vector<std::pair<std::vector<float>, float>>(const std::vector<const Game*>&)>;

// Legacy row-major batch evaluator for search_batched backward compatibility:
using RawBatchEvaluator =
    std::function<std::pair<std::vector<float>, std::vector<float>>(const std::vector<float>&, int)>;

class MCTS {
public:
    // Upper bound on cached NN evaluations. When the cache reaches this size, the
    // least-recently-used entry is evicted (true LRU, see cache_lookup/cache_insert).
    // Keeps memory bounded during long self-play sessions without discarding entries
    // that are still being hit every simulation.
    static constexpr size_t NN_CACHE_CAP = 100000;

    explicit MCTS(const MCTSConfig& config = MCTSConfig{});
    ~MCTS();

    // Clear the NN evaluation cache and reset retained tree.
    void clear();
    void set_seed(uint64_t seed);
    SearchResult search(
        const Game& root_state,
        int num_simulations,
        const Evaluator& evaluator,
        const BatchEvaluator& batch_evaluator = nullptr
    );
    SearchResult search_batched(
        const Game& root_state,
        int num_simulations,
        const RawBatchEvaluator& batch_eval,
        int batch_size
    );
    size_t cache_size() const { return nn_cache_.size(); }

    // Subtree reuse API
    bool advance_root(int my_action, int opp_action, const Game& new_state);
    bool advance_root(int action, const Game& new_state);
    void reset_root();
    size_t live_node_count() const;
    bool has_retained_root() const { return root_ != nullptr; }
    int retained_visits() const { return last_retained_visits_; }

    // Node allocation instrumentation (memory safety diagnostics)
    static int64_t total_allocated_nodes();
    static int64_t total_freed_nodes();
    static void reset_node_alloc_stats();
    int total_active_virtual_losses() const;

private:
    struct Node {
        uint64_t state_hash = 0;
        std::vector<int> legal_actions;
        std::vector<float> priors;
        std::vector<int> visit_counts;
        std::vector<float> total_values;
        std::vector<int> virtual_losses;
        std::vector<std::unique_ptr<Node>> children;
        bool is_terminal = false;
        float terminal_val = 0.0f;
        float eval_val = 0.0f;

        static inline std::atomic<int64_t> alloc_count{0};
        static inline std::atomic<int64_t> free_count{0};

        Node();
        ~Node();
    };

    struct CachedEval {
        std::vector<float> policy;
        float value;
        // Position of this key in nn_cache_lru_ (front = most recently used).
        std::list<uint64_t>::iterator lru_it;
    };

    MCTSConfig config_;
    std::mt19937_64 rng_;
    std::unordered_map<uint64_t, CachedEval> nn_cache_;
    // Recency order for nn_cache_: front is the most recently used key.
    std::list<uint64_t> nn_cache_lru_;

    std::unique_ptr<Node> root_;
    int last_retained_visits_ = 0;

    static void safe_delete_tree(std::unique_ptr<Node>& root);
    static size_t count_nodes(const Node* node);

    std::unique_ptr<Node> create_node(const Game& state, bool is_root, const Evaluator& evaluator);
    std::unique_ptr<Node> create_node_from_eval(const Game& state, bool is_root, const std::vector<float>& full_policy, float eval_val);
    std::unique_ptr<Node> create_unscored_node(const Game& state);
    void set_priors(Node& node, const float* policy, int policy_size, int num_actions, bool is_root);
    void backup_path(const std::vector<std::pair<Node*, int>>& path, float leaf_val);
    void cache_insert(uint64_t h, std::vector<float> policy, float value);
    // Look up h, marking it most-recently-used. Returns nullptr on a miss.
    const CachedEval* cache_lookup(uint64_t h);
    void cache_evict_lru();
    SearchResult aggregate(const Node& root, int num_legal, int num_actions) const;
    void sample_dirichlet(int k, float alpha, std::vector<float>& out_noise);
    int select_action(const Node& node, float parent_q) const;
};

} // namespace zeropawn
