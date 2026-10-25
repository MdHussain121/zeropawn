#include "zeropawn/mcts.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>

namespace zeropawn {

MCTS::Node::Node() {
    alloc_count.fetch_add(1, std::memory_order_relaxed);
}

MCTS::Node::~Node() {
    free_count.fetch_add(1, std::memory_order_relaxed);
}

int64_t MCTS::total_allocated_nodes() {
    return Node::alloc_count.load(std::memory_order_relaxed);
}

int64_t MCTS::total_freed_nodes() {
    return Node::free_count.load(std::memory_order_relaxed);
}

void MCTS::reset_node_alloc_stats() {
    Node::alloc_count.store(0, std::memory_order_relaxed);
    Node::free_count.store(0, std::memory_order_relaxed);
}

MCTS::MCTS(const MCTSConfig& config)
    : config_(config), rng_(config.seed) {
    nn_cache_.reserve(10000);
}

MCTS::~MCTS() {
    reset_root();
}

void MCTS::safe_delete_tree(std::unique_ptr<Node>& root) {
    if (!root) return;
    std::vector<std::unique_ptr<Node>> stack;
    stack.push_back(std::move(root));
    while (!stack.empty()) {
        auto curr = std::move(stack.back());
        stack.pop_back();
        for (auto& child : curr->children) {
            if (child) {
                stack.push_back(std::move(child));
            }
        }
    }
}

size_t MCTS::count_nodes(const Node* node) {
    if (!node) return 0;
    size_t count = 0;
    std::vector<const Node*> stack;
    stack.push_back(node);
    while (!stack.empty()) {
        const Node* curr = stack.back();
        stack.pop_back();
        count++;
        for (const auto& child : curr->children) {
            if (child) {
                stack.push_back(child.get());
            }
        }
    }
    return count;
}

size_t MCTS::live_node_count() const {
    return count_nodes(root_.get());
}

int MCTS::total_active_virtual_losses() const {
    if (!root_) return 0;
    int total = 0;
    std::vector<const Node*> stack;
    stack.push_back(root_.get());
    while (!stack.empty()) {
        const Node* curr = stack.back();
        stack.pop_back();
        for (int vl : curr->virtual_losses) {
            total += vl;
        }
        for (const auto& child : curr->children) {
            if (child) {
                stack.push_back(child.get());
            }
        }
    }
    return total;
}

void MCTS::reset_root() {
    safe_delete_tree(root_);
    last_retained_visits_ = 0;
}

void MCTS::clear() {
    nn_cache_.clear();
    reset_root();
}

void MCTS::set_seed(uint64_t seed) {
    config_.seed = seed;
    rng_.seed(seed);
}

void MCTS::sample_dirichlet(int k, float alpha, std::vector<float>& out_noise) {
    out_noise.resize(k);
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        out_noise[i] = gamma(rng_);
        sum += out_noise[i];
    }
    if (sum > 1e-8f) {
        for (int i = 0; i < k; ++i) {
            out_noise[i] /= sum;
        }
    } else {
        std::fill(out_noise.begin(), out_noise.end(), 1.0f / k);
    }
}

std::unique_ptr<MCTS::Node> MCTS::create_node_from_eval(
    const Game& state, bool is_root, const std::vector<float>& full_policy, float eval_val) {
    auto node = std::make_unique<Node>();
    node->state_hash = state.get_hash();
    node->legal_actions = state.legal_actions();
    int k = static_cast<int>(node->legal_actions.size());

    node->priors.resize(k, 0.0f);
    node->visit_counts.resize(k, 0);
    node->total_values.resize(k, 0.0f);
    node->virtual_losses.resize(k, 0);
    node->children.resize(k);

    if (state.is_terminal()) {
        node->is_terminal = true;
        node->terminal_val = state.terminal_value();
        return node;
    }

    if (k == 0) {
        node->is_terminal = true;
        node->terminal_val = 0.0f;
        return node;
    }

    // Mask and normalize priors for legal actions
    float p_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        int a = node->legal_actions[i];
        float p = (a >= 0 && a < static_cast<int>(full_policy.size())) ? full_policy[a] : 0.0f;
        node->priors[i] = p;
        p_sum += p;
    }

    if (p_sum > 1e-8f) {
        for (int i = 0; i < k; ++i) {
            node->priors[i] /= p_sum;
        }
    } else {
        for (int i = 0; i < k; ++i) {
            node->priors[i] = 1.0f / k;
        }
    }

    if (is_root && config_.add_noise && k > 0) {
        std::vector<float> noise;
        sample_dirichlet(k, config_.dirichlet_alpha, noise);
        float eps = config_.dirichlet_weight;
        for (int i = 0; i < k; ++i) {
            node->priors[i] = (1.0f - eps) * node->priors[i] + eps * noise[i];
        }
    }

    node->eval_val = eval_val;
    return node;
}

std::unique_ptr<MCTS::Node> MCTS::create_node(const Game& state, bool is_root, const Evaluator& evaluator) {
    if (state.is_terminal()) {
        return create_node_from_eval(state, is_root, {}, state.terminal_value());
    }
    if (state.legal_actions().empty()) {
        return create_node_from_eval(state, is_root, {}, 0.0f);
    }

    uint64_t h = state.get_hash();
    auto it = nn_cache_.find(h);
    if (it != nn_cache_.end()) {
        return create_node_from_eval(state, is_root, it->second.policy, it->second.value);
    }

    auto eval_res = evaluator(state);
    float eval_val = eval_res.second;
    // Simple size guard: evict oldest-ish entry if over cap.
    if (nn_cache_.size() >= MCTS::NN_CACHE_CAP) {
        nn_cache_.erase(nn_cache_.begin());
    }
    auto inserted = nn_cache_.emplace(h, CachedEval{std::move(eval_res.first), eval_val});
    return create_node_from_eval(state, is_root, inserted.first->second.policy, eval_val);
}

int MCTS::select_action(const Node& node, float parent_q) const {
    int k = static_cast<int>(node.legal_actions.size());
    if (k == 0) return -1;

    int total_visits = 0;
    for (int i = 0; i < k; ++i) {
        total_visits += node.visit_counts[i];
    }
    float sqrt_total = std::sqrt(static_cast<float>(total_visits));

    int best_idx = 0;
    float best_score = -1e9f;

    for (int i = 0; i < k; ++i) {
        int n = node.visit_counts[i] + node.virtual_losses[i];
        float q;
        if (n > 0) {
            q = (node.total_values[i] - static_cast<float>(node.virtual_losses[i])) / static_cast<float>(n);
        } else {
            q = parent_q - config_.fpu_reduction;
        }

        float u = config_.c_puct * node.priors[i] * sqrt_total / (1.0f + static_cast<float>(n));
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}

bool MCTS::advance_root(int my_action, int opp_action, const Game& new_state) {
    if (!config_.reuse_tree || !root_) {
        reset_root();
        return false;
    }

    // 1. Locate my_action in root
    int my_idx = -1;
    for (size_t i = 0; i < root_->legal_actions.size(); ++i) {
        if (root_->legal_actions[i] == my_action) {
            my_idx = static_cast<int>(i);
            break;
        }
    }
    if (my_idx < 0 || !root_->children[my_idx]) {
        reset_root();
        return false;
    }
    Node* child = root_->children[my_idx].get();

    // 2. Locate opp_action in child
    int opp_idx = -1;
    for (size_t i = 0; i < child->legal_actions.size(); ++i) {
        if (child->legal_actions[i] == opp_action) {
            opp_idx = static_cast<int>(i);
            break;
        }
    }
    if (opp_idx < 0 || !child->children[opp_idx]) {
        reset_root();
        return false;
    }
    Node* grandchild = child->children[opp_idx].get();

    // 3. Verify state integrity
    if (grandchild->state_hash != new_state.get_hash() ||
        grandchild->is_terminal != new_state.is_terminal() ||
        grandchild->legal_actions != new_state.legal_actions()) {
        reset_root();
        return false;
    }

    // 4. Verify node cap
    size_t subtree_nodes = count_nodes(grandchild);
    if (subtree_nodes > config_.max_nodes) {
        reset_root();
        return false;
    }

    // 5. Safe re-rooting: extract grandchild and safely release unreachable tree
    std::unique_ptr<Node> new_root = std::move(child->children[opp_idx]);
    std::fill(new_root->virtual_losses.begin(), new_root->virtual_losses.end(), 0);
    safe_delete_tree(root_);
    root_ = std::move(new_root);
    return true;
}

bool MCTS::advance_root(int action, const Game& new_state) {
    if (!config_.reuse_tree || !root_) {
        reset_root();
        return false;
    }

    int idx = -1;
    for (size_t i = 0; i < root_->legal_actions.size(); ++i) {
        if (root_->legal_actions[i] == action) {
            idx = static_cast<int>(i);
            break;
        }
    }
    if (idx < 0 || !root_->children[idx]) {
        reset_root();
        return false;
    }
    Node* child = root_->children[idx].get();

    if (child->state_hash != new_state.get_hash() ||
        child->is_terminal != new_state.is_terminal() ||
        child->legal_actions != new_state.legal_actions()) {
        reset_root();
        return false;
    }

    size_t subtree_nodes = count_nodes(child);
    if (subtree_nodes > config_.max_nodes) {
        reset_root();
        return false;
    }

    std::unique_ptr<Node> new_root = std::move(root_->children[idx]);
    std::fill(new_root->virtual_losses.begin(), new_root->virtual_losses.end(), 0);
    safe_delete_tree(root_);
    root_ = std::move(new_root);
    return true;
}

SearchResult MCTS::search(const Game& root_state, int num_simulations, const Evaluator& evaluator, const BatchEvaluator& batch_evaluator) {
    SearchResult result;
    result.action_visits.resize(root_state.num_actions(), 0.0f);
    result.batch_size = config_.batch_size;

    int retained_visits = 0;
    if (config_.reuse_tree && root_ != nullptr) {
        // Validate that retained root_ matches root_state
        if (root_->state_hash != root_state.get_hash() ||
            root_->is_terminal != root_state.is_terminal() ||
            root_->legal_actions != root_state.legal_actions() ||
            live_node_count() > config_.max_nodes) {
            reset_root();
        } else {
            for (int n : root_->visit_counts) {
                retained_visits += n;
            }
        }
    } else if (!config_.reuse_tree) {
        reset_root();
    }

    if (!root_) {
        root_ = create_node(root_state, true, evaluator);
    }

    int num_legal = static_cast<int>(root_->legal_actions.size());
    if (num_legal == 0 || root_->is_terminal) {
        result.root_value = root_->terminal_val;
        result.retained_visits = retained_visits;
        result.new_simulations = 0;
        last_retained_visits_ = retained_visits;
        if (!config_.reuse_tree) {
            reset_root();
        }
        return result;
    }

    if (config_.batch_size <= 1) {
        // Serial simulation path (exact baseline implementation)
        struct SimulationVirtualLossGuard {
            std::vector<std::pair<Node*, int>>& search_path;
            int virtual_loss;
            bool committed = false;

            SimulationVirtualLossGuard(std::vector<std::pair<Node*, int>>& path, int vl)
                : search_path(path), virtual_loss(vl) {}

            ~SimulationVirtualLossGuard() {
                if (!committed) {
                    for (auto& step : search_path) {
                        step.first->virtual_losses[step.second] -= virtual_loss;
                    }
                }
            }
        };

        try {
            for (int sim = 0; sim < num_simulations; ++sim) {
                Node* curr_node = root_.get();
                auto sim_state = root_state.clone();
                std::vector<std::pair<Node*, int>> search_path;
                SimulationVirtualLossGuard vl_guard(search_path, config_.virtual_loss);
                float parent_q = 0.0f;

                // Selection
                while (!curr_node->is_terminal && curr_node->legal_actions.size() > 0) {
                    int action_idx = select_action(*curr_node, parent_q);
                    if (action_idx < 0) break;

                    search_path.push_back({curr_node, action_idx});
                    curr_node->virtual_losses[action_idx] += config_.virtual_loss;

                    // Parent Q for next level (inverted for opponent perspective)
                    int n = curr_node->visit_counts[action_idx];
                    if (n > 0) {
                        parent_q = -(curr_node->total_values[action_idx] / static_cast<float>(n));
                    }

                    int action = curr_node->legal_actions[action_idx];
                    sim_state->apply_action(action);

                    if (!curr_node->children[action_idx]) {
                        // Expansion: create child node and stop selection
                        curr_node->children[action_idx] = create_node(*sim_state, false, evaluator);
                        curr_node = curr_node->children[action_idx].get();
                        break;
                    } else {
                        curr_node = curr_node->children[action_idx].get();
                    }
                }

                // Evaluation value at leaf
                float leaf_val = curr_node->is_terminal ? curr_node->terminal_val : curr_node->eval_val;

                // Backpropagation: value flips perspective at each step
                float v = leaf_val;
                for (int i = static_cast<int>(search_path.size()) - 1; i >= 0; --i) {
                    Node* node = search_path[i].first;
                    int act_idx = search_path[i].second;

                    node->virtual_losses[act_idx] -= config_.virtual_loss;
                    v = -v; // In zero-sum game, value for player making action is opponent's negative
                    node->total_values[act_idx] += v;
                    node->visit_counts[act_idx] += 1;
                }
                vl_guard.committed = true;
            }
        } catch (...) {
            if (!config_.reuse_tree) {
                reset_root();
            }
            throw;
        }
    } else {
        // Batched simulation path
        BatchEvaluator effective_batch_eval = batch_evaluator;
        if (!effective_batch_eval) {
            effective_batch_eval = [&evaluator](const std::vector<const Game*>& states) {
                std::vector<std::pair<std::vector<float>, float>> out;
                out.reserve(states.size());
                for (const auto* s : states) {
                    out.push_back(evaluator(*s));
                }
                return out;
            };
        }

        struct Candidate {
            std::vector<std::pair<Node*, int>> search_path;
            std::unique_ptr<Game> sim_state;
            Node* leaf_node = nullptr;
            Node* parent_node = nullptr;
            int action_idx = -1;
            bool needs_eval = false;
            float leaf_val = 0.0f;
        };

        struct BatchVirtualLossGuard {
            std::vector<Candidate>& candidates;
            int virtual_loss;
            bool committed = false;

            BatchVirtualLossGuard(std::vector<Candidate>& cands, int vl)
                : candidates(cands), virtual_loss(vl) {}

            ~BatchVirtualLossGuard() {
                if (!committed) {
                    for (auto& cand : candidates) {
                        for (auto& step : cand.search_path) {
                            step.first->virtual_losses[step.second] -= virtual_loss;
                        }
                    }
                }
            }
        };

        int sims_done = 0;
        try {
            while (sims_done < num_simulations) {
                int target_batch = std::min(config_.batch_size, num_simulations - sims_done);
                std::vector<Candidate> candidates;
                candidates.reserve(target_batch);
                BatchVirtualLossGuard vl_guard(candidates, config_.virtual_loss);

                std::vector<std::pair<Node*, int>> pending_expansions;

                for (int b = 0; b < target_batch; ++b) {
                    Candidate cand;
                    cand.sim_state = root_state.clone();
                    Node* curr_node = root_.get();
                    float parent_q = 0.0f;
                    bool collided = false;

                    while (!curr_node->is_terminal && !curr_node->legal_actions.empty()) {
                        int action_idx = select_action(*curr_node, parent_q);
                        if (action_idx < 0) break;

                        cand.search_path.push_back({curr_node, action_idx});
                        curr_node->virtual_losses[action_idx] += config_.virtual_loss;

                        int n = curr_node->visit_counts[action_idx];
                        if (n > 0) {
                            parent_q = -(curr_node->total_values[action_idx] / static_cast<float>(n));
                        }

                        int action = curr_node->legal_actions[action_idx];
                        cand.sim_state->apply_action(action);

                        if (!curr_node->children[action_idx]) {
                            // Check collision with already pending expansion in this batch
                            for (const auto& p : pending_expansions) {
                                if (p.first == curr_node && p.second == action_idx) {
                                    collided = true;
                                    break;
                                }
                            }
                            if (collided) {
                                for (auto& step : cand.search_path) {
                                    step.first->virtual_losses[step.second] -= config_.virtual_loss;
                                }
                                cand.search_path.clear();
                                break;
                            }

                            pending_expansions.push_back({curr_node, action_idx});
                            cand.parent_node = curr_node;
                            cand.action_idx = action_idx;

                            if (cand.sim_state->is_terminal()) {
                                cand.needs_eval = false;
                                cand.leaf_val = cand.sim_state->terminal_value();
                            } else if (cand.sim_state->legal_actions().empty()) {
                                cand.needs_eval = false;
                                cand.leaf_val = 0.0f;
                            } else {
                                uint64_t h = cand.sim_state->get_hash();
                                auto it = nn_cache_.find(h);
                                if (it != nn_cache_.end()) {
                                    cand.needs_eval = false;
                                    cand.leaf_val = it->second.value;
                                } else {
                                    cand.needs_eval = true;
                                }
                            }
                            break;
                        } else {
                            curr_node = curr_node->children[action_idx].get();
                        }
                    }

                    if (collided) {
                        break;
                    }

                    if (cand.parent_node == nullptr) {
                        cand.leaf_node = curr_node;
                        cand.needs_eval = false;
                        cand.leaf_val = curr_node->terminal_val;
                    }

                    candidates.push_back(std::move(cand));
                }

                if (candidates.empty()) {
                    break;
                }

                std::vector<const Game*> eval_states;
                std::vector<size_t> eval_indices;
                for (size_t i = 0; i < candidates.size(); ++i) {
                    if (candidates[i].needs_eval) {
                        eval_states.push_back(candidates[i].sim_state.get());
                        eval_indices.push_back(i);
                    }
                }

                if (!eval_states.empty()) {
                    auto batch_results = effective_batch_eval(eval_states);
                    if (batch_results.size() != eval_states.size()) {
                        throw std::runtime_error("batch_evaluator returned vector of wrong size");
                    }
                    for (size_t k = 0; k < eval_states.size(); ++k) {
                        size_t cand_idx = eval_indices[k];
                        Candidate& cand = candidates[cand_idx];
                        auto& eval_res = batch_results[k];
                        cand.leaf_val = eval_res.second;

                        uint64_t h = cand.sim_state->get_hash();
                        if (nn_cache_.size() >= MCTS::NN_CACHE_CAP) {
                            nn_cache_.erase(nn_cache_.begin());
                        }
                        auto inserted = nn_cache_.emplace(h, CachedEval{std::move(eval_res.first), cand.leaf_val});
                        cand.parent_node->children[cand.action_idx] = create_node_from_eval(
                            *cand.sim_state, false, inserted.first->second.policy, cand.leaf_val
                        );
                    }
                }

                for (auto& cand : candidates) {
                    if (!cand.needs_eval && cand.parent_node != nullptr) {
                        if (cand.sim_state->is_terminal() || cand.sim_state->legal_actions().empty()) {
                            cand.parent_node->children[cand.action_idx] = create_node_from_eval(
                                *cand.sim_state, false, {}, cand.leaf_val
                            );
                        } else {
                            uint64_t h = cand.sim_state->get_hash();
                            auto it = nn_cache_.find(h);
                            cand.parent_node->children[cand.action_idx] = create_node_from_eval(
                                *cand.sim_state, false, it->second.policy, it->second.value
                            );
                        }
                    }
                }

                for (auto& cand : candidates) {
                    float v = cand.leaf_val;
                    for (int i = static_cast<int>(cand.search_path.size()) - 1; i >= 0; --i) {
                        Node* node = cand.search_path[i].first;
                        int act_idx = cand.search_path[i].second;

                        node->virtual_losses[act_idx] -= config_.virtual_loss;
                        v = -v;
                        node->total_values[act_idx] += v;
                        node->visit_counts[act_idx] += 1;
                    }
                }

                vl_guard.committed = true;
                sims_done += static_cast<int>(candidates.size());
            }
        } catch (...) {
            if (!config_.reuse_tree) {
                reset_root();
            }
            throw;
        }
    }

    // Aggregate statistics at root
    int total_visits = 0;
    int max_visits = -1;
    int best_act = -1;
    float sum_q = 0.0f;

    for (int i = 0; i < num_legal; ++i) {
        int act = root_->legal_actions[i];
        int n = root_->visit_counts[i];
        total_visits += n;
        if (act >= 0 && act < static_cast<int>(result.action_visits.size())) {
            result.action_visits[act] = static_cast<float>(n);
        }
        if (n > max_visits) {
            max_visits = n;
            best_act = act;
        }
        if (n > 0) {
            sum_q += root_->total_values[i];
        }
    }

    result.best_action = best_act;
    result.root_value = (total_visits > 0) ? (sum_q / total_visits) : 0.0f;
    result.retained_visits = retained_visits;
    result.new_simulations = num_simulations;
    last_retained_visits_ = retained_visits;

    // Normalize action visits to distribution
    if (total_visits > 0) {
        for (auto& v : result.action_visits) {
            v /= static_cast<float>(total_visits);
        }
    }

    // Post-search lifecycle checks
    if (!config_.reuse_tree) {
        reset_root();
    } else {
        // Enforce hard cap: if tree grew beyond max_nodes, reset root
        if (live_node_count() > config_.max_nodes) {
            reset_root();
        }
    }

    return result;
}


std::unique_ptr<MCTS::Node> MCTS::create_unscored_node(const Game& state) {
    auto node = std::make_unique<Node>();
    node->legal_actions = state.legal_actions();
    int k = static_cast<int>(node->legal_actions.size());

    node->priors.assign(k, k > 0 ? 1.0f / k : 0.0f);
    node->visit_counts.assign(k, 0);
    node->total_values.assign(k, 0.0f);
    node->virtual_losses.assign(k, 0);
    node->children.resize(k);

    if (state.is_terminal()) {
        node->is_terminal = true;
        node->terminal_val = state.terminal_value();
        return node;
    }
    if (k == 0) {
        node->is_terminal = true;
        node->terminal_val = 0.0f;
        return node;
    }
    return node;
}

void MCTS::set_priors(Node& node, const float* policy, int policy_size, int num_actions, bool is_root) {
    int k = static_cast<int>(node.legal_actions.size());
    if (k == 0) return;

    float p_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        int a = node.legal_actions[i];
        float p = (a >= 0 && a < policy_size) ? policy[a] : 0.0f;
        node.priors[i] = p;
        p_sum += p;
    }

    if (p_sum > 1e-8f) {
        for (int i = 0; i < k; ++i) {
            node.priors[i] /= p_sum;
        }
    } else {
        for (int i = 0; i < k; ++i) {
            node.priors[i] = 1.0f / k;
        }
    }

    if (is_root && config_.add_noise && k > 0) {
        std::vector<float> noise;
        sample_dirichlet(k, config_.dirichlet_alpha, noise);
        float eps = config_.dirichlet_weight;
        for (int i = 0; i < k; ++i) {
            node.priors[i] = (1.0f - eps) * node.priors[i] + eps * noise[i];
        }
    }
}

void MCTS::backup_path(const std::vector<std::pair<Node*, int>>& path, float leaf_val) {
    float v = leaf_val;
    for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
        Node* node = path[i].first;
        int act_idx = path[i].second;

        node->virtual_losses[act_idx] -= config_.virtual_loss;
        v = -v; // Zero-sum value perspective alternation
        node->total_values[act_idx] += v;
        node->visit_counts[act_idx] += 1;
    }
}

void MCTS::cache_insert(uint64_t h, std::vector<float> policy, float value) {
    if (nn_cache_.size() >= MCTS::NN_CACHE_CAP) {
        nn_cache_.erase(nn_cache_.begin());
    }
    nn_cache_.emplace(h, CachedEval{std::move(policy), value});
}

SearchResult MCTS::aggregate(const Node& root, int num_legal, int num_actions) const {
    SearchResult result;
    result.action_visits.assign(num_actions, 0.0f);
    int total_visits = 0;
    int max_visits = -1;
    int best_act = -1;
    float sum_q = 0.0f;

    for (int i = 0; i < num_legal; ++i) {
        int act = root.legal_actions[i];
        int n = root.visit_counts[i];
        total_visits += n;
        if (act >= 0 && act < num_actions) {
            result.action_visits[act] = static_cast<float>(n);
        }
        if (n > max_visits) {
            max_visits = n;
            best_act = act;
        }
        if (n > 0) {
            sum_q += root.total_values[i];
        }
    }

    result.best_action = best_act;
    result.root_value = (total_visits > 0) ? (sum_q / static_cast<float>(total_visits)) : 0.0f;

    if (total_visits > 0) {
        for (auto& v : result.action_visits) {
            v /= static_cast<float>(total_visits);
        }
    }
    return result;
}

SearchResult MCTS::search_batched(const Game& root_state, int num_simulations,
                                  const RawBatchEvaluator& batch_eval, int batch_size) {
    const int B = batch_size > 0 ? batch_size : 1;
    const int state_size = root_state.state_size();
    const int num_actions = root_state.num_actions();

    // Root evaluation (single-element batch if not in cache)
    uint64_t root_h = root_state.get_hash();
    auto root_it = nn_cache_.find(root_h);
    auto root = create_unscored_node(root_state);
    int num_legal = static_cast<int>(root->legal_actions.size());
    if (num_legal == 0 || root->is_terminal) {
        SearchResult early;
        early.action_visits.assign(num_actions, 0.0f);
        early.root_value = root->terminal_val;
        return early;
    }

    if (root_it != nn_cache_.end()) {
        set_priors(*root, root_it->second.policy.data(),
                   static_cast<int>(root_it->second.policy.size()), num_actions, true);
    } else {
        std::vector<float> root_planes(static_cast<size_t>(state_size));
        root_state.encode_state(root_planes.data());
        auto root_res = batch_eval(root_planes, 1);
        if (!root_res.first.empty()) {
            int have = static_cast<int>(root_res.first.size());
            set_priors(*root, root_res.first.data(), have < num_actions ? have : num_actions,
                       num_actions, true);
        }
        cache_insert(root_h, std::move(root_res.first),
                     root_res.second.empty() ? 0.0f : root_res.second[0]);
    }

    struct PendingLeaf {
        Node* node = nullptr;
        std::unique_ptr<Game> state;
        uint64_t hash = 0;
        std::vector<std::vector<std::pair<Node*, int>>> paths;
    };

    int sims_done = 0;
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(B) * static_cast<size_t>(state_size));

    while (sims_done < num_simulations) {
        std::vector<PendingLeaf> batch;
        batch.reserve(static_cast<size_t>(B));
        std::unordered_map<Node*, size_t> pending_index;
        std::unordered_map<uint64_t, size_t> hash_index;

        // Gather up to B leaves (selection with virtual loss, optimistic expansion)
        while (sims_done < num_simulations && static_cast<int>(batch.size()) < B) {
            Node* curr_node = root.get();
            auto sim_state = root_state.clone();
            std::vector<std::pair<Node*, int>> search_path;
            float parent_q = 0.0f;
            bool consumed = false;

            while (!curr_node->is_terminal && !curr_node->legal_actions.empty()) {
                auto pit = pending_index.find(curr_node);
                if (pit != pending_index.end()) {
                    // Selected an unscored node already pending in this batch.
                    // Frontier is saturated for this batch: revert virtual loss on aborted rollout
                    // and break out to flush the accumulated batch to the GPU.
                    for (auto& step : search_path) {
                        step.first->virtual_losses[step.second] -= config_.virtual_loss;
                    }
                    consumed = true;
                    break;
                }

                int action_idx = select_action(*curr_node, parent_q);
                if (action_idx < 0) {
                    for (auto& step : search_path) {
                        step.first->virtual_losses[step.second] -= config_.virtual_loss;
                    }
                    consumed = true;
                    break;
                }

                search_path.push_back({curr_node, action_idx});
                curr_node->virtual_losses[action_idx] += config_.virtual_loss;

                // Parent Q for next level: correctly negated for opponent perspective (DEF-01)
                int n = curr_node->visit_counts[action_idx];
                if (n > 0) {
                    parent_q = -(curr_node->total_values[action_idx] / static_cast<float>(n));
                }

                int action = curr_node->legal_actions[action_idx];
                sim_state->apply_action(action);

                if (!curr_node->children[action_idx]) {
                    curr_node->children[action_idx] = create_unscored_node(*sim_state);
                    curr_node = curr_node->children[action_idx].get();
                    break;
                }
                curr_node = curr_node->children[action_idx].get();
            }
            if (consumed) {
                // Saturated frontier: flush currently accumulated batch
                break;
            }

            if (curr_node->is_terminal) {
                backup_path(search_path, curr_node->terminal_val);
                sims_done++;
                continue;
            }

            uint64_t h = sim_state->get_hash();
            auto it = nn_cache_.find(h);
            if (it != nn_cache_.end()) {
                set_priors(*curr_node, it->second.policy.data(),
                           static_cast<int>(it->second.policy.size()), num_actions, false);
                backup_path(search_path, it->second.value);
                sims_done++;
                continue;
            }

            // Deduplication by hash across distinct nodes in the same batch
            auto hit = hash_index.find(h);
            if (hit != hash_index.end()) {
                batch[hit->second].paths.push_back(std::move(search_path));
                sims_done++;
                continue;
            }

            PendingLeaf leaf;
            leaf.node = curr_node;
            leaf.state = std::move(sim_state);
            leaf.hash = h;
            leaf.paths.push_back(std::move(search_path));

            size_t idx = batch.size();
            pending_index[curr_node] = idx;
            hash_index[h] = idx;
            batch.push_back(std::move(leaf));
            sims_done++;
        }

        if (batch.empty()) break;

        // Dispatches single batch evaluation
        flat.clear();
        for (auto& leaf : batch) {
            size_t base = flat.size();
            flat.resize(base + static_cast<size_t>(state_size));
            leaf.state->encode_state(flat.data() + base);
        }

        auto res = batch_eval(flat, static_cast<int>(batch.size()));
        const std::vector<float>& all_policy = res.first;
        const std::vector<float>& all_values = res.second;

        for (size_t i = 0; i < batch.size(); ++i) {
            PendingLeaf& leaf = batch[i];
            float v = (i < all_values.size()) ? all_values[i] : 0.0f;
            std::vector<float> policy_row;
            size_t off = i * static_cast<size_t>(num_actions);
            if (off < all_policy.size()) {
                size_t avail = std::min(static_cast<size_t>(num_actions), all_policy.size() - off);
                policy_row.assign(all_policy.begin() + off, all_policy.begin() + off + avail);
            }
            set_priors(*leaf.node, policy_row.data(), static_cast<int>(policy_row.size()),
                       num_actions, false);
            cache_insert(leaf.hash, std::move(policy_row), v);
            for (auto& path : leaf.paths) {
                backup_path(path, v);
            }
        }
    }

    return aggregate(*root, num_legal, num_actions);
}


} // namespace zeropawn
