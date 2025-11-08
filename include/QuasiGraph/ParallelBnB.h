#pragma once

#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <queue>
#include <condition_variable>
#include <functional>
#include <memory>

namespace QuasiGraph {

/**
 * Work-stealing task scheduler for parallel branch-and-bound
 * Lock-free design for minimal overhead
 */
template<typename Task>
class WorkStealingScheduler {
public:
    using WorkerFunction = std::function<std::vector<Task>(Task)>;
    
    explicit WorkStealingScheduler(size_t num_threads = std::thread::hardware_concurrency())
        : num_threads_(num_threads), done_(false), active_tasks_(0), worker_func_(nullptr) {
        
        // Initialize per-thread queues
        task_queues_.resize(num_threads_);
        
        // Initialize mutexes using unique_ptr
        for (size_t i = 0; i < num_threads_; ++i) {
            queue_mutexes_.push_back(std::make_unique<std::mutex>());
        }
    }
    
    ~WorkStealingScheduler() {
        shutdown();
    }
    
    // Start scheduler with worker function
    void start(WorkerFunction func) {
        worker_func_ = func;
        
        // Start worker threads
        workers_.reserve(num_threads_);
        for (size_t i = 0; i < num_threads_; ++i) {
            workers_.emplace_back(&WorkStealingScheduler::workerThread, this, i);
        }
    }
    
    // Submit task to scheduler
    void submit(Task task, size_t preferred_thread = 0) {
        submit_internal(std::move(task), preferred_thread, true);
    }

private:
    void submit_internal(Task task, size_t preferred_thread, bool increment_counter) {
        if (increment_counter) {
            active_tasks_++;
        }
        size_t thread_id = preferred_thread % num_threads_;
        {
            std::lock_guard<std::mutex> lock(*queue_mutexes_[thread_id]);
            task_queues_[thread_id].push(std::move(task));
        }
        cv_.notify_one();
    }

public:
    
    // Wait for all tasks to complete
    void wait_completion() {
        using namespace std::chrono_literals;
        
        // Poll until all tasks done
        while (active_tasks_.load() > 0) {
            std::this_thread::sleep_for(10ms);
        }
        
        // Shutdown workers
        shutdown();
    }
    
    void shutdown() {
        done_ = true;
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }
    
private:
    void workerThread(size_t thread_id) {
        while (!done_) {
            Task task;
            
            // Try to get task from own queue
            if (tryPopTask(thread_id, task)) {
                // Execute task and generate new tasks
                auto new_tasks = worker_func_(task);
                
                // Update counter: -1 for current task, +N for new tasks
                active_tasks_.fetch_add(new_tasks.size() - 1);
                
                // Submit new tasks
                for (auto& new_task : new_tasks) {
                    submit_internal(std::move(new_task), thread_id, false);
                }
                
                if (active_tasks_.load() == 0) {
                    done_cv_.notify_all();
                }
                continue;
            }
            
            // Try work stealing from other threads
            bool stolen = false;
            for (size_t i = 1; i < num_threads_; ++i) {
                size_t victim = (thread_id + i) % num_threads_;
                if (tryStealTask(victim, task)) {
                    // Execute stolen task
                    auto new_tasks = worker_func_(task);
                    
                    // Update counter: -1 for stolen task, +N for new tasks
                    active_tasks_.fetch_add(new_tasks.size() - 1);
                    
                    // Submit new tasks
                    for (auto& new_task : new_tasks) {
                        submit_internal(std::move(new_task), thread_id, false);
                    }
                    
                    if (active_tasks_.load() == 0) {
                        done_cv_.notify_all();
                    }
                    stolen = true;
                    break;
                }
            }
            
            if (!stolen && active_tasks_.load() == 0) {
                // No work available and no active tasks
                break;
            } else if (!stolen) {
                // Wait for more work
                std::unique_lock<std::mutex> lock(cv_mutex_);
                cv_.wait_for(lock, std::chrono::milliseconds(1));
            }
        }
    }
    
    bool tryPopTask(size_t thread_id, Task& task) {
        std::lock_guard<std::mutex> lock(*queue_mutexes_[thread_id]);
        if (task_queues_[thread_id].empty()) {
            return false;
        }
        task = std::move(task_queues_[thread_id].front());
        task_queues_[thread_id].pop();
        return true;
    }
    
    bool tryStealTask(size_t victim_id, Task& task) {
        std::lock_guard<std::mutex> lock(*queue_mutexes_[victim_id]);
        if (task_queues_[victim_id].empty()) {
            return false;
        }
        task = std::move(task_queues_[victim_id].front());
        task_queues_[victim_id].pop();
        return true;
    }
    
    size_t num_threads_;
    std::atomic<bool> done_;
    std::atomic<size_t> active_tasks_;
    WorkerFunction worker_func_;
    
    std::vector<std::thread> workers_;
    std::vector<std::queue<Task>> task_queues_;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes_;
    
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    
    std::mutex done_mutex_;
    std::condition_variable done_cv_;
};

/**
 * Parallel Branch-and-Bound coordinator
 * Manages shared state and best solution across threads
 */
template<typename Solution>
class ParallelBnBCoordinator {
public:
    ParallelBnBCoordinator() : best_value_(0), nodes_explored_(0) {}
    
    // Update best solution (thread-safe)
    bool updateBest(const Solution& solution, size_t value) {
        std::lock_guard<std::mutex> lock(best_mutex_);
        if (value > best_value_) {
            best_solution_ = solution;
            best_value_ = value;
            return true;
        }
        return false;
    }
    
    // Get current best value for pruning
    size_t getBestValue() const {
        std::lock_guard<std::mutex> lock(best_mutex_);
        return best_value_;
    }
    
    // Get best solution
    Solution getBestSolution() const {
        std::lock_guard<std::mutex> lock(best_mutex_);
        return best_solution_;
    }
    
    // Increment nodes explored counter
    void incrementNodes(size_t count = 1) {
        nodes_explored_ += count;
    }
    
    size_t getNodesExplored() const {
        return nodes_explored_;
    }
    
private:
    mutable std::mutex best_mutex_;
    Solution best_solution_;
    size_t best_value_;
    std::atomic<size_t> nodes_explored_;
};

} // namespace QuasiGraph
