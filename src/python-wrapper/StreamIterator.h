#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;

// This class will be exposed to Python as an iterator for streaming results.
class StreamIterator {
public:
    StreamIterator() = default;

    ~StreamIterator() = default;

    // Called by Python's `__next__`
    std::string next() {
        std::unique_lock<std::mutex> lock(q_mutex);

        // Wait without holding GIL
        {
            py::gil_scoped_release release;
            cv.wait(lock, [this] { return !queue.empty() || finished || error_occurred; });
        }

        // Check for errors first
        if (error_occurred) {
            throw std::runtime_error("Stream error occurred");
        }

        if (queue.empty() && finished) {
            throw py::stop_iteration();
        }


        std::string token = std::move(queue.front());
        queue.pop();
        return token;
    }

    void push(const std::string &token) { {
            std::lock_guard lock(q_mutex);
            if (finished) return; // Don't push after finishing
            queue.push(token);
        }
        cv.notify_one();
    }

    void end() { {
            std::lock_guard lock(q_mutex);
            finished = true;
        }
        cv.notify_one();
    }

    void set_error() { {
            std::lock_guard lock(q_mutex);
            error_occurred = true;
        }
        cv.notify_one();
    }

private:
    std::queue<std::string> queue;
    std::mutex q_mutex;
    std::condition_variable cv;
    bool finished = false;
    bool error_occurred = false;
};
