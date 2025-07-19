#pragma once
#include <pybind11/pybind11.h>
namespace py = pybind11;

// This class will be exposed to Python as an iterator for streaming results.
class StreamIterator {
public:
    StreamIterator() = default;

    ~StreamIterator() = default;

    bool is_valid_utf8(const std::string& str) {
        int c, i, ix, n;
        for (ix = 0, n = str.length(); ix < n; ix++) {
            c = (unsigned char) str[ix];
            if (c <= 0x7F) continue;
            else if ((c & 0xE0) == 0xC0) i = 1;
            else if ((c & 0xF0) == 0xE0) i = 2;
            else if ((c & 0xF8) == 0xF0) i = 3;
            else return false;

            if (ix + i >= n) return false;
            while (i--) {
                if ((++ix >= n) || ((str[ix] & 0xC0) != 0x80)) return false;
            }
        }
        return true;
    }

    std::string next() {
        std::unique_lock<std::mutex> lock(q_mutex);

        {
            py::gil_scoped_release release;
            cv.wait(lock, [this] { return !queue.empty() || finished || error_occurred; });
        }

        if (error_occurred) {
            throw std::runtime_error("Stream error occurred");
        }

        if (queue.empty() && finished) {
            throw py::stop_iteration();
        }

        utf8_buffer += std::move(queue.front());
        queue.pop();

        while (!is_valid_utf8(utf8_buffer)) {
            if (queue.empty()) {
                if (finished) {
                    throw py::stop_iteration();
                }

                // Wait for more data
                cv.wait(lock, [this] { return !queue.empty() || finished || error_occurred; });
                if (error_occurred) throw std::runtime_error("Stream error occurred");
                if (queue.empty()) continue;
            }

            utf8_buffer += std::move(queue.front());
            queue.pop();
        }

        std::string complete_token = std::move(utf8_buffer);
        utf8_buffer.clear();
        return complete_token;
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
    std::string utf8_buffer;
    std::queue<std::string> queue;
    std::mutex q_mutex;
    std::condition_variable cv;
    bool finished = false;
    bool error_occurred = false;
};
