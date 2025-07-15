#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <iostream>
#include <thread>
#include <memory>

#include "../include/synexis/Synexis.h"
#include "../include/synexis/TaskParams.h"
#include "../include/synexis/sampler/StructParams.h"

namespace py = pybind11;

// This class will be exposed to Python as an iterator for streaming results.
class StreamIterator {
public:
    StreamIterator() = default;

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

    ~StreamIterator() {
        std::cout << "StreamIterator destroyed" << std::endl;
    }

    // Thread-safe push method - can be called from any thread
    void push(const std::string &token) {
        {
            std::lock_guard<std::mutex> lock(q_mutex);
            if (finished) return; // Don't push after finishing
            queue.push(token);
        }
        cv.notify_one();
    }

    // Signals the end of the stream - can be called from any thread
    void end() {
        {
            std::lock_guard<std::mutex> lock(q_mutex);
            finished = true;
        }
        cv.notify_one();
    }

    // Signal an error occurred - can be called from any thread
    void set_error() {
        {
            std::lock_guard<std::mutex> lock(q_mutex);
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

PYBIND11_MODULE(synexis_python, m) {
    m.doc() = "Python bindings for the Synexis C++ library";

    py::class_<StreamIterator>(m, "StreamIterator")
            .def("__iter__", [](StreamIterator &it) -> StreamIterator & { return it; })
            .def("__next__", &StreamIterator::next);

    // Binding for SamplingParams, needed by TaskParams
    py::class_<SamplingParams>(m, "SamplingParams")
            .def(py::init<>())
            .def_readwrite("temp", &SamplingParams::temp)
            .def_readwrite("top_k", &SamplingParams::top_k)
            .def_readwrite("top_p", &SamplingParams::top_p)
            .def_readwrite("min_p", &SamplingParams::min_p);

    // Binding for TaskParams
    py::class_<TaskParams>(m, "TaskParams")
            .def(py::init<>())
            .def_readwrite("prompt", &TaskParams::prompt)
            .def_readwrite("samplerParams", &TaskParams::samplerParams)
            .def_readwrite("stream", &TaskParams::stream);

    py::class_<Synexis>(m, "Synexis")
            .def(py::init([](const std::string &model_path, int n_slots) {
                // Release the GIL during model loading
                py::gil_scoped_release release;
                return std::make_unique<Synexis>(model_path, n_slots);
            }), py::arg("model_path"), py::arg("n_slots"))
            .def("run", &Synexis::run, "Starts the backend processing threads.")
            .def("stop", &Synexis::stop, "Stops the backend processing threads.")
            .def("add_task", [](Synexis &self, TaskParams params) {
                // Ensure stream is false for this synchronous method
                params.stream = false;
                params.on_token = nullptr;

                std::future<std::string> future = self.addTask(params.prompt, params);

                // Release the GIL while waiting for the result
                py::gil_scoped_release release;
                return future.get();
            }, py::arg("params"), "Adds a task for synchronous (non-streaming) generation.")
            .def("stream", [](Synexis &self, TaskParams &params) {
                // Use shared_ptr to ensure lifetime management
                auto iterator = std::make_shared<StreamIterator>();

                // Keep the iterator alive by capturing it in the lambdas
                params.stream = true;

                // CRITICAL: Don't acquire GIL in callbacks from other threads
                // The callbacks should be thread-safe without Python involvement
                params.on_token = [iterator](const std::string &token) {
                    // No GIL acquisition here - just call thread-safe method
                    iterator->push(token);
                };

                params.on_done = [iterator](const std::string &text) {
                    // No GIL acquisition here - just call thread-safe method
                    iterator->end();
                };

                // Optional: Add error callback if your TaskParams supports it
                // params.on_error = [iterator](const std::string &error) {
                //     iterator->set_error();
                // };

                try {
                    // The future is used to signal the end of the stream
                    std::future<std::string> future = self.addTask(params.prompt, params);

                    // Don't wait for the future here - let streaming happen asynchronously
                    // The callbacks will handle the completion

                } catch (const std::exception& e) {
                    iterator->set_error();
                }

                return iterator;
            }, py::arg("params"), "Adds a task for streaming generation and returns an iterator.");
}