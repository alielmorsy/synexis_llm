#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include <iostream>
#include <memory>

#include <synexis/Synexis.h>
#include <synexis/TaskParams.h>
#include <synexis/sampler/StructParams.h>
#include "StreamIterator.h"

namespace py = pybind11;

std::shared_ptr<StreamIterator> stream_task(Synexis &self, TaskParams &params) {
    auto iterator = std::make_shared<StreamIterator>();
    params.stream = true;
    params.on_token = [iterator](const std::string &token) {
        iterator->push(token);
    };

    params.on_done = [iterator](const std::string &text) {
        iterator->end();
    };

    //TODO
    // params.on_error = [iterator](const std::string &error) {
    //     iterator->set_error();
    // };

    try {
        py::gil_scoped_release release;
        std::future<std::string> future = self.addTask(params.prompt, params);
    } catch (const std::exception &e) {
        iterator->set_error();
        std::cerr << "Error while adding the task: " << e.what() << std::endl;
    }

    return iterator;
}

std::string get_template(Synexis &self) {
    return self.getTemplate();
}

PYBIND11_MODULE(synexis_python, m) {
    m.doc() = "Python bindings for the Synexis C++ library";

    py::class_<StreamIterator, std::shared_ptr<StreamIterator> >(m, "StreamIterator")
            .def("__iter__", [](std::shared_ptr<StreamIterator> it) -> std::shared_ptr<StreamIterator> { return it; })
            .def("__next__", &StreamIterator::next); {
        SamplingParams defaults{};

        py::class_<SamplingParams>(m, "SamplingParams")
                .def(py::init<float, int32_t, float, float>(),
                     py::arg("temp") = defaults.temp,
                     py::arg("top_k") = (defaults.top_k), // Corrected line
                     py::arg("top_p") = defaults.top_p,
                     py::arg("min_p") = defaults.min_p)
                .def_readwrite("temp", &SamplingParams::temp)
                .def_readwrite("top_k", &SamplingParams::top_k)
                .def_readwrite("top_p", &SamplingParams::top_p)
                .def_readwrite("min_p", &SamplingParams::min_p);
    }

    py::class_<TaskParams>(m, "TaskParams")
            .def(py::init<>())
            .def(py::init<std::string, SamplingParams>(),
                 py::arg("prompt"),
                 py::arg("samplerParams") = SamplingParams())
            .def_readwrite("prompt", &TaskParams::prompt)
            .def_readwrite("samplerParams", &TaskParams::samplerParams)
            .def_readwrite("maximumTokens", &TaskParams::maximumTokens);


    py::class_<Synexis>(m, "Synexis")
            .def(py::init([](const std::string &model_path, int n_slots) {
                // Release the GIL during model loading
                py::gil_scoped_release release;
                return std::make_unique<Synexis>(model_path, n_slots);
            }), py::arg("model_path"), py::arg("n_slots"))
            .def("run", &Synexis::run, py::call_guard<py::gil_scoped_release>(),
                 "Starts the backend processing threads.")

            .def("stop", &Synexis::stop, "Stops the backend processing threads.")
            .def("complete", [](Synexis &self, TaskParams params) {
                params.stream = false;
                params.on_token = nullptr;
                py::gil_scoped_release release;
                std::future<std::string> future = self.addTask(params.prompt, params);
                return future.get();
            }, py::arg("params"), "Adds a task for synchronous (non-streaming) generation.")

            .def("complete_stream", &stream_task, py::arg("params"),
                 "Adds a task for streaming generation and returns an iterator.")
            .def("get_template", &get_template, "Get the model template or fallback to the default one");
}
