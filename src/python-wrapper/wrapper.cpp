#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <queue>
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
        std::cout << "Generated Token " << token << std::endl;
        iterator->push(token);
    };

    params.on_done = [iterator](const std::string &text) {
        iterator->end();
    };
    params.on_error = [](const std::string &error) {
        py::gil_scoped_acquire acquire;
        throw py::value_error(error);
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
    py::class_<SynexisArguments>(m, "SynexisArguments")
            .def(py::init<std::string>(), py::arg("model_path"))
            .def_readwrite("model_path", &SynexisArguments::modelPath)
            .def_readwrite("model_projector_path", &SynexisArguments::modelProjectorPath)
            .def_readwrite("number_of_gpu_layers", &SynexisArguments::numberOfGpuLayers)
            .def_readwrite("number_of_threads", &SynexisArguments::numberOfThreads)
            .def_readwrite("use_mmap", &SynexisArguments::use_mmap)
            .def_readwrite("n_ctx", &SynexisArguments::n_ctx)
            .def_readwrite("n_batch", &SynexisArguments::n_batch)
            .def_readwrite("n_keep", &SynexisArguments::n_keep)
            .def_readwrite("n_discard", &SynexisArguments::n_discard)
            .def_readwrite("n_slots", &SynexisArguments::n_slots);

    py::class_<StreamIterator, std::shared_ptr<StreamIterator> >(m, "StreamIterator")
            .def("__iter__", [](std::shared_ptr<StreamIterator> it) -> std::shared_ptr<StreamIterator> { return it; })
            .def("__next__", &StreamIterator::next); {
        SamplingParams defaults{};

        py::class_<SamplingParams>(m, "SamplingParams")
                .def(py::init<float, int32_t, float, float, float>(),
                     py::arg("temp") = defaults.temp,
                     py::arg("top_k") = defaults.top_k,
                     py::arg("top_p") = defaults.top_p,
                     py::arg("min_p") = defaults.min_p,
                     py::arg("penalty_repeat") = defaults.penalty_repeat)
                .def_readwrite("temp", &SamplingParams::temp)
                .def_readwrite("top_k", &SamplingParams::top_k)
                .def_readwrite("top_p", &SamplingParams::top_p)
                .def_readwrite("min_p", &SamplingParams::min_p);
    }
    py::class_<TaskParams>(m, "TaskParams")
            .def(py::init<>())
            .def(py::init<std::string, SamplingParams, int, std::vector<std::string> >(),
                 py::arg("prompt"),
                 py::arg("sampling_params") = SamplingParams(), py::arg("maximum_tokens") = -1,
                 py::arg("stop_tokens") = std::vector<std::string>())
            .def_readwrite("prompt", &TaskParams::prompt)
            .def_readwrite("sampling_params", &TaskParams::samplerParams)
            .def_readwrite("maximum_tokens", &TaskParams::maximumTokens)
            .def_readwrite("stop_tokens", &TaskParams::stopTokens)
            .def("add_media", [](TaskParams &self, const py::bytes &media) {
                std::string_view view = media;
                self.addMedia(view);
            });

    py::class_<Synexis>(m, "Synexis")
            .def(py::init([](SynexisArguments &args) {
                // Release the GIL during model loading
                py::gil_scoped_release release;
                return std::make_unique<Synexis>(args);
            }), py::arg("args"))

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
            .def("get_template", &get_template, "Get the model template or fallback to the default one").
            def("get_tokens", [](Synexis &self) {
                py::dict d;
                d["bos_token"] = self.getToken("BOS");
                d["eos_token"] = self.getToken("EOS");
                return d;
            });
}
