// Thin LiteRT (TensorFlow Lite) wrapper: one interpreter per model, float32 tensors, XNNPACK on the CPU or
// the GPU delegate (OpenCL) when asked for and available.
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace memecam::vision {

class TfliteModel {
public:
    // `threads`: XNNPACK / interpreter threads for this model. `gpu`: try the GPU delegate first (falls back
    // to XNNPACK when the build has none, there is no OpenCL GPU or the model doesn't fit it).
    TfliteModel(const std::string& path, int threads, bool gpu = false);
    ~TfliteModel();
    TfliteModel(const TfliteModel&) = delete;
    TfliteModel& operator=(const TfliteModel&) = delete;

    float* input(int i = 0);
    const float* output(int i) const;
    int output_size(int i) const;            // number of floats
    std::vector<int> input_shape(int i = 0) const;
    std::vector<int> output_shape(int i) const;
    int num_inputs() const;
    int num_outputs() const;
    void invoke();
    const std::string& path() const { return path_; }
    bool on_gpu() const;

private:
    std::string path_;
    struct State;                 // LiteRT objects (kept out of this header)
    std::unique_ptr<State> s_;
};

}  // namespace memecam::vision
