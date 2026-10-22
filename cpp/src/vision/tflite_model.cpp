#include "vision/tflite_model.h"

#include <stdexcept>

#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/model_builder.h"

namespace memecam::vision {

namespace {
std::vector<int> dims_of(const TfLiteTensor* t) {
    std::vector<int> out;
    if (t && t->dims)
        for (int i = 0; i < t->dims->size; ++i) out.push_back(t->dims->data[i]);
    return out;
}
}  // namespace

struct TfliteModel::State {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interp;
    TfLiteDelegate* xnnpack = nullptr;
    ~State() {
        interp.reset();
        if (xnnpack) TfLiteXNNPackDelegateDelete(xnnpack);
    }
};

TfliteModel::TfliteModel(const std::string& path, int threads) : path_(path), s_(std::make_unique<State>()) {
    auto& model_ = s_->model;
    auto& interp_ = s_->interp;
    auto& xnnpack_ = s_->xnnpack;
    // LiteRT prints "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." per interpreter; keep quiet.
    tflite::LoggerOptions::SetMinimumLogSeverity(tflite::TFLITE_LOG_WARNING);
    model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
    if (!model_) throw std::runtime_error("cannot load model " + path);
    tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
    if (tflite::InterpreterBuilder(*model_, resolver)(&interp_) != kTfLiteOk || !interp_)
        throw std::runtime_error("cannot build interpreter for " + path);
    interp_->SetNumThreads(threads);
    TfLiteXNNPackDelegateOptions opts = TfLiteXNNPackDelegateOptionsDefault();
    opts.num_threads = threads;
    xnnpack_ = TfLiteXNNPackDelegateCreate(&opts);
    if (xnnpack_ && interp_->ModifyGraphWithDelegate(xnnpack_) != kTfLiteOk) {
        // Unsupported ops: run on the builtin CPU kernels instead.
        interp_.reset();
        TfLiteXNNPackDelegateDelete(xnnpack_);
        xnnpack_ = nullptr;
        tflite::InterpreterBuilder(*model_, resolver)(&interp_);
        interp_->SetNumThreads(threads);
    }
    if (interp_->AllocateTensors() != kTfLiteOk) throw std::runtime_error("cannot allocate tensors for " + path);
    for (int i = 0; i < num_inputs(); ++i)
        if (s_->interp->input_tensor(i)->type != kTfLiteFloat32)
            throw std::runtime_error("non-float input in " + path);
}

TfliteModel::~TfliteModel() = default;

float* TfliteModel::input(int i) { return s_->interp->typed_input_tensor<float>(i); }
const float* TfliteModel::output(int i) const { return s_->interp->typed_output_tensor<float>(i); }
int TfliteModel::output_size(int i) const {
    return static_cast<int>(s_->interp->output_tensor(i)->bytes / sizeof(float));
}
std::vector<int> TfliteModel::input_shape(int i) const { return dims_of(s_->interp->input_tensor(i)); }
std::vector<int> TfliteModel::output_shape(int i) const { return dims_of(s_->interp->output_tensor(i)); }
int TfliteModel::num_inputs() const { return static_cast<int>(s_->interp->inputs().size()); }
int TfliteModel::num_outputs() const { return static_cast<int>(s_->interp->outputs().size()); }

void TfliteModel::invoke() {
    if (s_->interp->Invoke() != kTfLiteOk) throw std::runtime_error("inference failed: " + path_);
}

}  // namespace memecam::vision
