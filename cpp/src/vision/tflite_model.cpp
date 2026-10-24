#include "vision/tflite_model.h"

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <stdexcept>

#include "tensorflow/lite/core/kernels/register.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/model_builder.h"
#ifdef MEMECAM_HAVE_GPU
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "vision/gpu.h"
#endif

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
    TfLiteDelegate* gpu = nullptr;
    ~State() {
        interp.reset();
        if (xnnpack) TfLiteXNNPackDelegateDelete(xnnpack);
#ifdef MEMECAM_HAVE_GPU
        if (gpu) TfLiteGpuDelegateV2Delete(gpu);
#endif
    }
};

#ifdef MEMECAM_HAVE_GPU
namespace {
// GPU delegate: FP16, tuned for repeated frames, compiled kernels cached on disk (the first start compiles
// them: a few seconds). `cache` false retries without the cache (the delegate fails if it can't write it).
// Returns nullptr when the build / driver has no usable OpenCL GPU.
TfLiteDelegate* make_gpu_delegate(const std::string& path, bool cache) {
    static const std::string dir = gpu_cache_dir();
    std::error_code ec;
    const std::string token = std::filesystem::path(path).filename().string() + "-" +
                              std::to_string(std::filesystem::file_size(path, ec));
    TfLiteGpuDelegateOptionsV2 o = TfLiteGpuDelegateOptionsV2Default();
    o.is_precision_loss_allowed = 1;
    o.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
    o.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
    o.inference_priority2 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    o.inference_priority3 = TFLITE_GPU_INFERENCE_PRIORITY_AUTO;
    o.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_CL_ONLY;
    if (cache && !dir.empty()) {
        o.experimental_flags |= TFLITE_GPU_EXPERIMENTAL_FLAGS_ENABLE_SERIALIZATION;
        o.serialization_dir = dir.c_str();
        o.model_token = token.c_str();   // copied by the delegate
    }
    return TfLiteGpuDelegateV2Create(&o);
}
}  // namespace
#endif

TfliteModel::TfliteModel(const std::string& path, int threads, bool gpu) : path_(path), s_(std::make_unique<State>()) {
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
#ifdef MEMECAM_HAVE_GPU
    for (bool cache : {true, false}) {
        if (!gpu || s_->gpu) break;
        TfLiteDelegate* d = make_gpu_delegate(path, cache);
        if (!d) break;
        if (interp_->ModifyGraphWithDelegate(d) == kTfLiteOk && interp_->AllocateTensors() == kTfLiteOk) {
            s_->gpu = d;
            break;
        }
        // A failed delegate can leave the interpreter half-modified: rebuild it (the delegate must outlive it).
        interp_.reset();
        TfLiteGpuDelegateV2Delete(d);
        tflite::InterpreterBuilder(*model_, resolver)(&interp_);
        interp_->SetNumThreads(threads);
    }
    if (gpu && !s_->gpu) {
        static std::once_flag warned;
        std::call_once(warned, [] { std::printf("[memecam] the GPU delegate failed for a model; it runs on the CPU\n"); });
    }
#else
    (void)gpu;
#endif
    if (!s_->gpu) {
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
    }
    if (interp_->AllocateTensors() != kTfLiteOk) throw std::runtime_error("cannot allocate tensors for " + path);
    for (int i = 0; i < num_inputs(); ++i)
        if (s_->interp->input_tensor(i)->type != kTfLiteFloat32)
            throw std::runtime_error("non-float input in " + path);
}

TfliteModel::~TfliteModel() = default;

bool TfliteModel::on_gpu() const { return s_->gpu != nullptr; }

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
