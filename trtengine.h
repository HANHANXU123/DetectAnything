#ifndef TRTENGINE_H
#define TRTENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <NvInfer.h>
#include <cuda_runtime.h>

// 通用 TensorRT 推理引擎：封装与具体任务无关的管线（反序列化 engine、
// 分配显存与锁页输出、创建 stream、enqueueV3、预热、分段计时）。
// 成员直接持有（本项目不使用 Pimpl/Impl 模式），故头文件直接依赖 NvInfer/CUDA；
// 上层 worker/UI 通过 itask.h(ITask) 解耦，不会包含本头文件。
class TrtEngine
{
public:
    TrtEngine();
    ~TrtEngine();
    TrtEngine(const TrtEngine &) = delete;
    TrtEngine &operator=(const TrtEngine &) = delete;

    // 加载 engine 文件并完成显存/锁页分配与预热；成功返回 true
    bool load(const std::string &enginePath);
    void release();
    bool isLoaded() const;

    // 输入 NCHW float blob，内部完成 H2D→enqueueV3→D2H(锁页)→同步，
    // 返回第 0 个输出的锁页缓冲指针（下次 forward 会覆盖）；失败返回 nullptr。
    // 多输出模型：所有输出会被拷回各自锁页缓冲，用 outputData(i) 取第 i 个。
    const float *forward(const cv::Mat &inputBlobNCHW);

    int inputW() const;
    int inputH() const;

    // 多输出支持：共 outputCount() 个输出，按 engine 绑定顺序索引。
    int outputCount() const;
    const float *outputData(int i) const;         // 第 i 个输出数据指针（越界返回 nullptr）
    std::vector<int> outputShape(int i) const;    // 第 i 个输出形状，如 {1,300,38}
    size_t outputSize(int i) const;               // 第 i 个输出元素个数（越界返回 0）

    // 原始输出访问：按 outputDataType(i) 解释 outputRaw(i) 指向的字节。
    // outputData(i) 仅当输出为 float 时有效；整型输出（如语义分割图内 argmax
    // 后的 INT32 类别索引图）必须用 outputRaw(i) + outputDataType(i) 正确解释，
    // 否则把 int32 位模式当 float 读会得到完全错误的数值。
    const void *outputRaw(int i) const;             // 第 i 个输出原始字节指针（越界返回 nullptr）
    nvinfer1::DataType outputDataType(int i) const; // 第 i 个输出数据类型（越界返回 kFLOAT）
    std::string outputName(int i) const;            // 第 i 个输出张量名（越界返回空串）

    std::vector<int> outputShape() const;         // = outputShape(0)，向后兼容单输出任务
    size_t outputSize() const;                    // = outputSize(0)，向后兼容单输出任务

    double lastH2DMs() const;               // 最近一次 forward 的 H2D 段耗时
    double lastInferMs() const;             // 最近一次 forward 的 GPU(推理+D2H+同步)段耗时

private:
    std::unique_ptr<nvinfer1::IRuntime> m_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    std::vector<void *> m_buffers;
    std::vector<nvinfer1::Dims> m_inputDims;
    std::vector<nvinfer1::Dims> m_outputDims;

    int m_inputIndex = 0;
    std::vector<int> m_outputIndices;   // 所有输出的 binding index（按绑定顺序）
    cudaStream_t m_stream = nullptr;

    int m_inputH = 640;
    int m_inputW = 640;
    bool m_initialized = false;

    // 锁页内存（pinned memory）输出缓冲：每个输出一个，使 D2H 拷贝真正异步，
    // 且一次分配、多帧复用，避免每帧 std::vector 堆分配开销。
    std::vector<float *> m_outputHosts;
    std::vector<size_t> m_outputHostSizes;   // 各输出元素个数（非字节数）
    std::vector<nvinfer1::DataType> m_outputTypes;  // 各输出真实数据类型（float/int32/...）
    std::vector<std::string> m_outputNames;         // 各输出张量名（按绑定顺序，供按名取用）
    std::vector<size_t> m_outputElemBytes;          // 各输出单元素字节数（float/int32 均为 4）

    double m_lastH2DMs = 0.0;
    double m_lastInferMs = 0.0;
};

#endif // TRTENGINE_H
