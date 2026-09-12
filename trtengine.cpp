#include "trtengine.h"
#include "logger.h"

#include <fstream>
#include <chrono>
#include <iterator>

namespace {
class TrtLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            Logger::instance().log(Logger::WARNING, QString("TensorRT: %1").arg(msg));
    }
} gLogger;

// 各数据类型的单元素字节数：用于按真实 dtype 分配/拷贝输出，
// 兼容语义分割 argmax 后的 INT32 类别索引图等非 float 输出。
size_t dataTypeBytes(nvinfer1::DataType dt)
{
    switch (dt) {
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF:  return 2;
    case nvinfer1::DataType::kINT8:  return 1;
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kBOOL:  return 1;
    default:                         return 4;
    }
}
}

TrtEngine::TrtEngine() = default;

TrtEngine::~TrtEngine()
{
    release();
}

bool TrtEngine::load(const std::string &enginePath)
{
    if (m_initialized)
        release();

    std::ifstream file(enginePath, std::ios::binary);
    if (!file.good()) {
        Logger::instance().error(QString("无法打开 engine 文件: %1").arg(enginePath.c_str()));
        return false;
    }
    std::vector<char> modelData((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
    file.close();

    cudaSetDevice(0);

    m_runtime.reset(nvinfer1::createInferRuntime(gLogger));
    if (!m_runtime) {
        Logger::instance().error("createInferRuntime 失败");
        return false;
    }

    m_engine.reset(m_runtime->deserializeCudaEngine(modelData.data(), modelData.size()));
    if (!m_engine) {
        Logger::instance().error("deserializeCudaEngine 失败");
        return false;
    }

    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        Logger::instance().error("createExecutionContext 失败");
        return false;
    }

    cudaStreamCreate(&m_stream);

    for (int i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char *name = m_engine->getIOTensorName(i);
        nvinfer1::Dims dims = m_engine->getTensorShape(name);
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) size *= dims.d[j];
        size *= sizeof(float);
        void *buf = nullptr;
        cudaMalloc(&buf, size);
        m_buffers.push_back(buf);
        // enqueueV3 要求通过 setTensorAddress 绑定输入/输出缓冲区地址
        m_context->setTensorAddress(name, buf);

        if (m_engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            m_inputIndex = i;
            m_inputDims.push_back(dims);
            m_inputH = dims.d[2];
            m_inputW = dims.d[3];
        } else {
            m_outputIndices.push_back(i);   // 支持多输出：记录每个输出的 binding index
            m_outputDims.push_back(dims);
            m_outputTypes.push_back(m_engine->getTensorDataType(name));  // 记录真实 dtype
            m_outputNames.push_back(name);  // 记录张量名，供上层按名定位输出
        }
    }

    // 为每个输出张量按形状预分配独立锁页内存，供 forward() 中异步 D2H 复用
    for (size_t k = 0; k < m_outputDims.size(); ++k) {
        nvinfer1::Dims od = m_outputDims[k];
        size_t oSize = 1;
        for (int d = 0; d < od.nbDims; ++d) oSize *= od.d[d];
        size_t elemBytes = dataTypeBytes(m_outputTypes[k]);   // 按真实 dtype 分配（int32/float 均 4 字节）
        float *host = nullptr;
        if (cudaMallocHost((void **)&host, oSize * elemBytes) != cudaSuccess) {
            Logger::instance().error("cudaMallocHost 分配锁页输出内存失败");
            release();
            return false;
        }
        m_outputHosts.push_back(host);
        m_outputHostSizes.push_back(oSize);
        m_outputElemBytes.push_back(elemBytes);
    }

    // ===== 预热（warmup）=====
    // 冷启动首帧推理实测可达 34~78ms（稳态仅 ~3ms），源于 CUDA 内核模块加载/JIT、
    // TensorRT 惰性初始化、GPU 时钟从空闲 P-state 爬升、首次显存与锁页内存访问等
    // 一次性开销。用与输入张量等大的零 blob 跑通完整 GPU 流水若干次，把这些开销在
    // load 阶段一次性吃掉，使真正的首帧推理即处于稳态。
    if (!m_outputHosts.empty() && !m_inputDims.empty()) {
        nvinfer1::Dims id = m_inputDims.front();
        size_t inElems = 1;
        for (int i = 0; i < id.nbDims; ++i) inElems *= id.d[i];
        std::vector<float> dummy(inElems, 0.0f);
        const int kWarmupIters = 20;
        auto wStart = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < kWarmupIters; ++i) {
            cudaMemcpyAsync(m_buffers[m_inputIndex], dummy.data(),
                            inElems * sizeof(float),
                            cudaMemcpyHostToDevice, m_stream);
            m_context->enqueueV3(m_stream);
            for (size_t k = 0; k < m_outputIndices.size(); ++k)
                cudaMemcpyAsync(m_outputHosts[k], m_buffers[m_outputIndices[k]],
                                m_outputHostSizes[k] * m_outputElemBytes[k],
                                cudaMemcpyDeviceToHost, m_stream);
            cudaStreamSynchronize(m_stream);
        }
        auto wEnd = std::chrono::high_resolution_clock::now();
        double warmMs = std::chrono::duration<double, std::milli>(wEnd - wStart).count();
        Logger::instance().info(QString("引擎预热 %1 次完成，耗时 %2 ms（首帧冷启动尖峰已消除）")
                                    .arg(kWarmupIters).arg(warmMs, 0, 'f', 1));
    }

    m_initialized = true;
    Logger::instance().info("TensorRT 引擎加载完成");
    return true;
}

void TrtEngine::release()
{
    for (void *buf : m_buffers)
        cudaFree(buf);
    m_buffers.clear();
    m_inputDims.clear();
    m_outputDims.clear();
    m_outputIndices.clear();
    for (float *h : m_outputHosts)
        cudaFreeHost(h);
    m_outputHosts.clear();
    m_outputHostSizes.clear();
    m_outputTypes.clear();
    m_outputNames.clear();
    m_outputElemBytes.clear();
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    m_context.reset();
    m_engine.reset();
    m_runtime.reset();
    m_initialized = false;
}

bool TrtEngine::isLoaded() const { return m_initialized; }

const float *TrtEngine::forward(const cv::Mat &inputBlobNCHW)
{
    if (!m_initialized || inputBlobNCHW.empty() || m_outputHosts.empty())
        return nullptr;

    auto t1 = std::chrono::high_resolution_clock::now();

    cudaMemcpyAsync(m_buffers[m_inputIndex], inputBlobNCHW.data,
                    inputBlobNCHW.total() * inputBlobNCHW.elemSize(),
                    cudaMemcpyHostToDevice, m_stream);

    auto t2 = std::chrono::high_resolution_clock::now();

    m_context->enqueueV3(m_stream);
    // 每个输出拷贝到各自锁页缓冲：cudaMemcpyAsync 对 pinned memory 才能真正异步，
    // 复用 load() 预分配的缓冲，避免每帧堆分配。
    for (size_t k = 0; k < m_outputIndices.size(); ++k) {
        cudaMemcpyAsync(m_outputHosts[k], m_buffers[m_outputIndices[k]],
                        m_outputHostSizes[k] * m_outputElemBytes[k],
                        cudaMemcpyDeviceToHost, m_stream);
    }
    cudaStreamSynchronize(m_stream);

    auto t3 = std::chrono::high_resolution_clock::now();

    m_lastH2DMs   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    m_lastInferMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
    return m_outputHosts[0];   // 向后兼容：返回第 0 个输出
}

int TrtEngine::inputW() const { return m_inputW; }
int TrtEngine::inputH() const { return m_inputH; }

int TrtEngine::outputCount() const { return (int)m_outputIndices.size(); }

const float *TrtEngine::outputData(int i) const
{
    if (i < 0 || i >= (int)m_outputHosts.size())
        return nullptr;
    return m_outputHosts[i];
}

std::vector<int> TrtEngine::outputShape(int i) const
{
    std::vector<int> shape;
    if (i >= 0 && i < (int)m_outputDims.size()) {
        nvinfer1::Dims od = m_outputDims[i];
        for (int d = 0; d < od.nbDims; ++d)
            shape.push_back(od.d[d]);
    }
    return shape;
}

size_t TrtEngine::outputSize(int i) const
{
    if (i < 0 || i >= (int)m_outputHostSizes.size())
        return 0;
    return m_outputHostSizes[i];
}

// 原始字节指针：整型输出（如 INT32 类别索引图）需由此取数据并按 dtype 解释
const void *TrtEngine::outputRaw(int i) const
{
    if (i < 0 || i >= (int)m_outputHosts.size())
        return nullptr;
    return static_cast<const void *>(m_outputHosts[i]);
}

nvinfer1::DataType TrtEngine::outputDataType(int i) const
{
    if (i < 0 || i >= (int)m_outputTypes.size())
        return nvinfer1::DataType::kFLOAT;
    return m_outputTypes[i];
}

// 输出张量名：供上层按名字（如 pred_score/anomaly_map）定位输出，规避绑定顺序差异
std::string TrtEngine::outputName(int i) const
{
    if (i < 0 || i >= (int)m_outputNames.size())
        return std::string();
    return m_outputNames[i];
}

// 无参兼容版：语义锁定第 0 个输出（单输出任务无需感知多输出接口）
std::vector<int> TrtEngine::outputShape() const { return outputShape(0); }
size_t TrtEngine::outputSize() const { return outputSize(0); }
double TrtEngine::lastH2DMs() const { return m_lastH2DMs; }
double TrtEngine::lastInferMs() const { return m_lastInferMs; }
