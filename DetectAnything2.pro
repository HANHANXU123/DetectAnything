QT += widgets

CONFIG += c++17

# 项目根目录加入头文件搜索路径（扁平结构：所有源/头文件同目录）
INCLUDEPATH += $$PWD
# 任务源码子目录：YOLO 目标检测；YOLO-seg 实例分割；segment 语义分割；OCR（det/rec/utils/DB 后处理 + 第三方 clipper）
INCLUDEPATH += $$PWD/yolo $$PWD/yolo-seg $$PWD/segment $$PWD/ocr $$PWD/ocr/clipper

# ============== OpenCV 配置 (MSVC) ==============
OPENCV_SDK = E:/opencv/build
INCLUDEPATH += $$OPENCV_SDK/include
LIBS += -L$$OPENCV_SDK/x64/vc16/lib
CONFIG(debug, debug|release) {
    LIBS += -lopencv_world480d
} else {
    LIBS += -lopencv_world480
}

# ============== TensorRT 配置 ==============
TENSORRT_SDK = E:/TensorRT-8.6.1.6
CUDA_SDK = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8"
INCLUDEPATH += "$$TENSORRT_SDK/include" \
               "$$CUDA_SDK/include"
LIBS += -L"$$TENSORRT_SDK/lib" \
        -lnvinfer \
        -lnvonnxparser \
        -L"$$CUDA_SDK/lib/x64" \
        -lcudart

SOURCES += \
    main.cpp \
    mainwindow.cpp \
    logger.cpp \
    yolo/yolo.cpp \
    yolo-seg/yoloseg.cpp \
    segment/segment.cpp \
    inferenceworker.cpp \
    trtengine.cpp \
    taskfactory.cpp \
    ocr/ocr.cpp \
    ocr/ppocr_det.cpp \
    ocr/ppocr_rec.cpp \
    ocr/ppocr_utils.cpp \
    ocr/postprocess_det.cpp \
    ocr/clipper/clipper.cpp

HEADERS += \
    mainwindow.h \
    logger.h \
    yolo/yolo.h \
    yolo-seg/yoloseg.h \
    segment/segment.h \
    inferenceworker.h \
    itask.h \
    trtengine.h \
    taskfactory.h \
    ocr/ocr.h \
    ocr/ppocr_det.h \
    ocr/ppocr_rec.h \
    ocr/ppocr_utils.h \
    ocr/clipper/clipper.hpp

FORMS += \
    mainwindow.ui

TRANSLATIONS += \
    DetectAnything_zh_CN.ts
CONFIG += lrelease
CONFIG += embed_translations

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

# ============== 部署 DLL ==============
win32 {
    CONFIG(debug, debug|release) {
        OPENCV_DLL = $$OPENCV_SDK/x64/vc16/bin/opencv_world480d.dll
        EXE_DIR = $$OUT_PWD/debug
    } else {
        OPENCV_DLL = $$OPENCV_SDK/x64/vc16/bin/opencv_world480.dll
        EXE_DIR = $$OUT_PWD/release
    }
    QMAKE_POST_LINK = copy /y "$$shell_path($$OPENCV_DLL)" "$$shell_path($$EXE_DIR)\\" $$escape_expand(\\n) \
                      copy /y "$$shell_path($$TENSORRT_SDK/lib/nvinfer.dll)" "$$shell_path($$EXE_DIR)\\" $$escape_expand(\\n) \
                      copy /y "$$shell_path($$TENSORRT_SDK/lib/nvonnxparser.dll)" "$$shell_path($$EXE_DIR)\\"
}
