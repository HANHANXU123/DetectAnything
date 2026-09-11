#ifndef TASKFACTORY_H
#define TASKFACTORY_H

#include <memory>
#include <QString>

#include "itask.h"

// 任务工厂：按 TaskType 创建具体任务实例，并提供其 engine 相对路径。
// 这是唯一耦合所有具体任务的地方（组合根）；新增任务只需在此登记一行。
class TaskFactory
{
public:
    // 创建任务；尚未实现的类型返回 nullptr（UI 侧按占位处理）
    static std::unique_ptr<ITask> create(TaskType type);
    // 返回该任务对应的 engine 相对路径（相对项目根目录）
    static QString enginePathFor(TaskType type);
};

#endif // TASKFACTORY_H
