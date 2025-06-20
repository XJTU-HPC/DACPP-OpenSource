#include <sycl/sycl.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <functional>
#include <memory>

// 数据句柄
struct DataHandle {
    void* host_ptr;
    size_t size;
    std::unordered_map<sycl::device, void*> device_ptrs;
    std::unordered_map<sycl::device, bool> valid_on_device;
};

// 数据视图
struct DataView {
    std::shared_ptr<DataHandle> handle;
    size_t offset;
    size_t length;
    bool is_output;
};

// 任务结构
struct Task {
    std::string name;
    std::vector<DataView> inputs;
    std::vector<DataView> outputs;
    std::function<void(sycl::handler&, const std::vector<void*>&)> kernel;
};

// 设备状态
struct DeviceState {
    sycl::queue queue;
    std::unordered_map<std::shared_ptr<DataHandle>, bool> valid_data;
    sycl::event last_event;
};

class DeviceManager {
private:
    std::vector<sycl::event> all_events;
    std::vector<DeviceState> devices;
    std::mutex mutex;

public:
    
    DeviceManager() {
        auto platforms = sycl::platform::get_platforms();
        for (auto& p : platforms) {
            auto devs = p.get_devices(sycl::info::device_type::gpu);
            for (auto& d : devs) {
                sycl::queue q(d);
                devices.push_back(DeviceState{q, {}, sycl::event{}});
                // std::cout << "设备已初始化: " << d.get_info<sycl::info::device::name>() << std::endl;
            }
        }
    }

    bool has_valid_data(int dev_id, const std::shared_ptr<DataHandle>& data) {
        return devices[dev_id].valid_data.count(data) && devices[dev_id].valid_data[data];
    }

    void migrate_data_if_needed(int dev_id, const std::vector<DataView>& views) {
        for (auto& view : views) {
            auto& handle = view.handle;
            if (!has_valid_data(dev_id, handle)) {
                // 数据迁移（简化为 memcpy）
                void* device_ptr = sycl::malloc_device(handle->size, devices[dev_id].queue);
                devices[dev_id].queue.memcpy(device_ptr, handle->host_ptr, handle->size).wait();
                handle->device_ptrs[devices[dev_id].queue.get_device()] = device_ptr;
                devices[dev_id].valid_data[handle] = true;
            }
        }
    }

    void submit_task(const Task& task) {
        std::lock_guard<std::mutex> lock(mutex);
    
        int dev_id = 0; // 简化选择策略
        auto& device = devices[dev_id];
        auto queue = device.queue;
        auto sycl_dev = queue.get_device();
    
        // 迁移输入数据（如果尚未存在）
        migrate_data_if_needed(dev_id, task.inputs);
    
        // 确保输出数据有分配（但不需要复制 host 到 device）
        for (auto& view : task.outputs) {
            auto& handle = view.handle;
    
            // 如果未分配 device_ptr，分配 device 内存
            if (handle->device_ptrs.count(sycl_dev) == 0) {
                void* device_ptr = sycl::malloc_device(handle->size, queue);
                handle->device_ptrs[sycl_dev] = device_ptr;
            }
    
            // 标记 output 数据为无效（后续执行内核后再标记为有效）
            device.valid_data[handle] = false;
        }
    
        // 提交内核
        sycl::event evt = devices[dev_id].queue.submit([&](sycl::handler& cgh) {
            std::vector<void*> args;
            for (auto& view : task.inputs) {
                auto ptr = view.handle->device_ptrs[devices[dev_id].queue.get_device()];
                args.push_back(static_cast<char*>(ptr) + view.offset);
            }
            for (auto& view : task.outputs) {
                auto ptr = view.handle->device_ptrs[devices[dev_id].queue.get_device()];
                args.push_back(static_cast<char*>(ptr) + view.offset);
            }
            task.kernel(cgh, args);
        });
        
        devices[dev_id].last_event = evt;
        
        // 保存事件
        all_events.push_back(evt);
    
        // 执行完毕后标记输出数据为有效
        for (auto& view : task.outputs) {
            device.valid_data[view.handle] = true;
        }
    }

    bool wait_all() {
        std::lock_guard<std::mutex> lock(mutex);
        bool all_success = true;
    
        // 所有输出数据收集器：确保只拷贝一次每个 handle
        std::unordered_map<std::shared_ptr<DataHandle>, sycl::device> output_data_to_copy;
    
        for (size_t dev_id = 0; dev_id < devices.size(); ++dev_id) {
            auto& device = devices[dev_id];
            try {
                device.last_event.wait_and_throw();
            } catch (const sycl::exception& e) {
                std::cerr << "任务执行失败: " << e.what() << std::endl;
                all_success = false;
            }
    
            // 记录哪些 DataHandle 在该设备上有效，后续需要从这些设备拷贝回 host
            for (const auto& pair : device.valid_data) {
                const auto& handle = pair.first;
                const auto& valid = pair.second;
                if (valid) {
                    // 只记录输出数据，避免中间输入数据反复 copy
                    output_data_to_copy[handle] = device.queue.get_device();
                }
            }
        }
    
        // 将所有输出数据复制回主机
        for (const auto& [handle, dev] : output_data_to_copy) {
            void* device_ptr = handle->device_ptrs[dev];
            // 找到对应的队列
            auto it = std::find_if(devices.begin(), devices.end(), [&](const DeviceState& ds) {
                return ds.queue.get_device() == dev;
            });
            if (it != devices.end()) {
                it->queue.memcpy(handle->host_ptr, device_ptr, handle->size).wait();
            }
        }
        all_events.clear();
        return all_success;
    }
};