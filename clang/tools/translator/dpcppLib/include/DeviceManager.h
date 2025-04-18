#include <sycl/sycl.hpp>
#include <vector>
#include <mutex>
#include <iostream>
#include <thread>
#include <chrono>

class DeviceManager {
private:
    std::vector<sycl::queue> devices;  // 设备队列
    std::vector<sycl::event> last_events; // 每个设备的最后一个任务事件
    std::mutex mutex;  // 线程安全锁

public:
    // 构造函数：检测所有 GPU 设备，并初始化
    DeviceManager() {
        auto platforms = sycl::platform::get_platforms();
        for (const auto& platform : platforms) {
            auto device_list = platform.get_devices(sycl::info::device_type::gpu);
            for (const auto& device : device_list) {
                devices.emplace_back(device); // 创建 SYCL 队列
                last_events.emplace_back();  // 初始时没有任务
                sycl::event init_event = devices.back().submit([](sycl::handler& cgh) {
                    cgh.single_task([] {});
                });
                last_events.push_back(init_event);
                std::cout << "检测到设备：" << device.get_info<sycl::info::device::name>() << std::endl;
            }
        }
    }

    // **查找空闲设备**（如果所有设备都忙，则返回 -1）
    int find_free_device() {
        printf("正在检查空闲设备...\n");
        std::lock_guard<std::mutex> lock(mutex);
        // std::cout << "正在检查空闲设备..." << std::endl;
    
        for (size_t i = 0; i < devices.size(); i++) {
            try {
                auto status = last_events[i].get_info<sycl::info::event::command_execution_status>();               
                if (status == sycl::info::event_command_status::complete) {
                    return i;
                }
            } catch (sycl::exception& e) {
                std::cerr << "设备 " << i << " 状态查询失败：" << e.what() << std::endl;
                return i; // 遇到错误时假定设备可用
            }
        }
        std::cout << "所有设备都在工作..." << std::endl;
        return -1;
    }

    // **分配任务到空闲设备**
    bool submit_task(std::function<void(sycl::handler&)> task) {
        // std::lock_guard<std::mutex> lock(mutex);
        int device_id = find_free_device();

        if (device_id == -1) {
            std::cout << "所有设备都在工作，任务排队等待..." << std::endl;
            return false;
        }

        // 提交任务
        last_events[device_id] = devices[device_id].submit([&](sycl::handler& cgh) {
            task(cgh);
        });

        std::cout << "任务提交到设备 " << device_id << "：" 
                  << devices[device_id].get_device().get_info<sycl::info::device::name>()
                  << std::endl;
        return true;
    }
    
    void split_and_submit(size_t total_task_size, std::function<void(sycl::handler&, size_t, size_t)> task) {
        size_t chunk_size = total_task_size / devices.size();  // 将任务划分为每个设备的任务块
    
        for (size_t i = 0; i < devices.size(); i++) {
            size_t start_index = i * chunk_size;
            size_t end_index = (i == devices.size() - 1) ? total_task_size : start_index + chunk_size;
    
            // 使用 submit_task 提交任务
            bool success = submit_task([&, start_index, end_index](sycl::handler& cgh) {
                task(cgh, start_index, end_index);
            });
    
        }
    }
};