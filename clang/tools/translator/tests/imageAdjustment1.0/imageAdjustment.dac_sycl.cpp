#include <iostream>
#include <vector>
#include "ReconTensor.h"

namespace dacpp {
    typedef std::vector<std::any> list;
}

// 定义图像类型，假设每个像素由 RGB 三个值组成
struct Pixel {
    int r, g, b;
    // 重载 << 操作符，使 Pixel 类型的对象可以被输出
    friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel) {
        os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
        return os;
    }
};




// 色彩调整操作：增加红色分量


// 亮度增强操作：增加每个像素的 RGB 分量




// 打印图像的前几个像素，作为调试
void print_image(const std::vector<std::vector<Pixel>>& image, int num_rows = 5, int num_cols = 5) {
    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            std::cout << "(" << image[i][j].r << "," << image[i][j].g << "," << image[i][j].b << ") ";
        }
        std::cout << std::endl;
    }
}

#include <sycl/sycl.hpp>
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void image_1(Pixel* image_tensor,Pixel* image_tensor2,sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor2_acc) 
{
    image_tensor2[0].r = std::min(255, image_tensor[0].r + 50);
}


// 生成函数调用
void imageAdjustment_image_1(const dacpp::Tensor<Pixel, 2> & image_tensor, dacpp::Tensor<Pixel, 2> & image_tensor2) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_image_tensor;
    info_image_tensor.dim = image_tensor.getDim();
    for(int i = 0; i < info_image_tensor.dim; i++) info_image_tensor.dimLength.push_back(image_tensor.getShape(i));
    // 数据信息初始化
    DataInfo info_image_tensor2;
    info_image_tensor2.dim = image_tensor2.getDim();
    for(int i = 0; i < info_image_tensor2.dim; i++) info_image_tensor2.dimLength.push_back(image_tensor2.getShape(i));
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_image_tensor));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_image_tensor));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops image_tensor_Ops;
    
    idx1.setDimId(0);
    image_tensor_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops image_tensor2_Ops;
    
    idx1.setDimId(0);
    image_tensor2_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor2_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int image_tensor_Size = para_gene_tool.init_device_memory_size(info_image_tensor,image_tensor_Ops);

    //生成设备内存分配大小
    int image_tensor2_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_image_tensor2);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_image_tensor2,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(image_tensor_Ops,image_tensor_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,image_tensor2_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(image_tensor_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    Pixel *d_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    // 设备内存分配
    Pixel *d_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    // 归约设备内存分配
    Pixel *reduction_image_tensor2 = malloc_device<Pixel>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<Pixel> image_tensor_tool;
    Pixel* r_image_tensor=(Pixel*)malloc(sizeof(Pixel)*image_tensor_Size);
    
    // 数据算子组初始化
    Dac_Ops image_tensor_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    image_tensor_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    image_tensor_ops.push_back(idx2);
    image_tensor_tool.init(info_image_tensor,image_tensor_ops);
    image_tensor_tool.Reconstruct(r_image_tensor,image_tensor);
	std::vector<int> info_partition_image_tensor=para_gene_tool.init_partition_data_shape(info_image_tensor,image_tensor_ops);
    sycl::buffer<int> info_partition_image_tensor_buffer(info_partition_image_tensor.data(), sycl::range<1>(info_partition_image_tensor.size()));
    // 数据重组
    DataReconstructor<Pixel> image_tensor2_tool;
    Pixel* r_image_tensor2=(Pixel*)malloc(sizeof(Pixel)*image_tensor2_Size);
    
    // 数据算子组初始化
    Dac_Ops image_tensor2_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    image_tensor2_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    image_tensor2_ops.push_back(idx2);
    image_tensor2_tool.init(info_image_tensor2,image_tensor2_ops);
    image_tensor2_tool.Reconstruct(r_image_tensor2,image_tensor2);
	std::vector<int> info_partition_image_tensor2=para_gene_tool.init_partition_data_shape(info_image_tensor2,image_tensor2_ops);
    sycl::buffer<int> info_partition_image_tensor2_buffer(info_partition_image_tensor2.data(), sycl::range<1>(info_partition_image_tensor2.size()));
    
    // 数据移动
    q.memcpy(d_image_tensor,r_image_tensor,image_tensor_Size*sizeof(Pixel)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_image_tensor_accessor = info_partition_image_tensor_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_image_tensor2_accessor = info_partition_image_tensor2_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            image_1(d_image_tensor+(idx1_*SplitLength[0][0]+idx2_*SplitLength[0][1]),d_image_tensor2+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),info_partition_image_tensor_accessor,info_partition_image_tensor2_accessor);
        });
    }).wait();
    

	
    // // 归约
    // if(Reduction_Split_Size > 1)
    // {
    //     for(int i=0;i<Reduction_Size;i++) {
    //         q.submit([&](handler &h) {
    // 	        h.parallel_for(
    //             range<1>(Reduction_Split_Size),
    //             reduction(reduction_image_tensor2+i, 
    //             sycl::plus<>(),
    //             property::reduction::initialize_to_identity()),
    //             [=](id<1> idx,auto &reducer) {
    //                 reducer.combine(d_image_tensor2[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
    //  	        });
    //      }).wait();
    //     }
    //     q.memcpy(d_image_tensor2,reduction_image_tensor2, Reduction_Size*sizeof(Pixel)).wait();
    // }


	
    // 归并结果返回
    q.memcpy(r_image_tensor2, d_image_tensor2, image_tensor2_Size*sizeof(Pixel)).wait();
    image_tensor2_tool.UpdateData(r_image_tensor2,image_tensor2);

    // 内存释放
    
    sycl::free(d_image_tensor, q);
    sycl::free(d_image_tensor2, q);
}

void image_2(Pixel* image_tensor2,Pixel* image_tensor3,sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor2_acc, sycl::accessor<int, 1, sycl::access::mode::read_write> info_image_tensor3_acc) 
{
    int value = 30;
    image_tensor3[0].r = std::min(255, image_tensor2[0].r + value);
    image_tensor3[0].g = std::min(255, image_tensor2[0].g + value);
    image_tensor3[0].b = std::min(255, image_tensor2[0].b + value);
}


// 生成函数调用
void imageAdjustment_image_2(const dacpp::Tensor<Pixel, 2> & image_tensor, dacpp::Tensor<Pixel, 2> & image_tensor2) { 
    // 设备选择
    auto selector = default_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_image_tensor;
    info_image_tensor.dim = image_tensor.getDim();
    for(int i = 0; i < info_image_tensor.dim; i++) info_image_tensor.dimLength.push_back(image_tensor.getShape(i));
    // 数据信息初始化
    DataInfo info_image_tensor2;
    info_image_tensor2.dim = image_tensor2.getDim();
    for(int i = 0; i < info_image_tensor2.dim; i++) info_image_tensor2.dimLength.push_back(image_tensor2.getShape(i));
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_image_tensor));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_image_tensor));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops image_tensor_Ops;
    
    idx1.setDimId(0);
    image_tensor_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops image_tensor2_Ops;
    
    idx1.setDimId(0);
    image_tensor2_Ops.push_back(idx1);

    idx2.setDimId(1);
    image_tensor2_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    idx1.setDimId(0);
    In_Ops.push_back(idx1);

    idx2.setDimId(1);
    In_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int image_tensor_Size = para_gene_tool.init_device_memory_size(info_image_tensor,image_tensor_Ops);

    //生成设备内存分配大小
    int image_tensor2_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_image_tensor2);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_image_tensor2,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(image_tensor_Ops,image_tensor_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,image_tensor2_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(image_tensor_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    Pixel *d_image_tensor=malloc_device<Pixel>(image_tensor_Size,q);
    // 设备内存分配
    Pixel *d_image_tensor2=malloc_device<Pixel>(image_tensor2_Size,q);
    // 归约设备内存分配
    Pixel *reduction_image_tensor2 = malloc_device<Pixel>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<Pixel> image_tensor_tool;
    Pixel* r_image_tensor=(Pixel*)malloc(sizeof(Pixel)*image_tensor_Size);
    
    // 数据算子组初始化
    Dac_Ops image_tensor_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    image_tensor_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    image_tensor_ops.push_back(idx2);
    image_tensor_tool.init(info_image_tensor,image_tensor_ops);
    image_tensor_tool.Reconstruct(r_image_tensor,image_tensor);
	std::vector<int> info_partition_image_tensor=para_gene_tool.init_partition_data_shape(info_image_tensor,image_tensor_ops);
    sycl::buffer<int> info_partition_image_tensor_buffer(info_partition_image_tensor.data(), sycl::range<1>(info_partition_image_tensor.size()));
    // 数据重组
    DataReconstructor<Pixel> image_tensor2_tool;
    Pixel* r_image_tensor2=(Pixel*)malloc(sizeof(Pixel)*image_tensor2_Size);
    
    // 数据算子组初始化
    Dac_Ops image_tensor2_ops;
    
    idx1.setDimId(0);
    idx1.setSplitLength(8);
    image_tensor2_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    image_tensor2_ops.push_back(idx2);
    image_tensor2_tool.init(info_image_tensor2,image_tensor2_ops);
    image_tensor2_tool.Reconstruct(r_image_tensor2,image_tensor2);
	std::vector<int> info_partition_image_tensor2=para_gene_tool.init_partition_data_shape(info_image_tensor2,image_tensor2_ops);
    sycl::buffer<int> info_partition_image_tensor2_buffer(info_partition_image_tensor2.data(), sycl::range<1>(info_partition_image_tensor2.size()));
    
    // 数据移动
    q.memcpy(d_image_tensor,r_image_tensor,image_tensor_Size*sizeof(Pixel)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        // 访问器初始化
        
        auto info_partition_image_tensor_accessor = info_partition_image_tensor_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_image_tensor2_accessor = info_partition_image_tensor2_buffer.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
            const auto idx1_=(item_id/idx2.split_size+(0))%idx1.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
			
            image_2(d_image_tensor+(idx1_*SplitLength[0][0]+idx2_*SplitLength[0][1]),d_image_tensor2+(idx1_*SplitLength[1][0]+idx2_*SplitLength[1][1]),info_partition_image_tensor_accessor,info_partition_image_tensor2_accessor);
        });
    }).wait();
    

	
    // // 归约
    // if(Reduction_Split_Size > 1)
    // {
    //     for(int i=0;i<Reduction_Size;i++) {
    //         q.submit([&](handler &h) {
    // 	        h.parallel_for(
    //             range<1>(Reduction_Split_Size),
    //             reduction(reduction_image_tensor2+i, 
    //             sycl::plus<>(),
    //             property::reduction::initialize_to_identity()),
    //             [=](id<1> idx,auto &reducer) {
    //                 reducer.combine(d_image_tensor2[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
    //  	        });
    //      }).wait();
    //     }
    //     q.memcpy(d_image_tensor2,reduction_image_tensor2, Reduction_Size*sizeof(Pixel)).wait();
    // }


	
    // 归并结果返回
    q.memcpy(r_image_tensor2, d_image_tensor2, image_tensor2_Size*sizeof(Pixel)).wait();
    image_tensor2_tool.UpdateData(r_image_tensor2,image_tensor2);

    // 内存释放
    
    sycl::free(d_image_tensor, q);
    sycl::free(d_image_tensor2, q);
}

int main() {
    // 初始化一个简单的图像（10x10），所有像素值初始化为(100, 100, 100)

    int width, height;
    std::cout << "Enter width: ";
    std::cin >> width;  // 错误：无法修改const变量

    std::cout << "Enter height: ";
    std::cin >> height;  // 错误：无法修改const变量
    std::vector<Pixel> image(height*width, {100, 100, 100});
    std::vector<Pixel> image2(height*width, {100, 100, 100});
    //std::vector<std::vector<Pixel>> image2(height, std::vector<Pixel>(width, {100, 100, 100}));

    // 打印初始图像
    std::cout << "Original Image:" << std::endl;
    //print_image(image);

    dacpp::Tensor<Pixel, 2> image_tensor({height, width}, image);
    dacpp::Tensor<Pixel, 2> image_tensor2({height, width}, image2);

    // 执行色彩调整操作
    imageAdjustment_image_1(image_tensor, image_tensor2);
    std::cout << "\nImage After Color Adjustment:" << std::endl;

    std::vector<Pixel> image3 = image2;
    dacpp::Tensor<Pixel, 2> image_tensor3({height, width}, image3);


    // 执行亮度增强操作
    imageAdjustment_image_2(image_tensor2, image_tensor3);
    std::cout << "\nImage After Brightness Enhancement:" << std::endl;
    image_tensor3.print();

    return 0;
}
