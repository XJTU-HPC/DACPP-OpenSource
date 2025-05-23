#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include "mpi_template.h"

namespace MPI_TEMPLATE {

void replaceTextInString(std::string& text, 
    const std::string &find, 
    const std::string &replace){
	std::string::size_type pos = 0;
	while ((pos = text.find(find, pos)) != std::string::npos){
		text.replace(pos, find.length(), replace);
		pos += replace.length();
	}
}

std::string templateString(std::string templ, 
    std::vector<std::pair<std::string, std::string>> replacements){
	for(auto &element : replacements)
		replaceTextInString(templ, element.first, element.second);
	return templ;
}

string CodeGen_mpi_datatype(string data_type) {
    if(data_type == "std::complex<double>" || data_type == "complex<double>") {
        return "MPI_C_DOUBLE_COMPLEX";
    }
    else if(data_type == "std::complex<float>" || data_type == "complex<float>") {
        return "MPI_C_FLOAT_COMPLEX";
    }
    else if(data_type == "std::complex<long double>" || data_type == "complex<long double>") {
        return "MPI_C_LONG_DOUBLE_COMPLEX";
    }
    else {
        std::transform(data_type.begin(), data_type.end(), data_type.begin(),
                   [](unsigned char c){ return std::toupper(c); });
        return "MPI_" + data_type;
    }
}
const char *MPI_HEADER_Template = R"~~~(
#include <mpi.h>
int mpi_rank, mpi_size;

)~~~";


const char *MPI_INIT_Template = R"~~~(
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    // 仅允许 rank 0 保留标准输出
    if (mpi_rank != 0) {
        freopen("/dev/null", "w", stdout);
    }
    )~~~";

const char *PARA_CHECK_1D_Template = R"~~~(
    if ({{RES_DATA_SIZE}} % {{MPI_SIZE}} != 0) {
        if (mpi_rank == 0) std::cerr << "Error: output data size is not divisible by the number of MPI processes!" << std::endl;
        MPI_Finalize();
        std::exit(EXIT_SUCCESS);
    }
    )~~~";

string CodeGen_ParaCheck1D(string mpi_size, string res_data_size) {
    return templateString(PARA_CHECK_1D_Template,
    {
        {"{{MPI_SIZE}}", mpi_size},
        {"{{RES_DATA_SIZE}}", res_data_size}
    });
}

const char *PARA_CHECK_2D_Template = R"~~~(
    int n = static_cast<int>(std::sqrt({{MPI_SIZE}}));
    int N = {{RES_DATA_SIZE}};
    if (n * n != size || N % n != 0) {
    if (mpi_rank == 0) std::cerr << "Error: size must be a perfect square and N divisible by n!" << std::endl;
    MPI_Finalize();
    return -1;
    )~~~";

string CodeGen_ParaCheck2D(string mpi_size, string res_data_size) {
    return templateString(PARA_CHECK_2D_Template,
    {
        {"{{MPI_SIZE}}", mpi_size},
        {"{{RES_DATA_SIZE}}", res_data_size}
    });
}

const char *COMM_SPLIT_Template = R"~~~(
    MPI_Comm grid_comm;
    int dims[2] = {n, n};
    int periods[2] = {0, 0};
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
    int coords[2];
    MPI_Cart_coords(grid_comm, mpi_rank, 2, coords);
    int row = coords[0], col = coords[1];

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row, col, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col, row, &col_comm);
    )~~~";

const char *SCATTER_1D_split_Template = R"~~~(
    // 一维数据分发
    MPI_Datatype total_{{DATA_NAME}}_type, resized_total_{{DATA_NAME}}_type;
    MPI_Type_vector(1, {{RES_DATA_SIZE}}/{{MPI_SIZE}}+{{SPLIT_LENGTH}}-{{SPLIT_STEP}}, 1, {{MPI_TYPE}}, &total_{{DATA_NAME}}_type);
    MPI_Type_create_resized(total_{{DATA_NAME}}_type, 0, ({{RES_DATA_SIZE}}/{{MPI_SIZE}}) * sizeof({{DATA_TYPE}}), &resized_total_{{DATA_NAME}}_type);
    MPI_Type_commit(&resized_total_{{DATA_NAME}}_type);
    MPI_Scatter(total_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type,
    local_std_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type, 0, MPI_COMM_WORLD); 
    MPI_Type_free(&total_{{DATA_NAME}}_type);
    MPI_Type_free(&resized_total_{{DATA_NAME}}_type); 
    )~~~";

string CodeGen_Scatter1DSplit(string data_type, string data_name, string res_data_size, string mpi_size, 
    string split_length, string split_step) {
    return templateString(SCATTER_1D_split_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{SPLIT_LENGTH}}", split_length},
        {"{{SPLIT_STEP}}", split_step},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}

const char *SCATTER_1D_index_Template = R"~~~(
    // 一维数据分发
    MPI_Datatype total_{{DATA_NAME}}_type, resized_total_{{DATA_NAME}}_type;
    MPI_Type_vector(1, {{RES_DATA_SIZE}}/{{MPI_SIZE}}, 1, {{MPI_TYPE}}, &total_{{DATA_NAME}}_type);
    MPI_Type_create_resized(total_{{DATA_NAME}}_type, 0, ({{RES_DATA_SIZE}}/{{MPI_SIZE}}) * sizeof({{DATA_TYPE}}), &resized_total_{{DATA_NAME}}_type);
    MPI_Type_commit(&resized_total_{{DATA_NAME}}_type);
    MPI_Scatter(total_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type,
    local_std_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type, 0, MPI_COMM_WORLD); 
    MPI_Type_free(&total_{{DATA_NAME}}_type);
    MPI_Type_free(&resized_total_{{DATA_NAME}}_type); 
    )~~~";

string CodeGen_Scatter1DIndex(string data_type, string data_name, string res_data_size, string mpi_size) {
    return templateString(SCATTER_1D_index_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}
const char *SCATTER_2D_ROW_INDEX_Template = R"~~~(
    // 按行分发二维数据
    int size_total_{{DATA_NAME}} = total_{{DATA_NAME}}.size();
    MPI_Scatter(total_{{DATA_NAME}}.data(), size_total_{{DATA_NAME}}/{{MPI_SIZE}}, {{MPI_TYPE}},
                local_std_{{DATA_NAME}}.data(), size_total_{{DATA_NAME}}/{{MPI_SIZE}}, {{MPI_TYPE}}, 0, MPI_COMM_WORLD);
    )~~~";

string CodeGen_Scatter2D_row_index(string data_type, string data_name, string mpi_size) {
    return templateString(SCATTER_2D_ROW_INDEX_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}

const char *SCATTER_2D_ROW_SPLIT_Template = R"~~~(
    // 按行分发二维数据
    MPI_Datatype total_{{DATA_NAME}}_type, resized_total_{{DATA_NAME}}_type;
    MPI_Type_vector(1, local_col_{{DATA_NAME}}*local_row_{{DATA_NAME}}, 
        1, {{MPI_TYPE}}, &total_{{DATA_NAME}}_type);
    MPI_Type_create_resized(total_{{DATA_NAME}}_type, 0, 
        local_col_{{DATA_NAME}}*({{RES_DATA_SIZE}}/{{MPI_SIZE}}) * sizeof({{DATA_TYPE}}), &resized_total_{{DATA_NAME}}_type);
    MPI_Type_commit(&resized_total_{{DATA_NAME}}_type);
    MPI_Scatter(total_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type,
    local_std_{{DATA_NAME}}.data(), 1, resized_total_{{DATA_NAME}}_type, 0, MPI_COMM_WORLD); 
    MPI_Type_free(&total_{{DATA_NAME}}_type);
    MPI_Type_free(&resized_total_{{DATA_NAME}}_type);
    )~~~";

string CodeGen_Scatter2D_row_split(string data_type, string data_name, string res_data_size, string mpi_size,
    string split_length, string split_step) {
    return templateString(SCATTER_2D_ROW_SPLIT_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{SPLIT_LENGTH}}", split_length},
        {"{{SPLIT_STEP}}", split_step},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}

const char *SCATTER_2D_Template = R"~~~(
    // 分发行块
    if (col == 0) {
        MPI_Scatter(A.data(), N * a_cols, MPI_FLOAT,
                    local_A.data(), N * a_cols, MPI_FLOAT, 0, col_comm);
    }
    MPI_Bcast(local_A.data(), N * a_cols, MPI_FLOAT, 0, row_comm);
    // 分发列块
    if (row == 0) {
        MPI_Scatterv(B.data(), sendcounts, displsB, resized_col_type,
                    local_B.data(), block_size * N, MPI_FLOAT, 0, row_comm);
    }
    MPI_Bcast(local_B.data(), block_size * N, MPI_FLOAT, 0, col_comm);
    )~~~";

const char *GATHER_1D_Template = R"~~~(
    // 输出数据收集
    MPI_Gather(
        local_std_{{RES_DATA_NAME}}.data(), {{RES_DATA_SIZE}}/{{MPI_SIZE}}, {{MPI_TYPE}}, // 发送连续数据
        total_{{RES_DATA_NAME}}.data(), {{RES_DATA_SIZE}}/{{MPI_SIZE}}, {{MPI_TYPE}}, // 接收为调整后的块类型
        0, MPI_COMM_WORLD);
    )~~~";

string CodeGen_gather1D(string res_data_name, string res_data_size, string mpi_size, string data_type) {
    return templateString(GATHER_1D_Template,
    {
        {"{{RES_DATA_NAME}}", res_data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}

const char *GATHER_2D_ROW_Template = R"~~~(
    // 输出数据收集
    int size_local_std_{{RES_DATA_NAME}} = local_std_{{RES_DATA_NAME}}.size();
    MPI_Gather(
        local_std_{{RES_DATA_NAME}}.data(), size_local_std_{{RES_DATA_NAME}}, {{MPI_TYPE}}, // 发送连续数据
        total_{{RES_DATA_NAME}}.data(), size_local_std_{{RES_DATA_NAME}}, {{MPI_TYPE}}, // 接收为调整后的块类型
        0, MPI_COMM_WORLD);
    )~~~";

string CodeGen_gather2D_row(string res_data_name, string res_data_size, string mpi_size, string data_type) {
    return templateString(GATHER_2D_ROW_Template,
    {
        {"{{RES_DATA_NAME}}", res_data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{MPI_TYPE}}", CodeGen_mpi_datatype(data_type)}
    });
}
const char *MPI_FINISH_Template = R"~~~(
    MPI_Finalize();
    )~~~";

const char *DACPP_TO_STD_INDEX_Template = R"~~~(
    // DACPP类型数据转换为C++标准类型
    std::vector<{{DATA_TYPE}}> total_{{DATA_NAME}};
    {{DATA_NAME}}.tensor2Array(total_{{DATA_NAME}});
    std::vector<{{DATA_TYPE}}> local_std_{{DATA_NAME}}(total_{{DATA_NAME}}.size()/{{MPI_SIZE}});
    // std::vector<{{DATA_TYPE}}> local_std_{{DATA_NAME}}({{RES_DATA_SIZE}}/{{MPI_SIZE}});
    )~~~";

string CodeGen_dacpp2std_index(string data_type, string data_name, string res_data_size, string mpi_size) {
    return templateString(DACPP_TO_STD_INDEX_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size}
    });
}

const char *DACPP_TO_STD_SPLIT_Template = R"~~~(
    // DACPP类型数据转换为C++标准类型
    std::vector<{{DATA_TYPE}}> total_{{DATA_NAME}};
    {{DATA_NAME}}.tensor2Array(total_{{DATA_NAME}});
    std::vector<{{DATA_TYPE}}> local_std_{{DATA_NAME}}({{RES_DATA_SIZE}}/{{MPI_SIZE}}+{{SPLIT_LENGTH}}-{{SPLIT_STEP}});
    )~~~";

string CodeGen_dacpp2std_split(string data_type, string data_name, string res_data_size, string mpi_size, 
    string split_length, string split_step) {
    return templateString(DACPP_TO_STD_SPLIT_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{SPLIT_LENGTH}}", split_length},
        {"{{SPLIT_STEP}}", split_step}
    });
}

const char *DACPP_TO_STD_SPLIT_2D_Template = R"~~~(
    // DACPP类型数据转换为C++标准类型
    int local_col_{{DATA_NAME}} = info_{{DATA_NAME}}.dimLength[1];
    int local_row_{{DATA_NAME}} = {{RES_DATA_SIZE}}/{{MPI_SIZE}}+{{SPLIT_LENGTH}}-{{SPLIT_STEP}};
    std::vector<{{DATA_TYPE}}> total_{{DATA_NAME}};
    {{DATA_NAME}}.tensor2Array(total_{{DATA_NAME}});
    std::vector<{{DATA_TYPE}}> local_std_{{DATA_NAME}}(local_col_{{DATA_NAME}}*local_row_{{DATA_NAME}});
    )~~~";

string CodeGen_dacpp2std_split2d(string data_type, string data_name, string res_data_size, string mpi_size, 
    string split_length, string split_step) {
    return templateString(DACPP_TO_STD_SPLIT_2D_Template,
    {
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_NAME}}", data_name},
        {"{{RES_DATA_SIZE}}", res_data_size},
        {"{{MPI_SIZE}}", mpi_size},
        {"{{SPLIT_LENGTH}}", split_length},
        {"{{SPLIT_STEP}}", split_step}
    });
}

const char *STD_TO_DACPP_Template = R"~~~(
    // 数据由C++标准类型转换为DACPP
    dacpp::Tensor<{{DATA_TYPE}}, {{DATA_DIM}}> local_dacpp_{{DATA_NAME}}(local_std_{{DATA_NAME}});
    )~~~";

string CodeGen_std2dacpp(string data_type, string data_dim, string data_name) {
    return templateString(STD_TO_DACPP_Template,
    {
        {"{{DATA_NAME}}", data_name},
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_DIM}}", data_dim}
    });
}

const char *STD_TO_DACPP_2D_Template = R"~~~(
    // 数据由C++标准类型转换为DACPP
    dacpp::Tensor<{{DATA_TYPE}}, 2> local_dacpp_{{DATA_NAME}}(
    {info_{{DATA_NAME}}.dimLength[0]/{{MPI_SIZE}}, info_{{DATA_NAME}}.dimLength[1]}, local_std_{{DATA_NAME}});
    )~~~";

string CodeGen_std2dacpp_2d(string data_type, string mpi_size, string data_name) {
    return templateString(STD_TO_DACPP_2D_Template,
    {
        {"{{DATA_NAME}}", data_name},
        {"{{DATA_TYPE}}", data_type},
        {"{{MPI_SIZE}}", mpi_size}
    });
}
const char *STD_TO_DACPP_2D_SPLIT_Template = R"~~~(
    // 数据由C++标准类型转换为DACPP
    dacpp::Tensor<{{DATA_TYPE}}, 2> local_dacpp_{{DATA_NAME}}(
    {local_row_{{DATA_NAME}}, local_col_{{DATA_NAME}}}, local_std_{{DATA_NAME}});
    )~~~";

string CodeGen_std2dacpp_2dsplit(string data_type, string mpi_size, string data_name) {
    return templateString(STD_TO_DACPP_2D_SPLIT_Template,
    {
        {"{{DATA_NAME}}", data_name},
        {"{{DATA_TYPE}}", data_type},
        {"{{MPI_SIZE}}", mpi_size}
    });
}

const char *LOCAL_DACPP_KEEP_Template = R"~~~(
    // 被保形算子作用的数据转换为局部数据
    dacpp::Tensor<{{DATA_TYPE}}, {{DATA_DIM}}> local_dacpp_{{DATA_NAME}}({{DATA_NAME}});
    )~~~";

string CodeGen_local_dacpp_keep(string data_type, string data_dim, string data_name) {
    return templateString(LOCAL_DACPP_KEEP_Template,
    {
        {"{{DATA_NAME}}", data_name},
        {"{{DATA_TYPE}}", data_type},
        {"{{DATA_DIM}}", data_dim}
    });
}

const char *LOCAL_DACPP_2DKEEP_Template = R"~~~(
    // 被保形算子作用的数据转换为局部数据
    dacpp::Tensor<{{DATA_TYPE}}, 2> local_dacpp_{{DATA_NAME}}(
    {info_{{DATA_NAME}}.dimLength[0], info_{{DATA_NAME}}.dimLength[1]}, total_{{DATA_NAME}});
    )~~~";

string CodeGen_local_dacpp_2dkeep(string data_type, string data_name) {
    return templateString(LOCAL_DACPP_2DKEEP_Template,
    {
        {"{{DATA_NAME}}", data_name},
        {"{{DATA_TYPE}}", data_type}
    });
}

const char *RES_DATA_SIZE_Template = R"~~~(
    // 计算输出数据的大小
    int res_data_size = info_{{DATA_NAME}}.dimLength[0];
    )~~~";

string CodeGen_res_data_size(string data_name) {
    return templateString(RES_DATA_SIZE_Template,
    {
        {"{{DATA_NAME}}", data_name}
    }) + CodeGen_ParaCheck1D("mpi_size", "res_data_size");
}

const char *DACPP_TO_STD_GATHER_Template = R"~~~(
    local_dacpp_{{DATA_NAME}}.tensor2Array(local_std_{{DATA_NAME}});
    )~~~";

string CodeGen_dacpp2std_gather(string data_name) {
    return templateString(DACPP_TO_STD_GATHER_Template,
    {
        {"{{DATA_NAME}}", data_name}
    });
}

const char *STD_TO_DACPP_GATHER_Template = R"~~~(
    //输出数据从C++标准类型转换为DACPP
    if(mpi_rank == 0) {
        {{DATA_NAME}}.array2Tensor(total_{{DATA_NAME}});
    }
    )~~~";

string CodeGen_std2dacpp_gather(string data_name) {
    return templateString(STD_TO_DACPP_GATHER_Template,
    {
        {"{{DATA_NAME}}", data_name}
    });
}
}