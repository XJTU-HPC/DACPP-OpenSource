#ifndef MPI_TEMPLATE_H
#define MPI_TEMPLATE_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>
#include"dacInfo.h"

using std::string;

namespace MPI_TEMPLATE {
    extern const char *MPI_HEADER_Template;
    extern const char *MPI_INIT_Template;
    extern const char *PARA_CHECK_1D_Template;
    extern const char *PARA_CHECK_2D_Template;
    extern const char *COMM_SPLIT_Template;
    extern const char *SCATTER_1D_split_Template;
    extern const char *SCATTER_1D_index_Template;
    extern const char *SCATTER_2D_ROW_INDEX_Template;
    extern const char *SCATTER_2D_ROW_SPLIT_Template;
    extern const char *SCATTER_2D_Template;
    extern const char *GATHER_1D_Template;
    extern const char *GATHER_2D_Template;
    extern const char *GATHER_2D_ROW_Template;
    extern const char *MPI_FINISH_Template;
    extern const char *DACPP_TO_STD_INDEX_Template;
    extern const char *DACPP_TO_STD_SPLIT_Template;
    extern const char *DACPP_TO_STD_SPLIT_2D_Template;
    extern const char *STD_TO_DACPP_Template;
    extern const char *LOCAL_DACPP_KEEP_Template;
    extern const char *RES_DATA_SIZE_Template;
    extern const char *DACPP_TO_STD_GATHER_Template;
    extern const char *STD_TO_DACPP_GATHER_Template;
    extern const char *LOCAL_DACPP_2DKEEP_Template;
    extern const char *STD_TO_DACPP_2D_Template;
    extern const char *STD_TO_DACPP_2D_SPLIT_Template;

    void replaceTextInString(std::string& text, const std::string &find, const std::string &replace);

    std::string templateString(std::string templ, std::vector<std::pair<std::string, std::string>> replacements);

    string CodeGen_mpi_datatype(string data_type);

    string CodeGen_ParaCheck1D(string mpi_size, string res_data_size);

    string CodeGen_ParaCheck2D(string mpi_size, string res_data_size);

    string CodeGen_Scatter1DSplit(string data_type, string data_name, string res_data_size, string mpi_size, string split_length, string split_step);

    string CodeGen_Scatter1DIndex(string data_type, string data_name, string res_data_size, string mpi_size);

    string CodeGen_Scatter2D_row_index(string data_type, string data_name, string mpi_size);

    string CodeGen_Scatter2D_row_split(string data_type, string data_name, string res_data_size, string mpi_size, string split_length, string split_step);

    string CodeGen_dacpp2std_index(string data_type, string data_name, string res_data_size, string mpi_size);

    string CodeGen_dacpp2std_split(string data_type, string data_name, string res_data_size, string mpi_size, string split_length, string split_step);

    string CodeGen_dacpp2std_split2d(string data_type, string data_name, string res_data_size, string mpi_size, string split_length, string split_step);
    
    string CodeGen_std2dacpp(string data_type, string data_dim, string data_name);
    
    string CodeGen_local_dacpp_keep(string data_type, string data_dim, string data_name);

    string CodeGen_res_data_size(string res_data_name);

    string CodeGen_gather1D(string res_data_name, string res_data_size, string mpi_size, string data_type);

    string CodeGen_gather2D_row(string res_data_name, string res_data_size, string mpi_size, string data_type);

    string CodeGen_dacpp2std_gather(string data_name);

    string CodeGen_std2dacpp_gather(string data_name);

    string CodeGen_local_dacpp_2dkeep(string data_type, string data_name);

    string CodeGen_std2dacpp_2d(string data_type, string mpi_size, string data_name);

    string CodeGen_std2dacpp_2dsplit(string data_type, string mpi_size, string data_name);
}

#endif