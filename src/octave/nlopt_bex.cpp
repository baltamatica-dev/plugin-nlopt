/* SPDX-License-Identifier: MIT */
#include "nlopt_bex.hpp"


/* workaround */
extern double bxGetScalar(const bxArray* ba) {
    return *bxGetDoubles(ba);
}

bool bxIsNumeric(const bxArray* ba) {
    return bxIsInt32(ba) || bxIsInt64(ba)
        || bxIsRealSingle(ba) || bxIsRealDouble(ba);
}

int mexCallMATLAB(int nlhs, bxArray *plhs[], int nrhs, bxArray *prhs[], const char *functionName) {
	return 1;
}


const char* nlopt_version_help = R"(
nlopt 绑定

    nlopt 插件
    Github:     https://github.com/baltamatica-dev/
    LICENSE:    MIT license
    Copyright (c) 2022 Chengyu HAN

版本信息
    bex SDK: 2.2.1
    stevengj/nlopt: (master)
        Github:     https://github.com/stevengj/nlopt
        LICENSE:    GNU LGPL && MIT
        Copyright (c) 2022 Steven G. Johnson
)"; /* nlopt_version_help */
BALTAM_PLUGIN_FCN(nlopt_version) {
    bxPrintf(nlopt_version_help);
} /* nlopt_version */


/**
 * @brief [可选] 插件初始化函数.
 *
 * bxPluginInit 由 load_plugin(name, args...) 调用
 * 用于进行一些初始化工作
 *
 * @param nInit
 * @param pInit[]
 */
int bxPluginInit(int nInit, const bxArray* pInit[]) {
    return 0;
} /* bxPluginInit */

/**
 * @brief [可选] 插件终止时清理函数.
 *
 * bxPluginFini 由 unload_plugin() 调用
 * 用于进行一些清理工作
 */
int bxPluginFini() {
    return 0;
} /* bxPluginFini */

/**
 * @brief 【必选】 列出插件提供的函数.
 *
 * bxPluginFunctions 返回 指向函数列表的指针.
 */
bexfun_info_t * bxPluginFunctions() {
    // 已定义的插件函数个数
    constexpr size_t TOTAL_PLUGIN_FUNCTIONS = 1;
    bexfun_info_t* func_list_dyn = new bexfun_info_t[TOTAL_PLUGIN_FUNCTIONS + 1];

    size_t i = 0;
    func_list_dyn[i].name = "nlopt_version";
    func_list_dyn[i].ptr  = nlopt_version;
    func_list_dyn[i].help = nlopt_version_help;

    // 最后一个元素, `name` 字段必须为空字符串 `""`
    i++;
    func_list_dyn[i].name = "";
    func_list_dyn[i].ptr  = nullptr;
    func_list_dyn[i].help = nullptr;

    assert((TOTAL_PLUGIN_FUNCTIONS == i));
    return func_list_dyn;
} /* bxPluginFunctions */
