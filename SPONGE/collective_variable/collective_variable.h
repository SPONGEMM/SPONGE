#pragma once

#include <functional>

#include "../common.h"
#include "../control.h"

// 传给CV需要计算的值，分别相当于能量、力和维里的需求
enum COLLECTIVE_VARIABLE_NEED
{
    CV_NEED_NONE = 0,
    // 需要GPU上的CV值
    CV_NEED_GPU_VALUE = 1,
    // 需要GPU上的对坐标的导数
    CV_NEED_CRD_GRADS = 2,
    // 需要CPU上的CV值
    CV_NEED_CPU_VALUE = 4,
    // 需要GPU上的对盒子边长的导数
    CV_NEED_VIRIAL = 8,
    // 需要所有的
    CV_NEED_ALL = 15,
};

// 提前声明CV控制器
struct COLLECTIVE_VARIABLE_CONTROLLER;

// CV计算的原型结构体
struct COLLECTIVE_VARIABLE_PROTOTYPE
{
    char module_name[CHAR_LENGTH_MAX];
    int last_modify_date = 20260216;

    // CV类型的名字
    char type_name[CHAR_LENGTH_MAX];

    // CV的cuda流，方便CV并行计算
    deviceStream_t device_stream;

    // CV本身的值（CPU，GPU）
    float value, *d_value;
    COLLECTIVE_VARIABLE_CONTROLLER* manager = NULL;
    bool supports_open_boundary = true;
    // GPU上的对坐标的导数（求力用）和对盒子边长的导数（求维里用）
    VECTOR* crd_grads = NULL;
    LTMatrix3* virial = NULL;
    // 记录上次更新某个需求的step，避免在同一步内重复计算某个CV很多遍
    std::map<COLLECTIVE_VARIABLE_NEED, int> last_update_step;
    // 没有virtual的不是接口，只是辅助初始化、记录更新步的函数，不会被默认调用
    // 判断此步是否更新过，避免重新计算
    int Check_Whether_Computed_At_This_Step(int step, int need);
    // 快慢的区别在于是否需要print
    // 记录快速计算CV的更新步（同时计算CV值、力和维里）
    void Record_Update_Step_Of_Fast_Computing_CV(int step, int need);
    // 记录慢速计算CV的更新步（分别计算CV值、力和维里）
    void Record_Update_Step_Of_Slow_Computing_CV(int step, int need);
    // 公用的初始化
    void Super_Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                       int atom_numbers, const char* module_name);
    void Validate_Boundary(Boundary boundary) const;
    // 带virtual的是接口函数，必须各自实现
    // 子类自身的初始化
    virtual void Initial(COLLECTIVE_VARIABLE_CONTROLLER* manager,
                         int atom_numbers, const char* module_name) = 0;
    // 子类计算CV的具体实现细节
    virtual void Compute(int atom_numbers, VECTOR* crd, Boundary boundary,
                         int need, int step) = 0;
};

// 所有CV的列表
typedef std::vector<COLLECTIVE_VARIABLE_PROTOTYPE*> CV_LIST;
// 根据名称调用构造CV结构体的函数原型
typedef std::map<std::string,
                 std::function<COLLECTIVE_VARIABLE_PROTOTYPE*(
                     COLLECTIVE_VARIABLE_CONTROLLER*, const char*)>>
    CV_MAP_TYPE;
// CV名字到CV原型结构体指针的映射
typedef std::map<std::string, COLLECTIVE_VARIABLE_PROTOTYPE*> CV_INSTANCE_TYPE;

// 所有CV的种类和初始化一个该CV的函数的映射
extern CV_MAP_TYPE* CV_MAP;
// 所有CV的实例的映射
extern CV_INSTANCE_TYPE* CV_INSTANCE_MAP;

// CV控制器
struct COLLECTIVE_VARIABLE_CONTROLLER : public CONTROLLER
{
    // 系统的原子数，初始化时传入
    int atom_numbers;
    // 系统的controller，初始化CV时传入
    CONTROLLER* controller = NULL;
    // 自身初始化
    void Initial(CONTROLLER* controller,
                 int* no_direct_interaction_virtual_atom_numbers);
    // 读取cv_in_file里的信息
    void Commands_From_In_File(CONTROLLER* controller);
    // 检查cv_in_file里的信息是否用完
    void Input_Check();
    // 以上的定义方一般不用

    // 打印的CV列表
    CV_LIST print_cv_list;
    // 打印初始化
    void Print_Initial();
    // 每步打印CV信息
    void Step_Print();
    // 为打印计算
    void Compute_CV_For_Print(int atom_numbers, VECTOR* crd, Boundary boundary,
                              int steps, int interval, bool print_zeroth_frame);
    // 通过CV的名字获取一个新建立的CV结构体
    COLLECTIVE_VARIABLE_PROTOTYPE* get_CV(const char* cv_name);

    // cv定义的虚原子到原子序号的映射
    CheckMap cv_vatom_name;
    // 原子序号的映射到cv定义的虚原子的映射
    std::map<int, std::string> cv_vatom_index;
    // Native protocol metadata retained for typed bias consumers.
    std::map<std::string, std::vector<float>> protocol_cv_period;
    std::map<std::string, std::vector<float>> protocol_cv_sigma;
    std::map<std::string, std::vector<float>> protocol_cv_reference;
    std::map<std::string, std::vector<int>> protocol_virtual_atom_indices;
    std::map<std::string, std::vector<float>> protocol_virtual_atom_weight;
    // Runtime still consumes the `meta` section; this retains the stable
    // bundled object identity used by diagnostic and restart paths.
    std::string protocol_metadynamics_name;
    // 初始化时获得使用的所有CV
    // 传入参数：
    //     name: 需要找寻的CV的任务名
    //     N: 需要的CV的个数（正数表示有且只有N个，非正数表示至少N的绝对值个)
    //     verbose_level: 输出信息的详细程度
    //     layout: 打印信息的空格数*4
    // 输出：一个CV原型的vector（cv_list）
    CV_LIST Ask_For_CV(const char* name, int N, float verbose_level = 0,
                       int layout = 1);

    // 初始化时获得使用的float参数
    // 传入参数：
    //    name: 需要找寻参数的任务名
    //    parameter name: 需要寻找的参数名
    //    N: 需要的参数的个数
    //    layout: 打印信息的空格数*4
    //    raise_error_when_missing: 当缺失参数时是否报错
    //    default_value: 当缺失参数时设置的默认值
    //    verbose_level: 输出信息的详细程度
    //    unit: 参数的单位，只做显示用
    // 输出：一个对应类型的CPU指针
    float* Ask_For_Float_Parameter(const char* name, const char* parameter_name,
                                   int N = 1, int layout = 1,
                                   bool raise_error_when_missing = true,
                                   float default_value = 0,
                                   float verbose_level = 0,
                                   const char* unit = NULL);
    // 初始化时获得使用的int参数
    // 传入参数：
    //    name: 需要找寻参数的任务名
    //    parameter name: 需要寻找的参数名
    //    N: 需要的参数的个数
    //    layout: 打印信息的空格数*4
    //    raise_error_when_missing: 当缺失参数时是否报错
    //    default_value: 当缺失参数时设置的默认值
    //    verbose_level: 输出信息的详细程度
    //    unit: 参数的单位，只做显示用
    // 输出：一个对应类型的CPU指针
    int* Ask_For_Int_Parameter(const char* name, const char* parameter_name,
                               int N = 1, int layout = 1,
                               bool raise_error_when_missing = true,
                               int default_value = 0, float verbose_level = 0,
                               const char* unit = NULL);
    std::vector<std::string> Ask_For_String_Parameter(
        const char* name, const char* parameter_name, int N = 1, int layout = 1,
        bool raise_error_when_missing = true, const char* default_value = 0,
        float verbose_level = 0, const char* unit = NULL);
    // 初始化时获得不定长的参数
    std::vector<int> Ask_For_Indefinite_Length_Int_Parameter(
        const char* name, const char* parameter_name);
    // 初始化时获得不定长的float参数
    std::vector<float> Ask_For_Indefinite_Length_Float_Parameter(
        const char* name, const char* parameter_name);
};

// 将新的CV结构体信息传入CV控制器的cv_map中
// 相当于提供字符串（第2个参数）转结构体（第1个参数）的功能。第三个参数只是为了防止临时变量重名，只要不重复即可
#define REGISTER_CV_STRUCTURE(STRUCTURE, cv_type, count)                      \
    static struct REGISTER_##STRUCTURE##count                                 \
    {                                                                         \
        REGISTER_##STRUCTURE##count()                                         \
        {                                                                     \
            auto f =                                                          \
                [](COLLECTIVE_VARIABLE_CONTROLLER* manager, const char* name) \
            {                                                                 \
                COLLECTIVE_VARIABLE_PROTOTYPE* cv = new STRUCTURE;            \
                strcpy(cv->type_name, cv_type);                               \
                cv->Initial(manager, manager->atom_numbers, name);            \
                return cv;                                                    \
            };                                                                \
            CV_MAP[0][cv_type] = f;                                           \
        }                                                                     \
    } register_##STRUCTURE##count;
