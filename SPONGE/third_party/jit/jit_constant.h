value += R"JIT(
//存储各种常数
//圆周率
#define CONSTANT_Pi 3.1415926535897932f
//自然对数的底
#define CONSTANT_e 2.7182818284590452f
// 玻尔兹曼常量
// 单位见下一行
// kB unit: kcal mol^-1 K^-1
// 使用 kcal 作为能量单位
// kB = 8.31441 / 4.18407 / 1000
#define CONSTANT_kB 0.00198716f
// 程序中使用的单位时间与物理时间的换算
// 1 / 20.455 * dt = 1 ps
#define CONSTANT_TIME_CONVERTION 20.455f
// 程序压强与物理压强的换算
// 程序压强单位为 kcal 每 mol 每 A^3
// 物理压强单位为 bar
// 计算关系见下一行
// physical_pressure = program_pressure * CONSTANT_PRES_CONVERTION
#define CONSTANT_PRES_CONVERTION 6.946827162543585e4f
// 反向计算关系见下一行
// program_pressure = physical_pressure * CONSTANT_PRES_CONVERTION_INVERSE
#define CONSTANT_PRES_CONVERTION_INVERSE 0.00001439506089041446f
//角度制到弧度制的转换系数
#define CONSTANT_RAD_TO_DEG 57.2957795f
//弧度制到角度制的转换系数
#define CONSTANT_DEG_TO_RAD 0.0174532925f

#define CHAR_LENGTH_MAX 512
#define FULL_MASK 0xffffffff

//用于计算边界循环所定义的结构体
struct UNSIGNED_INT_VECTOR
{
    unsigned int uint_x;
    unsigned int uint_y;
    unsigned int uint_z;
};

//用于计算边界循环或者一些三维数组大小所定义的结构体
struct INT_VECTOR
{
    int int_x;
    int int_y;
    int int_z;
};

__device__ __forceinline__ void Warp_Sum_To(float* y, float& x, int delta = 32);
__host__ __device__ __forceinline__ float BSpline_4_1(float x);
__host__ __device__ __forceinline__ float BSpline_4_2(float x);
__host__ __device__ __forceinline__ float BSpline_4_3(float x);
__host__ __device__ __forceinline__ float BSpline_4_4(float x);
__host__ __device__ __forceinline__ float dBSpline_4_1(float x);
__host__ __device__ __forceinline__ float dBSpline_4_2(float x);
__host__ __device__ __forceinline__ float dBSpline_4_3(float x);
__host__ __device__ __forceinline__ float dBSpline_4_4(float x);
//用于存储各种三维float矢量而定义的结构体
)JIT";
