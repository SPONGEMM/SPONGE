/*
nvrtc for CUDA backend,
directly compiling for CPU backend
*/
#pragma once

#include "../../common.h"
#include "../../control.h"

namespace sponge_jit_detail
{
inline const std::string& Embedded_Common_Header()
{
    static const std::string header = []()
    {
        std::string value;
        value.reserve(65536);
#include "jit.h"
        return value;
    }();
    return header;
}
}  // namespace sponge_jit_detail

#ifdef USE_CUDA
struct JIT_Function
{
   private:
    CUfunction function;

   public:
    std::string error_reason;

    void Compile(std::string source)
    {
        std::string common_h = sponge_jit_detail::Embedded_Common_Header();
        const char* headers[1] = {common_h.c_str()};
        const char* header_names[1] = {"common.h"};
        nvrtcProgram prog;
        nvrtcCreateProgram(&prog, source.c_str(), NULL, 1, headers,
                           header_names);

        std::string arch = "-arch sm_";
        deviceProp prop;

        getDeviceProperties(&prop, 0);
        int runtime_arch_bin = prop.major * 10 + prop.minor;
        arch += std::to_string(runtime_arch_bin);
        const char* opts[] = {"--use_fast_math", arch.c_str()};
        if (nvrtcCompileProgram(prog, 1, opts) != NVRTC_SUCCESS)
        {
            size_t logSize;
            nvrtcGetProgramLogSize(prog, &logSize);
            char* log_ = new char[logSize];
            nvrtcGetProgramLog(prog, log_);
            error_reason = log_;
            delete log_;
            return;
        }
        size_t pos1 = source.find("extern");
        size_t pos2 =
            source.find(string_format("%q%C%q%", {{"q", {'"'}}}), pos1);
        if (pos2 == source.npos)
        {
            error_reason =
                R"(extern "C" should be placed in front of the function name)";
            return;
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_last_of(" ", pos1);
        std::string name = string_strip(source.substr(pos2, pos1 - pos2));
        if (name == "__launch_bounds__")
        {
            pos1 = source.find_first_of("(", pos1 + 1);
            pos2 = source.find_last_of(" ", pos1);
            name = string_strip(source.substr(pos2, pos1 - pos2));
        }
        size_t ptxSize;
        nvrtcGetPTXSize(prog, &ptxSize);
        char* ptx = new char[ptxSize];
        nvrtcGetPTX(prog, ptx);
        CUmodule module;
        if (cuModuleLoadDataEx(&module, ptx, 0, 0, 0) != CUDA_SUCCESS)
        {
            error_reason = string_format(
                "Fail to load the module from PTX for %f%", {{"f", name}});
            return;
        }
        if (cuModuleGetFunction(&function, module, name.c_str()) !=
            CUDA_SUCCESS)
        {
            error_reason = string_format(
                "Fail to get the name from the module for %f%", {{"f", name}});
            return;
        }
        delete ptx;

        return;
    }

    void operator()(dim3 blocks, dim3 threads, cudaStream_t stream,
                    unsigned int shared_memory_size, std::vector<void*> args)
    {
        CUresult result = cuLaunchKernel(
            function, blocks.x, blocks.y, blocks.z, threads.x, threads.y,
            threads.z, shared_memory_size, stream, &args[0], NULL);
        if (result != CUDA_SUCCESS)
        {
            const char* name;
            const char* string;
            cuGetErrorName(result, &name);
            cuGetErrorString(result, &string);
            error_reason = string_format("Kernel Launch Error %NAME%: %STRING%",
                                         {{"NAME", name}, {"STRING", string}});
            printf("Kernel Launch Error %s: %s\n", name, string);
        }
    }
};
#else

#include <clang/Basic/Diagnostic.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/CodeGen/CodeGenAction.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Lex/PreprocessorOptions.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/TargetParser/Triple.h>

#if defined(__linux__)
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <vector>

struct JIT_Function
{
   private:
    void (*function)(void** args) = nullptr;
    std::unique_ptr<llvm::orc::LLJIT> jit_engine;

    static void Initialize_ORC_Runtime()
    {
        static std::once_flag init_once;
        std::call_once(init_once,
                       []()
                       {
                           llvm::InitializeNativeTarget();
                           llvm::InitializeNativeTargetAsmPrinter();
                       });
    }

#if defined(__linux__)
    static void Push_Unique_Candidate(std::vector<std::string>& candidates,
                                      const std::string& candidate)
    {
        if (candidate.empty())
        {
            return;
        }
        if (std::find(candidates.begin(), candidates.end(), candidate) ==
            candidates.end())
        {
            candidates.push_back(candidate);
        }
    }

    static std::vector<std::string> Build_Runtime_Candidates(
        std::initializer_list<const char*> lib_names)
    {
        std::vector<std::string> candidates;
        const char* conda_prefix = std::getenv("CONDA_PREFIX");
        if (conda_prefix != nullptr && conda_prefix[0] != '\0')
        {
            std::string lib_dir = std::string(conda_prefix) + "/lib/";
            for (const char* lib_name : lib_names)
            {
                Push_Unique_Candidate(candidates, lib_dir + lib_name);
            }
        }
        for (const char* lib_name : lib_names)
        {
            Push_Unique_Candidate(candidates, lib_name);
        }
        return candidates;
    }

    static std::vector<std::string> Build_OpenMP_Runtime_Candidates()
    {
        return Build_Runtime_Candidates(
            {"libomp.so", "libomp.so.5", "libiomp5.so", "libgomp.so.1"});
    }

    static std::vector<std::string> Build_Libatomic_Candidates()
    {
        return Build_Runtime_Candidates({"libatomic.so.1"});
    }

    static std::string Join_Candidates(
        const std::vector<std::string>& candidates)
    {
        std::string joined;
        for (const auto& candidate : candidates)
        {
            if (!joined.empty())
            {
                joined += ", ";
            }
            joined += candidate;
        }
        return joined;
    }

    static bool Try_Load_Runtime_Library(
        const std::vector<std::string>& candidates, void** loaded_handle,
        std::string* loaded_path, std::string* load_error,
        const char* required_symbol = nullptr)
    {
        if (loaded_handle != nullptr)
        {
            *loaded_handle = nullptr;
        }
        if (loaded_path != nullptr)
        {
            loaded_path->clear();
        }
        for (const auto& candidate : candidates)
        {
            dlerror();
            void* handle = dlopen(candidate.c_str(), RTLD_LAZY | RTLD_GLOBAL);
            if (handle == nullptr)
            {
                continue;
            }
            if (required_symbol != nullptr &&
                dlsym(handle, required_symbol) == nullptr)
            {
                dlclose(handle);
                continue;
            }
            if (loaded_handle != nullptr)
            {
                *loaded_handle = handle;
            }
            if (loaded_path != nullptr)
            {
                *loaded_path = candidate;
            }
            if (load_error != nullptr)
            {
                load_error->clear();
            }
            return true;
        }
        if (load_error != nullptr)
        {
            const char* err = dlerror();
            *load_error = "Fail to load runtime library (tried: " +
                          Join_Candidates(candidates) + ")";
            if (required_symbol != nullptr)
            {
                *load_error += ", required symbol: ";
                *load_error += required_symbol;
            }
            *load_error += ": ";
            *load_error += (err != nullptr) ? err : "unknown error";
        }
        return false;
    }
#endif

    static bool Ensure_OpenMP_Runtime_Loaded(std::string& load_error)
    {
#if defined(__linux__)
        static std::once_flag load_once;
        static bool load_success = false;
        static std::string load_failure_reason;
        static void* openmp_handle = nullptr;
        static std::string openmp_runtime_path;
        std::call_once(
            load_once,
            []()
            {
                const auto openmp_candidates =
                    Build_OpenMP_Runtime_Candidates();
                if (!Try_Load_Runtime_Library(
                        openmp_candidates, &openmp_handle, &openmp_runtime_path,
                        &load_failure_reason, "__kmpc_fork_call"))
                {
                    return;
                }
                load_success = true;

                const auto libatomic_candidates = Build_Libatomic_Candidates();
                void* libatomic_handle = nullptr;
                std::string libatomic_path;
                std::string ignored_error;
                Try_Load_Runtime_Library(libatomic_candidates,
                                         &libatomic_handle, &libatomic_path,
                                         &ignored_error);
                if (libatomic_handle != nullptr)
                {
                    (void)libatomic_handle;
                }
            });
        if (!load_success)
        {
            load_error = load_failure_reason;
            return false;
        }
#else
        (void)load_error;
#endif
        load_error.clear();
        return true;
    }

    static const std::string& InMemory_Common_Header()
    {
        static const std::string header = []()
        {
            std::string value = R"CPRE(
#ifndef __CUDACC__
#define __device__
#define __host__
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(THREADS)
struct sponge_jit_dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};
static sponge_jit_dim3 threadIdx = {0u, 0u, 0u};
__forceinline__ unsigned int __ballot_sync(unsigned int, int pred)
{
    return pred ? 0xffffffffu : 0u;
}
template <typename T>
__forceinline__ T __shfl_down_sync(unsigned int, T value, int)
{
    return value;
}
__forceinline__ float atomicAdd(float* x, float y);
__forceinline__ int atomicAdd(int* x, int y);
__forceinline__ double atomicAdd(double* x, double y);
__forceinline__ float rnorm3df(float x, float y, float z);
extern "C" float powf(float x, float y);
extern "C" float expf(float x);
extern "C" float erfcf(float x);
extern "C" float logf(float x);
extern "C" float sqrtf(float x);
extern "C" float cbrtf(float x);
extern "C" float cosf(float x);
extern "C" float sinf(float x);
extern "C" float tanf(float x);
extern "C" float acosf(float x);
extern "C" float asinf(float x);
extern "C" float atanf(float x);
extern "C" float atan2f(float y, float x);
extern "C" float fabsf(float x);
extern "C" float copysignf(float x, float y);
extern "C" float fmaxf(float x, float y);
extern "C" float fminf(float x, float y);
extern "C" float floorf(float x);
#ifndef NULL
#define NULL 0
#endif
#ifndef warpSize
#define warpSize 0
#endif
#endif
)CPRE";
            value += sponge_jit_detail::Embedded_Common_Header();
            return value;
        }();
        return header;
    }

    bool Build_Module_From_Source(const std::string& source,
                                  llvm::LLVMContext& llvm_context,
                                  std::unique_ptr<llvm::Module>& output_module)
    {
        constexpr const char* kInputFile = "sponge_jit_runtime_input.cpp";
        constexpr const char* kCommonHeader = "#include \"common.h\"";
        std::string source_with_header = source;
        const std::string& common_header = InMemory_Common_Header();
        const auto include_pos = source_with_header.find(kCommonHeader);
        if (include_pos == std::string::npos)
        {
            source_with_header = common_header + "\n" + source_with_header;
        }
        else
        {
            source_with_header.replace(
                include_pos, std::string(kCommonHeader).size(), common_header);
        }

        std::vector<std::string> arg_storage = {
            "-xc++", "-std=c++17",  "-fopenmp", "-O3",
            "-w",    "-ffast-math", "-DUSE_CPU"};
        arg_storage.push_back(kInputFile);

        std::vector<const char*> arg_ptrs;
        arg_ptrs.reserve(arg_storage.size());
        for (const auto& arg : arg_storage)
        {
            arg_ptrs.push_back(arg.c_str());
        }

        std::string diag_log;
        llvm::raw_string_ostream diag_stream(diag_log);
        auto invocation = std::make_shared<clang::CompilerInvocation>();
        clang::CompilerInstance compiler(invocation);
        auto diag_client = std::make_unique<clang::TextDiagnosticPrinter>(
            diag_stream, invocation->getDiagnosticOpts());
        compiler.createDiagnostics(diag_client.release(), true);
        if (!compiler.hasDiagnostics())
        {
            error_reason =
                "Fail to initialize Clang diagnostics for JIT source";
            return false;
        }

        bool parsed = clang::CompilerInvocation::CreateFromArgs(
            *invocation, arg_ptrs, compiler.getDiagnostics(), "sponge-jit");
        diag_stream.flush();
        if (!parsed || compiler.getDiagnostics().hasErrorOccurred())
        {
            error_reason = "Fail to create Clang invocation for JIT source";
            if (!diag_log.empty())
            {
                error_reason += ":\n";
                error_reason += diag_log;
            }
            return false;
        }

        auto& target_options = compiler.getInvocation().getTargetOpts();
        if (target_options.Triple.empty())
        {
            target_options.Triple = llvm::sys::getDefaultTargetTriple();
        }
        clang::TargetInfo* target_info = clang::TargetInfo::CreateTargetInfo(
            compiler.getDiagnostics(), target_options);
        if (target_info == nullptr)
        {
            error_reason = "Fail to create Clang target info for JIT source";
            return false;
        }
        compiler.setTarget(target_info);
        compiler.createFileManager();
        compiler.createSourceManager();
        compiler.getPreprocessorOpts().RetainRemappedFileBuffers = true;
        compiler.getPreprocessorOpts().addRemappedFile(
            kInputFile,
            llvm::MemoryBuffer::getMemBufferCopy(source_with_header, kInputFile)
                .release());

        clang::EmitLLVMOnlyAction emit_action(&llvm_context);
        if (!compiler.ExecuteAction(emit_action))
        {
            diag_stream.flush();
            error_reason = "Fail to emit LLVM IR for JIT source";
            if (!diag_log.empty())
            {
                error_reason += ":\n";
                error_reason += diag_log;
            }
            return false;
        }
        output_module = emit_action.takeModule();
        if (output_module == nullptr)
        {
            error_reason = "Clang emitted null LLVM module for JIT source";
            return false;
        }
        return true;
    }

    bool Compile_With_ORC(const std::string& source,
                          const std::string& func_name)
    {
        Initialize_ORC_Runtime();
        std::string openmp_load_error;
        if (!Ensure_OpenMP_Runtime_Loaded(openmp_load_error))
        {
            error_reason = openmp_load_error;
            return false;
        }
        auto jit = llvm::orc::LLJITBuilder().create();
        if (!jit)
        {
            error_reason =
                "Fail to create LLJIT: " + llvm::toString(jit.takeError());
            return false;
        }
        jit_engine = std::move(*jit);

#if defined(__linux__)
        const auto openmp_runtime_candidates =
            Build_OpenMP_Runtime_Candidates();
        bool openmp_generator_added = false;
        for (const auto& candidate : openmp_runtime_candidates)
        {
            auto openmp_generator =
                llvm::orc::DynamicLibrarySearchGenerator::Load(
                    candidate.c_str(),
                    jit_engine->getDataLayout().getGlobalPrefix());
            if (!openmp_generator)
            {
                continue;
            }
            jit_engine->getMainJITDylib().addGenerator(
                std::move(*openmp_generator));
            openmp_generator_added = true;
        }
        if (!openmp_generator_added)
        {
            error_reason =
                "Fail to create ORC symbol resolver for OpenMP runtime "
                "(tried: " +
                Join_Candidates(openmp_runtime_candidates) + ")";
            return false;
        }
#endif

        auto generator =
            llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
                jit_engine->getDataLayout().getGlobalPrefix());
        if (!generator)
        {
            error_reason = "Fail to create ORC symbol resolver: " +
                           llvm::toString(generator.takeError());
            return false;
        }
        jit_engine->getMainJITDylib().addGenerator(std::move(*generator));

        auto llvm_context = std::make_unique<llvm::LLVMContext>();
        std::unique_ptr<llvm::Module> module;
        if (!llvm_context)
        {
            error_reason = "Fail to allocate LLVM context for JIT source";
            return false;
        }
        if (!Build_Module_From_Source(source, *llvm_context, module))
        {
            return false;
        }

        module->setDataLayout(jit_engine->getDataLayout());
        if (module->getTargetTriple().str().empty())
        {
            module->setTargetTriple(
                llvm::Triple(llvm::sys::getDefaultTargetTriple()));
        }

        if (auto err = jit_engine->addIRModule(llvm::orc::ThreadSafeModule(
                std::move(module), std::move(llvm_context))))
        {
            error_reason = "Fail to add LLVM IR module into ORC JIT: " +
                           llvm::toString(std::move(err));
            return false;
        }

        auto symbol = jit_engine->lookup(func_name);
        if (!symbol)
        {
            error_reason = "Fail to lookup JIT symbol " + func_name + ": " +
                           llvm::toString(symbol.takeError());
            return false;
        }
        function = symbol->toPtr<void (*)(void**)>();
        if (function == nullptr)
        {
            error_reason =
                "Resolved JIT symbol " + func_name + " has null address";
            return false;
        }
        return true;
    }

   public:
    std::string error_reason;
    void Compile(std::string source)
    {
        error_reason.clear();
        function = nullptr;
        jit_engine.reset();

        size_t pos1 = source.find("extern");
        size_t pos2 =
            source.find(string_format("%q%C%q%", {{"q", {'"'}}}), pos1);
        if (pos2 == source.npos)
        {
            error_reason =
                R"(extern "C" should be placed in front of the function name)";
            return;
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_last_of(" ", pos1);
        std::string func_name = string_strip(source.substr(pos2, pos1 - pos2));
        if (func_name == "__launch_bounds__")
        {
            pos1 = source.find_first_of("(", pos1 + 1);
            pos2 = source.find_last_of(" ", pos1);
            func_name = string_strip(source.substr(pos2, pos1 - pos2));
        }
        pos1 = source.find_first_of("(", pos2);
        pos2 = source.find_first_of(")", pos1);
        std::vector<std::string> args =
            string_split(source.substr(pos1 + 1, pos2 - pos1 - 1), ",");
        source = source.replace(pos1 + 1, pos2 - pos1 - 1, "void** args");
        pos2 = source.find_first_of("{", pos1) + 1;
        auto is_identifier_char = [](char c)
        { return (c == '_') || std::isalnum(static_cast<unsigned char>(c)); };
        for (int i = args.size() - 1; i >= 0; i--)
        {
            std::string new_arg = string_strip(args[i]);
            if (new_arg.empty())
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            size_t var_end = new_arg.find_last_not_of(" \t");
            if (var_end == std::string::npos)
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            size_t var_start = var_end;
            while (var_start > 0 && is_identifier_char(new_arg[var_start]))
            {
                var_start--;
            }
            if (!is_identifier_char(new_arg[var_start]))
            {
                var_start++;
            }
            if (var_start > var_end)
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            std::string type_name = string_strip(new_arg.substr(0, var_start));
            std::string var_name = string_strip(
                new_arg.substr(var_start, var_end - var_start + 1));
            if (type_name.empty() || var_name.empty())
            {
                error_reason = "Fail to parse argument list for JIT wrapper.";
                return;
            }
            std::string stmt = type_name + " " + var_name + " = *((" +
                               type_name + "*)(args[" + std::to_string(i) +
                               "]));";
            source.insert(pos2, stmt);
        }
        if (!Compile_With_ORC(source, func_name))
        {
            return;
        }
    }

    void operator()(dim3 blocks, dim3 threads, deviceStream_t stream,
                    unsigned int shared_memory_size,
                    std::initializer_list<const void*> args)
    {
        std::vector<void*> temp;
        temp.reserve(args.size());
        for (const void* ptr : args)
        {
            temp.push_back(const_cast<void*>(ptr));
        }
        if (function == nullptr)
        {
            error_reason = "JIT function pointer is null";
            return;
        }
        function(temp.data());
    }

    void operator()(dim3 blocks, dim3 threads, deviceStream_t stream,
                    unsigned int shared_memory_size, std::vector<void*> args)
    {
        if (function == nullptr)
        {
            error_reason = "JIT function pointer is null";
            return;
        }
        function(args.data());
    }
};
#endif
