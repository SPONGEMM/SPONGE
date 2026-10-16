#ifdef _WIN32
#define PLUGIN_API extern "C" __declspec(dllexport)
#include <windows.h>
#ifndef RTLD_NOW
#define RTLD_NOW 0
#endif
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0
#endif
#define dlopen(filename, mode) LoadLibrary(filename)
#else
#define PLUGIN_API extern "C"
#include <dlfcn.h>
#endif

#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "../include/sponge_plugin_api.h"
#include "Python.h"
#include "dlpack.h"

static const SPONGE_PLUGIN_API* sponge_api = NULL;
static const SPONGE_LOCAL_MP_API_V1* sponge_local_mp = NULL;
static int is_initialized = 0;
static std::string py_script_path;
static DLDeviceType dlpack_device_type = kDLCPU;
static int dlpack_device_index = 0;
static bool checked_runtime_errors = false;
static bool runtime_error_latched = false;
static std::string runtime_error_stage;
static std::string runtime_error_message;

static void RuntimeFailure(const char* stage, const std::string& message)
{
    if (!checked_runtime_errors)
    {
        fprintf(stderr, "PRIPS [%s]: %s\n", stage, message.c_str());
        exit(1);
    }
    if (!runtime_error_latched)
    {
        runtime_error_stage = stage;
        runtime_error_message = message;
        runtime_error_latched = true;
    }
}

static void CapturePythonFailure(const char* stage)
{
    PyObject *type = nullptr, *value = nullptr, *trace = nullptr;
    PyErr_Fetch(&type, &value, &trace);
    PyErr_NormalizeException(&type, &value, &trace);
    std::string message = "Python callback failed";
    PyObject* text = value ? PyObject_Str(value) : nullptr;
    if (text)
    {
        const char* utf8 = PyUnicode_AsUTF8(text);
        if (utf8) message = utf8;
    }
    Py_XDECREF(text);
    PyErr_Clear();
    PyObject* traceback = PyImport_ImportModule("traceback");
    PyObject* lines =
        traceback && type
            ? PyObject_CallMethod(traceback, "format_exception", "OOO", type,
                                  value ? value : Py_None,
                                  trace ? trace : Py_None)
            : nullptr;
    PyObject* separator = lines ? PyUnicode_FromString("") : nullptr;
    PyObject* joined = separator ? PyUnicode_Join(separator, lines) : nullptr;
    if (joined)
    {
        const char* utf8 = PyUnicode_AsUTF8(joined);
        if (utf8) message = utf8;
    }
    Py_XDECREF(joined);
    Py_XDECREF(separator);
    Py_XDECREF(lines);
    Py_XDECREF(traceback);
    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(trace);
    PyErr_Clear();
    RuntimeFailure(stage, message);
}

static bool RunPythonCode(const char* stage, const char* source)
{
    if (runtime_error_latched) return false;
    if (!checked_runtime_errors)
    {
        if (PyRun_SimpleString(source) == 0) return true;
        RuntimeFailure(stage, "Python callback failed");
        return false;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = main ? PyModule_GetDict(main) : nullptr;
    PyObject* result =
        globals ? PyRun_String(source, Py_file_input, globals, globals)
                : nullptr;
    const bool success = result != nullptr;
    Py_XDECREF(result);
    if (!success) CapturePythonFailure(stage);
    PyGILState_Release(gil);
    return success;
}

PLUGIN_API int32_t Enable_Checked_Runtime_Errors_V1()
{
    checked_runtime_errors = true;
    return 0;
}

PLUGIN_API int32_t Get_Runtime_Error_V1(const char** stage,
                                        const char** message)
{
    if (stage)
        *stage = runtime_error_latched ? runtime_error_stage.c_str() : nullptr;
    if (message)
        *message =
            runtime_error_latched ? runtime_error_message.c_str() : nullptr;
    return runtime_error_latched ? 1 : 0;
}

PLUGIN_API int32_t Prepare_Local_MP_Setup_V1()
{
    if (runtime_error_latched) return -1;
    if (!is_initialized || !Py_IsInitialized()) return 0;
    if (!checked_runtime_errors)
    {
        RuntimeFailure("Prepare_Local_MP_Setup_V1",
                       "setup requires checked errors");
        return -1;
    }
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = main ? PyModule_GetDict(main) : nullptr;
    PyObject* driver =
        globals ? PyRun_String("getattr(Sponge, '_local_mp_driver', None)",
                               Py_eval_input, globals, globals)
                : nullptr;
    if (driver == Py_None)
    {
        Py_DECREF(driver);
        PyGILState_Release(gil);
        return 0;
    }
    if (driver != nullptr && (sponge_local_mp == nullptr ||
                              sponge_local_mp->submit_owned_outputs == nullptr))
        PyErr_SetString(PyExc_RuntimeError,
                        "native owned output submission is unavailable");
    PyObject* result =
        driver != nullptr && !PyErr_Occurred()
            ? PyObject_CallMethod(driver, "prepare_setup", nullptr)
            : nullptr;
    Py_XDECREF(driver);
    const bool success = result != nullptr;
    if (!success) CapturePythonFailure("Prepare_Local_MP_Setup_V1");
    Py_XDECREF(result);
    PyGILState_Release(gil);
    return success ? 1 : -1;
}

static bool controller_command_exist(const char* key)
{
    return sponge_api != NULL && sponge_api->get_command != NULL &&
           sponge_api->get_command(key) != NULL;
}

static const char* controller_command(const char* key)
{
    if (!controller_command_exist(key)) return NULL;
    return sponge_api->get_command(key);
}

static void controller_printf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    char buffer[4096];
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    if (sponge_api != NULL && sponge_api->log_message != NULL)
    {
        sponge_api->log_message(buffer);
    }
}

PLUGIN_API std::string Name()
{
    return std::string("Python Runtime Interface Plugin");
}

#ifndef PRIPS_VERSION
#define PRIPS_VERSION "0.0.0"
#endif

PLUGIN_API std::string Version() { return std::string(PRIPS_VERSION); }

PLUGIN_API std::string Version_Check(int i)
{
    if (i != SPONGE_PRIPS_API_VERSION && i != SPONGE_LOCAL_MP_API_VERSION)
    {
        return std::string("Reason:\n\tPRIPS v" PRIPS_VERSION
                           " expects SPONGE plugin API version " +
                           std::to_string(SPONGE_PRIPS_API_VERSION) +
                           " or local-MP " +
                           std::to_string(SPONGE_LOCAL_MP_API_VERSION) +
                           ", but got " + std::to_string(i));
    }
    return std::string();
}

static void delete_dltensor(DLManagedTensor* managed_tensor)
{
    DLTensor* tensor = &managed_tensor->dl_tensor;
    free(tensor->shape);
    if (tensor->strides != NULL) free(tensor->strides);
    free(managed_tensor);
}

// strides虽然dlpack支持，但是很多引擎不支持，所以尽量不给
static PyObject* create_dltensor(void* data, int N_dim, int64_t* shape,
                                 int64_t* strides, DLDataTypeCode type_code,
                                 uint8_t bits = 32, uint16_t lanes = 1)
{
    DLManagedTensor* managed_tensor =
        (DLManagedTensor*)malloc(sizeof(DLManagedTensor));
    memset(managed_tensor, 0, sizeof(DLManagedTensor));
    managed_tensor->deleter = delete_dltensor;
    DLTensor* tensor = &managed_tensor->dl_tensor;
    tensor->data = data;
    tensor->device = {dlpack_device_type, dlpack_device_index};
    tensor->ndim = N_dim;
    tensor->dtype = {(uint8_t)type_code, bits, lanes};
    tensor->shape =
        N_dim == 0 ? NULL : (int64_t*)malloc(sizeof(int64_t) * N_dim);
    if (N_dim > 0) memcpy(tensor->shape, shape, sizeof(int64_t) * N_dim);
    if (strides != NULL)
    {
        tensor->strides = (int64_t*)malloc(sizeof(int64_t) * N_dim);
        memcpy(tensor->strides, strides, sizeof(int64_t) * N_dim);
    }
    PyObject* a = PyCapsule_New((void*)managed_tensor, "dltensor", NULL);
    return a;
}

// These capsules own only DLPack metadata, never the SPONGE device arrays.
static void delete_borrowed_capsule(PyObject* capsule)
{
    if (!PyCapsule_IsValid(capsule, "dltensor")) return;
    auto* tensor = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    tensor->deleter(tensor);
}

static PyObject* create_neighbor_capsule(const int32_t* data, int ndim,
                                         int64_t* shape)
{
    PyObject* capsule =
        create_dltensor(const_cast<int32_t*>(data), ndim, shape, NULL, kDLInt);
    if (capsule != NULL)
        PyCapsule_SetDestructor(capsule, delete_borrowed_capsule);
    return capsule;
}

static PyObject* MlipNeighborEpoch(PyObject*, PyObject*)
{
    if (sponge_local_mp == NULL ||
        sponge_local_mp->get_mlip_neighbor_epoch == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "MLIP neighbor epoch requires local-MP API v1");
        return NULL;
    }
    return PyLong_FromUnsignedLongLong(
        sponge_local_mp->get_mlip_neighbor_epoch());
}

static PyObject* MlipNeighborView(PyObject*, PyObject*)
{
    if (sponge_local_mp == NULL ||
        sponge_local_mp->get_mlip_neighbor_view == NULL ||
        sponge_local_mp->get_mlip_neighbor_epoch == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "MLIP neighbor view requires local-MP API v1");
        return NULL;
    }
    SPONGE_MLIP_NEIGHBOR_VIEW view = {};
    const int status = sponge_local_mp->get_mlip_neighbor_view(&view);
    if (status != SPONGE_MLIP_NEIGHBOR_READY)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "MLIP neighbor graph unavailable (status %d)", status);
        return NULL;
    }
    if (view.neighbor_epoch == 0 ||
        view.neighbor_epoch != sponge_local_mp->get_mlip_neighbor_epoch() ||
        view.owned_count < 0 || view.edge_count < 0 ||
        (view.owned_count == 0 && view.edge_count != 0) ||
        view.row_ptr == NULL ||
        (view.edge_count > 0 &&
         (view.source_local_index == NULL || view.image_shift == NULL)))
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "Invalid MLIP neighbor graph metadata");
        return NULL;
    }
    int64_t row_shape[] = {static_cast<int64_t>(view.owned_count) + 1};
    int64_t source_shape[] = {view.edge_count};
    int64_t shift_shape[] = {view.edge_count, 3};
    // A valid int32 base also supports zero-length views in DLPack consumers.
    const int32_t* sources = view.source_local_index == NULL
                                 ? view.row_ptr
                                 : view.source_local_index;
    const int32_t* shifts =
        view.image_shift == NULL ? view.row_ptr : view.image_shift;
    PyObject* rows = create_neighbor_capsule(view.row_ptr, 1, row_shape);
    PyObject* source = create_neighbor_capsule(sources, 1, source_shape);
    PyObject* shift = create_neighbor_capsule(shifts, 2, shift_shape);
    if (rows == NULL || source == NULL || shift == NULL)
    {
        Py_XDECREF(rows);
        Py_XDECREF(source);
        Py_XDECREF(shift);
        return NULL;
    }
    return Py_BuildValue(
        "(KiiNNN)", static_cast<unsigned long long>(view.neighbor_epoch),
        view.owned_count, view.edge_count, rows, source, shift);
}

// MD information
static PyObject* Atom_Numbers(PyObject* self, PyObject* args)
{
    return Py_BuildValue(
        "i", sponge_api == NULL || sponge_api->get_atom_numbers == NULL
                 ? 0
                 : sponge_api->get_atom_numbers());
}

static PyObject* Steps(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i",
                         sponge_api == NULL || sponge_api->get_steps == NULL
                             ? 0
                             : sponge_api->get_steps());
}

static PyObject* Coordinate(PyObject* self, PyObject* args)
{
    const int atom_numbers =
        sponge_api == NULL || sponge_api->get_atom_numbers == NULL
            ? 0
            : sponge_api->get_atom_numbers();
    void* crd = sponge_api == NULL || sponge_api->get_coordinate_ptr == NULL
                    ? NULL
                    : sponge_api->get_coordinate_ptr();
    int64_t shape[2] = {atom_numbers, 3};
    return create_dltensor(crd, 2, shape, NULL, kDLFloat);
}

static PyObject* Force(PyObject* self, PyObject* args)
{
    const int atom_numbers =
        sponge_api == NULL || sponge_api->get_atom_numbers == NULL
            ? 0
            : sponge_api->get_atom_numbers();
    void* frc = sponge_api == NULL || sponge_api->get_force_ptr == NULL
                    ? NULL
                    : sponge_api->get_force_ptr();
    int64_t shape[2] = {atom_numbers, 3};
    return create_dltensor(frc, 2, shape, NULL, kDLFloat);
}

// Domain decomposition
static PyObject* Local_Atom_Numbers(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_local_atom_numbers == NULL)
        return Py_BuildValue("");
    return Py_BuildValue("i", sponge_api->get_local_atom_numbers());
}

static PyObject* Local_Ghost_Numbers(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_local_ghost_numbers == NULL)
        return Py_BuildValue("");
    return Py_BuildValue("i", sponge_api->get_local_ghost_numbers());
}

static PyObject* Local_PP_Rank(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_local_pp_rank == NULL)
        return Py_BuildValue("");
    return Py_BuildValue("i", sponge_api->get_local_pp_rank());
}

static PyObject* Atom_Local(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_atom_local_ptr == NULL ||
        sponge_api->get_local_atom_numbers == NULL ||
        sponge_api->get_local_ghost_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* atom_local = sponge_api->get_atom_local_ptr();
    if (atom_local == NULL) return Py_BuildValue("");
    int64_t shape[1] = {sponge_api->get_local_atom_numbers() +
                        sponge_api->get_local_ghost_numbers()};
    return create_dltensor(atom_local, 1, shape, NULL, kDLInt);
}

static PyObject* Atom_Local_Label(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_atom_local_label_ptr == NULL ||
        sponge_api->get_local_max_atom_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* atom_local_label = sponge_api->get_atom_local_label_ptr();
    if (atom_local_label == NULL) return Py_BuildValue("");
    int64_t shape[1] = {sponge_api->get_local_max_atom_numbers()};
    return create_dltensor(atom_local_label, 1, shape, NULL, kDLUInt, 8);
}

static PyObject* Atom_Local_Id(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_atom_local_id_ptr == NULL ||
        sponge_api->get_local_max_atom_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* atom_local_id = sponge_api->get_atom_local_id_ptr();
    if (atom_local_id == NULL) return Py_BuildValue("");
    int64_t shape[1] = {sponge_api->get_local_max_atom_numbers()};
    return create_dltensor(atom_local_id, 1, shape, NULL, kDLInt);
}

static PyObject* Local_Coordinate(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_local_coordinate_ptr == NULL ||
        sponge_api->get_local_atom_numbers == NULL ||
        sponge_api->get_local_ghost_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* crd = sponge_api->get_local_coordinate_ptr();
    if (crd == NULL) return Py_BuildValue("");
    int64_t shape[2] = {sponge_api->get_local_atom_numbers() +
                            sponge_api->get_local_ghost_numbers(),
                        3};
    return create_dltensor(crd, 2, shape, NULL, kDLFloat);
}

static PyObject* Local_Force(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_local_force_ptr == NULL ||
        sponge_api->get_local_atom_numbers == NULL ||
        sponge_api->get_local_ghost_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* frc = sponge_api->get_local_force_ptr();
    if (frc == NULL) return Py_BuildValue("");
    int64_t shape[2] = {sponge_api->get_local_atom_numbers() +
                            sponge_api->get_local_ghost_numbers(),
                        3};
    return create_dltensor(frc, 2, shape, NULL, kDLFloat);
}

// Neighbor List
static PyObject* Neighbor_List_Index(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_neighbor_list_index_ptr == NULL ||
        sponge_api->get_atom_numbers == NULL ||
        sponge_api->get_neighbor_list_max_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    void* nl_index = sponge_api->get_neighbor_list_index_ptr();
    if (nl_index == NULL) return Py_BuildValue("");
    int64_t shape[2] = {sponge_api->get_atom_numbers(),
                        sponge_api->get_neighbor_list_max_numbers()};
    return create_dltensor(nl_index, 2, shape, NULL, kDLInt);
}

static PyObject* Neighbor_List_Numbers(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_neighbor_list_count == NULL ||
        sponge_api->get_atom_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    const int atom_numbers = sponge_api->get_atom_numbers();
    PyObject* numbers = PyList_New(atom_numbers);
    for (int i = 0; i < atom_numbers; i++)
    {
        PyList_SET_ITEM(
            numbers, i,
            PyLong_FromLong(sponge_api->get_neighbor_list_count(i)));
    }
    return numbers;
}

static PyObject* Neighbor_List_Max_Numbers(PyObject* self, PyObject* args)
{
    if (sponge_api == NULL || sponge_api->get_neighbor_list_max_numbers == NULL)
    {
        return Py_BuildValue("");
    }
    return Py_BuildValue("i", sponge_api->get_neighbor_list_max_numbers());
}

// Domain information
// CONTROLLER
static PyObject* Control_Printf(PyObject* self, PyObject* args, PyObject* kw)
{
    static char* kwlist[] = {(char*)"toprint", NULL};
    char* buffer;
    if (!PyArg_ParseTupleAndKeywords(args, kw, "s", kwlist, &buffer))
    {
        return NULL;
    }
    controller_printf("%s", buffer);
    return Py_BuildValue("");
}

static PyObject* Control_MPI_Rank(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i",
                         sponge_api == NULL || sponge_api->get_mpi_rank == NULL
                             ? 0
                             : sponge_api->get_mpi_rank());
}

static PyObject* LocalMPRegisterCapabilities(PyObject*, PyObject* args)
{
    double cutoff, skin;
    int depth;
    unsigned long long hash;
    if (!PyArg_ParseTuple(args, "ddiK", &cutoff, &skin, &depth, &hash))
        return NULL;
    if (sponge_local_mp == NULL ||
        sponge_local_mp->register_mlip_capabilities == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PRIPS local-MP v1 capability callback is unavailable");
        return NULL;
    }
    return PyLong_FromLong(sponge_local_mp->register_mlip_capabilities(
        cutoff, skin, depth, SPONGE_MLIP_FEATURE_FP32,
        static_cast<uint64_t>(hash)));
}

static PyObject* LocalMPDeviceIndex(PyObject*, PyObject*)
{
    return PyLong_FromLong(
        sponge_local_mp == NULL ? -1 : sponge_local_mp->get_device_index());
}

static PyObject* LocalMPDomainEpoch(PyObject*, PyObject*)
{
    return PyLong_FromUnsignedLongLong(
        sponge_local_mp == NULL ? 0 : sponge_local_mp->get_domain_epoch());
}

static PyObject* LocalMPNeedPotential(PyObject*, PyObject*)
{
    return PyLong_FromLong(
        sponge_local_mp == NULL ? 0 : sponge_local_mp->get_need_potential());
}

static PyObject* LocalMPNeedPressure(PyObject*, PyObject*)
{
    return PyLong_FromLong(
        sponge_local_mp == NULL ? 0 : sponge_local_mp->get_need_pressure());
}

static PyObject* LocalMPCell(PyObject*, PyObject*)
{
    float cell[9];
    uint8_t pbc_mask = 0;
    if (sponge_local_mp == NULL ||
        sponge_local_mp->get_cell(cell, &pbc_mask) != 0)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PRIPS local-MP v1 cell is unavailable");
        return NULL;
    }
    PyObject* rows = PyTuple_New(3);
    for (int row = 0; row < 3; ++row)
    {
        PyObject* values = PyTuple_New(3);
        for (int col = 0; col < 3; ++col)
            PyTuple_SET_ITEM(values, col,
                             PyFloat_FromDouble(cell[row * 3 + col]));
        PyTuple_SET_ITEM(rows, row, values);
    }
    PyObject* result = PyTuple_New(2);
    PyTuple_SET_ITEM(result, 0, rows);
    PyTuple_SET_ITEM(result, 1, PyLong_FromLong(pbc_mask));
    return result;
}

static bool CurrentTorchCUDAStream(uintptr_t* stream)
{
    PyObject* torch = PyImport_ImportModule("torch");
    if (torch == NULL) return false;
    PyObject* cuda = PyObject_GetAttrString(torch, "cuda");
    Py_DECREF(torch);
    if (cuda == NULL) return false;
    PyObject* current =
        PyObject_CallMethod(cuda, "current_stream", "i", dlpack_device_index);
    Py_DECREF(cuda);
    if (current == NULL) return false;
    PyObject* handle = PyObject_GetAttrString(current, "cuda_stream");
    Py_DECREF(current);
    if (handle == NULL) return false;
    void* pointer = PyLong_AsVoidPtr(handle);
    Py_DECREF(handle);
    if (PyErr_Occurred()) return false;
    *stream = reinterpret_cast<uintptr_t>(pointer);
    return true;
}

static PyObject* LocalMPTensorExchange(PyObject*, PyObject* args, bool reverse)
{
    PyObject* tensor;
    if (!PyArg_ParseTuple(args, "O", &tensor)) return NULL;
    if (sponge_local_mp == NULL || sponge_api == NULL ||
        sponge_local_mp->exchange_on_stream == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "PRIPS local-MP v1 tensor exchange is unavailable");
        return NULL;
    }
    uintptr_t producer_stream = 0;
    if (!CurrentTorchCUDAStream(&producer_stream)) return NULL;
    // DLPack uses 1 for CUDA's legacy default stream, never the raw null
    // handle. Passing the current stream avoids an implicit default handoff.
    PyObject* capsule = NULL;
    {
        PyObject* method = PyObject_GetAttrString(tensor, "__dlpack__");
        if (method == NULL) return NULL;
        PyObject* empty = PyTuple_New(0);
        PyObject* keywords =
            Py_BuildValue("{s:K}", "stream",
                          static_cast<unsigned long long>(
                              producer_stream == 0 ? 1 : producer_stream));
        if (empty != NULL && keywords != NULL)
            capsule = PyObject_Call(method, empty, keywords);
        Py_XDECREF(keywords);
        Py_XDECREF(empty);
        Py_DECREF(method);
    }
    if (capsule == NULL) return NULL;
    DLManagedTensor* managed = static_cast<DLManagedTensor*>(
        PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed == NULL)
    {
        Py_DECREF(capsule);
        return NULL;
    }
    DLTensor& view = managed->dl_tensor;
    const int expected_rows = sponge_api->get_local_atom_numbers() +
                              sponge_api->get_local_ghost_numbers();
    const bool contiguous =
        view.ndim == 2 &&
        (view.strides == NULL ||
         (view.strides[1] == 1 && view.strides[0] == view.shape[1]));
    if (view.ndim != 2 || view.shape[0] != expected_rows ||
        view.shape[1] <= 0 || view.shape[1] > INT_MAX ||
        view.dtype.code != kDLFloat || view.dtype.bits != 32 ||
        view.dtype.lanes != 1 || dlpack_device_type != kDLCUDA ||
        view.device.device_type != kDLCUDA ||
        view.device.device_id != dlpack_device_index || !contiguous)
    {
        Py_DECREF(capsule);
        PyErr_SetString(PyExc_ValueError,
                        "exchange requires contiguous FP32 CUDA "
                        "local-plus-ghost rows on the selected device");
        return NULL;
    }
    float* data = reinterpret_cast<float*>(static_cast<char*>(view.data) +
                                           view.byte_offset);
    const int status = sponge_local_mp->exchange_on_stream(
        data, expected_rows, static_cast<int>(view.shape[1]), reverse,
        sponge_local_mp->get_domain_epoch(), producer_stream);
    Py_DECREF(capsule);
    return PyLong_FromLong(status);
}

static PyObject* LocalMPUpdateGhost(PyObject* self, PyObject* args)
{
    return LocalMPTensorExchange(self, args, false);
}

static PyObject* LocalMPAccumulateGhost(PyObject* self, PyObject* args)
{
    return LocalMPTensorExchange(self, args, true);
}

struct TorchDeviceBorrow
{
    PyObject* capsule = NULL;
    DLTensor* view = NULL;
    ~TorchDeviceBorrow() { Py_XDECREF(capsule); }

    bool acquire(PyObject* tensor, uintptr_t stream)
    {
        PyObject* method = PyObject_GetAttrString(tensor, "__dlpack__");
        if (method == NULL) return false;
        PyObject* empty = PyTuple_New(0);
        PyObject* keywords = Py_BuildValue(
            "{s:K}", "stream",
            static_cast<unsigned long long>(stream == 0 ? 1 : stream));
        if (empty != NULL && keywords != NULL)
            capsule = PyObject_Call(method, empty, keywords);
        Py_XDECREF(keywords);
        Py_XDECREF(empty);
        Py_DECREF(method);
        if (capsule == NULL) return false;
        auto* managed = static_cast<DLManagedTensor*>(
            PyCapsule_GetPointer(capsule, "dltensor"));
        if (managed == NULL) return false;
        view = &managed->dl_tensor;
        return true;
    }

    bool valid(int rank, int64_t rows, int64_t width, int code, int bits) const
    {
        if (view == NULL || view->ndim != rank || view->shape[0] != rows ||
            view->device.device_type != kDLCUDA ||
            view->device.device_id != dlpack_device_index ||
            view->dtype.code != code || view->dtype.bits != bits ||
            view->dtype.lanes != 1)
            return false;
        if (rank == 2 && view->shape[1] != width) return false;
        const bool empty = rows == 0;
        if (!empty && view->data == NULL) return false;
        if (view->strides != NULL && !empty)
        {
            if (rank == 1 && view->strides[0] != 1) return false;
            if (rank == 2 &&
                (view->strides[1] != 1 || view->strides[0] != width))
                return false;
        }
        return true;
    }

    void* data() const
    {
        return view->data == NULL
                   ? NULL
                   : static_cast<char*>(view->data) + view->byte_offset;
    }
};

static PyObject* LocalMPSubmitOwnedOutputs(PyObject*, PyObject* args)
{
    PyObject *forces_tensor, *energy_tensor, *virial_tensor;
    if (!PyArg_ParseTuple(args, "OOO", &forces_tensor, &energy_tensor,
                          &virial_tensor))
        return NULL;
    if (sponge_local_mp == NULL ||
        sponge_local_mp->submit_owned_outputs == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "native owned output submission is unavailable");
        return NULL;
    }
    uintptr_t producer_stream = 0;
    if (!CurrentTorchCUDAStream(&producer_stream)) return NULL;
    TorchDeviceBorrow forces, energy, virial;
    if (!forces.acquire(forces_tensor, producer_stream) ||
        (energy_tensor != Py_None &&
         !energy.acquire(energy_tensor, producer_stream)) ||
        (virial_tensor != Py_None &&
         !virial.acquire(virial_tensor, producer_stream)))
        return NULL;
    const int owned = sponge_local_mp->prips_api->get_local_atom_numbers();
    if (!forces.valid(2, owned, 3, kDLFloat, 32) ||
        (energy_tensor != Py_None && !energy.valid(1, 1, 0, kDLFloat, 32)) ||
        (virial_tensor != Py_None && !virial.valid(2, 3, 3, kDLFloat, 32)))
    {
        PyErr_SetString(PyExc_ValueError,
                        "owned output submission requires contiguous FP32 "
                        "CUDA tensors on the selected device with shapes "
                        "[owned,3], optional [1] energy and optional [3,3] "
                        "physical virial in SPONGE units");
        return NULL;
    }
    const int status = sponge_local_mp->submit_owned_outputs(
        sponge_local_mp->get_domain_epoch(), owned,
        static_cast<const float*>(forces.data()),
        energy_tensor == Py_None ? NULL
                                 : static_cast<const float*>(energy.data()),
        virial_tensor == Py_None ? NULL
                                 : static_cast<const float*>(virial.data()),
        producer_stream);
    // Host completion keeps every borrowed capsule alive through native reads.
    return PyLong_FromLong(status);
}

static PyObject* LocalMPAssembleEdgeGradients(PyObject*, PyObject* args)
{
    PyObject *ids_tensor, *gradients_tensor, *forces_tensor, *virial_tensor;
    unsigned long long neighbor_epoch;
    double length_scale;
    if (!PyArg_ParseTuple(args, "OOKdOO", &ids_tensor, &gradients_tensor,
                          &neighbor_epoch, &length_scale, &forces_tensor,
                          &virial_tensor))
        return NULL;
    if (sponge_local_mp == NULL ||
        sponge_local_mp->assemble_selected_edge_gradients == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError,
                        "native edge output assembly is unavailable");
        return NULL;
    }
    if (neighbor_epoch == 0 || !std::isfinite(length_scale) ||
        length_scale <= 0)
    {
        PyErr_SetString(PyExc_ValueError,
                        "invalid edge neighbor epoch or length conversion");
        return NULL;
    }
    uintptr_t producer_stream = 0;
    if (!CurrentTorchCUDAStream(&producer_stream)) return NULL;
    TorchDeviceBorrow ids, gradients, forces, virial;
    if (!ids.acquire(ids_tensor, producer_stream) ||
        !gradients.acquire(gradients_tensor, producer_stream) ||
        !forces.acquire(forces_tensor, producer_stream) ||
        (virial_tensor != Py_None &&
         !virial.acquire(virial_tensor, producer_stream)))
        return NULL;
    const int64_t count = ids.view->ndim == 1 ? ids.view->shape[0] : -1;
    const int owned = sponge_local_mp->prips_api->get_local_atom_numbers();
    if (count < 0 || count > INT_MAX || !ids.valid(1, count, 0, kDLInt, 64) ||
        !gradients.valid(2, count, 3, kDLFloat, 32) ||
        !forces.valid(2, owned, 3, kDLFloat, 32) ||
        (virial_tensor != Py_None && !virial.valid(2, 3, 3, kDLFloat, 32)))
    {
        PyErr_SetString(PyExc_ValueError,
                        "edge assembly requires contiguous int64 IDs and FP32 "
                        "gradients/outputs "
                        "on the selected CUDA device with shapes [E], [E,3], "
                        "[owned,3], [3,3]");
        return NULL;
    }
    const int status = sponge_local_mp->assemble_selected_edge_gradients(
        sponge_local_mp->get_domain_epoch(),
        static_cast<uint64_t>(neighbor_epoch), static_cast<int>(count),
        static_cast<const int64_t*>(ids.data()),
        static_cast<const float*>(gradients.data()),
        static_cast<float*>(forces.data()),
        virial_tensor == Py_None ? NULL : static_cast<float*>(virial.data()),
        virial_tensor == Py_None ? 0 : 1, length_scale, producer_stream);
    // The callback returns only after every input read and output write ends.
    // Keep all capsules alive through that completion boundary.
    return PyLong_FromLong(status);
}

static PyMethodDef SpongeMethods[] = {
    {"_submit_owned_outputs", (PyCFunction)LocalMPSubmitOwnedOutputs,
     METH_VARARGS,
     "Add owned force, optional energy and physical virial in SPONGE units; "
     "host complete."},
    {"_mlip_neighbor_epoch", (PyCFunction)MlipNeighborEpoch, METH_NOARGS,
     "Return neighbor epoch; zero means no published graph."},
    {"_mlip_neighbor_view", (PyCFunction)MlipNeighborView, METH_NOARGS,
     "Return (epoch, owned_count, edge_count, row_ptr, source_local_index, "
     "image_shift). Arrays are borrowed int32 DLPack capsules with shapes "
     "[owned_count+1], [edge_count], [edge_count,3], valid until epoch "
     "changes. "
     "Request a view only when the epoch changes; unavailable graphs raise."},
    {"_atom_numbers", (PyCFunction)Atom_Numbers, METH_VARARGS, ""},
    {"_steps", (PyCFunction)Steps, METH_VARARGS, ""},
    {"_coordinate", (PyCFunction)Coordinate, METH_VARARGS, ""},
    {"_force", (PyCFunction)Force, METH_VARARGS, ""},
    {"_local_atom_numbers", (PyCFunction)Local_Atom_Numbers, METH_VARARGS, ""},
    {"_local_ghost_numbers", (PyCFunction)Local_Ghost_Numbers, METH_VARARGS,
     ""},
    {"_local_pp_rank", (PyCFunction)Local_PP_Rank, METH_VARARGS, ""},
    {"_atom_local", (PyCFunction)Atom_Local, METH_VARARGS, ""},
    {"_atom_local_label", (PyCFunction)Atom_Local_Label, METH_VARARGS, ""},
    {"_atom_local_id", (PyCFunction)Atom_Local_Id, METH_VARARGS, ""},
    {"_local_crd", (PyCFunction)Local_Coordinate, METH_VARARGS, ""},
    {"_local_frc", (PyCFunction)Local_Force, METH_VARARGS, ""},
    {"_neighbor_list_number", (PyCFunction)Neighbor_List_Numbers, METH_VARARGS,
     ""},
    {"_neighbor_list_index", (PyCFunction)Neighbor_List_Index, METH_VARARGS,
     ""},
    {"_neighbor_list_max_numbers", (PyCFunction)Neighbor_List_Max_Numbers,
     METH_VARARGS, ""},
    {"_printf", (PyCFunction)Control_Printf, METH_VARARGS | METH_KEYWORDS, ""},
    {"_MPI_rank", (PyCFunction)Control_MPI_Rank, METH_VARARGS, ""},
    {"_register_mlip_capabilities", (PyCFunction)LocalMPRegisterCapabilities,
     METH_VARARGS, ""},
    {"_device_index", (PyCFunction)LocalMPDeviceIndex, METH_NOARGS, ""},
    {"_domain_epoch", (PyCFunction)LocalMPDomainEpoch, METH_NOARGS, ""},
    {"_need_potential", (PyCFunction)LocalMPNeedPotential, METH_NOARGS, ""},
    {"_need_pressure", (PyCFunction)LocalMPNeedPressure, METH_NOARGS, ""},
    {"_cell", (PyCFunction)LocalMPCell, METH_NOARGS, ""},
    {"_update_ghost_tensor", (PyCFunction)LocalMPUpdateGhost, METH_VARARGS, ""},
    {"_accumulate_ghost_tensor", (PyCFunction)LocalMPAccumulateGhost,
     METH_VARARGS, ""},
    {"_assemble_edge_gradients", (PyCFunction)LocalMPAssembleEdgeGradients,
     METH_VARARGS,
     "Assemble selected native candidate edge gradients in model units."},
    {NULL, NULL, 0, NULL}};

static PyModuleDef SpongeModule = {PyModuleDef_HEAD_INIT,
                                   "Sponge",
                                   NULL,
                                   -1,
                                   SpongeMethods,
                                   NULL,
                                   NULL,
                                   NULL,
                                   NULL};

PyMODINIT_FUNC PyInit_sponge(void) { return PyModule_Create(&SpongeModule); }

static PyMethodDef PripsMethods[] = {{NULL, NULL, 0, NULL}};

static PyModuleDef pripsModule = {PyModuleDef_HEAD_INIT,
                                  "prips",
                                  NULL,
                                  -1,
                                  PripsMethods,
                                  NULL,
                                  NULL,
                                  NULL,
                                  NULL};

PyMODINIT_FUNC PyInit_prips(void)
{
    PyObject* m = PyModule_Create(&pripsModule);
    PyModule_AddStringConstant(m, "__version__", Version().c_str());
    PyObject* m0 = PyInit_sponge();
    PyModule_AddIntConstant(m0, "_backend",
                            static_cast<int>(dlpack_device_type));
    PyModule_AddIntConstant(m0, "_backend_index", dlpack_device_index);
    PyModule_AddIntConstant(m0, "_local_mp", sponge_local_mp != NULL);
    PyModule_AddObject(m, "Sponge", m0);
    return m;
}

PLUGIN_API void Set_Backend_Device_Type(int device_type)
{
    dlpack_device_type = static_cast<DLDeviceType>(device_type);
}

PLUGIN_API void Initial_Stable(const SPONGE_PLUGIN_API* api)
{
    if (runtime_error_latched) return;
    sponge_api = api;
    if (sponge_api == NULL)
    {
        RuntimeFailure("Initial_Stable",
                       "PRIPS initialization failed: null API.");
        return;
    }
    dlpack_device_type = static_cast<DLDeviceType>(sponge_api->device_type);
    dlpack_device_index =
        sponge_local_mp == NULL ? 0 : sponge_local_mp->get_device_index();
    if (dlpack_device_index < 0) dlpack_device_index = 0;
    controller_printf("    initializing pyplugin\n");
    if (controller_command_exist("py"))
    {
        py_script_path = controller_command("py");
    }
    else
    {
        const char* py_env = std::getenv("SPONGE_PRIPS_PY");
        if (py_env != NULL && py_env[0] != '\0')
        {
            py_script_path = py_env;
            controller_printf(
                "        No 'py' command found. Falling back to "
                "SPONGE_PRIPS_PY.\n");
        }
        else
        {
            controller_printf(
                "        No 'py' command found. Pyplugin will not be "
                "initialized.\n");
            return;
        }
    }
    if (PyImport_AppendInittab("prips", &PyInit_prips) != 0)
    {
        RuntimeFailure("Initial_Stable",
                       "could not register the embedded prips module");
        return;
    }
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        RuntimeFailure("Initial_Stable", "Python initialization failed");
        return;
    }
    else
    {
        controller_printf("        Python Initialized\n");
    }

    wchar_t* temp_args[1] = {(wchar_t*)L"SPONGE"};
    PySys_SetArgv(1, temp_args);
    PyObject* main = PyImport_AddModule("__main__");
    if (PyErr_Occurred() || main == nullptr ||
        PyModule_AddIntConstant(main, "_prips_checked_errors",
                                checked_runtime_errors) != 0)
    {
        CapturePythonFailure("Python bootstrap");
        return;
    }
    if (!RunPythonCode("Python bootstrap", R"XYJ(
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util as iu

old_excepthook = sys.excepthook
def new_hook(exctype, value, traceback):
    old_excepthook(exctype, value, traceback)
    exit(1)
if not _prips_checked_errors:
    sys.excepthook = new_hook

from prips import Sponge

# The embedded extension is named prips; expose the installed Python package
# directory so a user script can import prips.local_mp at module import time.
import prips as _prips_module
_prips_module.__path__ = [str(Path(p) / "prips") for p in sys.path
                          if (Path(p) / "prips" / "local_mp" / "__init__.py").is_file()]

class SpongeDLPackTensor:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, *args, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return (Sponge._backend, Sponge._backend_index)

Sponge.SpongeDLPackTensor = SpongeDLPackTensor

class MD_INFORMATION:
    def __init__(self):
        self._crd = Sponge.SpongeDLPackTensor(Sponge._coordinate())
        self._frc = Sponge.SpongeDLPackTensor(Sponge._force())

    class system_information:
        @property
        def steps(self):
            return Sponge._steps()

    @property
    def atom_numbers(self):
        return Sponge._atom_numbers()

    @property
    def crd(self):
        return self._crd

    @property
    def frc(self):
        return self._frc

Sponge.MD_INFORMATION = MD_INFORMATION
Sponge.md_info = MD_INFORMATION()
Sponge.md_info.sys = MD_INFORMATION.system_information()

class NEIGHBOR_LIST:
    def __init__(self):
        self._number = Sponge._neighbor_list_number()
        self._max_neighbor_numbers = Sponge._neighbor_list_max_numbers()
        index_capsule = Sponge._neighbor_list_index()
        self._index = None if index_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(index_capsule))

    @property
    def index(self):
        return self._index

    @property
    def number(self):
        return self._number

    @property
    def max_neighbor_numbers(self):
        return self._max_neighbor_numbers

Sponge.NEIGHBOR_LIST = NEIGHBOR_LIST
Sponge.neighbor_list = None

class DOMAIN_INFORMATION:
    @property
    def atom_numbers(self):
        return Sponge._local_atom_numbers()

    @property
    def ghost_numbers(self):
        return Sponge._local_ghost_numbers()

    @property
    def pp_rank(self):
        return Sponge._local_pp_rank()

    @property
    def cell(self):
        # Box changes need not replace the local atom map or this wrapper.
        return Sponge._cell()[0] if Sponge._local_mp else None

    @property
    def pbc(self):
        pbc_mask = Sponge._cell()[1] if Sponge._local_mp else 0
        return tuple(bool(pbc_mask & (1 << axis)) for axis in range(3))

    def __init__(self):
        self.domain_epoch = Sponge._domain_epoch()
        atom_local_capsule = Sponge._atom_local()
        atom_local_label_capsule = Sponge._atom_local_label()
        atom_local_id_capsule = Sponge._atom_local_id()
        crd_capsule = Sponge._local_crd()
        frc_capsule = Sponge._local_frc()
        self._atom_local = None if atom_local_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(atom_local_capsule))
        self._atom_local_label = None if atom_local_label_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(atom_local_label_capsule))
        self._atom_local_id = None if atom_local_id_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(atom_local_id_capsule))
        self._crd = None if crd_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(crd_capsule))
        self._frc = None if frc_capsule is None else Sponge.backend(Sponge.SpongeDLPackTensor(frc_capsule))

    @property
    def atom_local(self):
        return self._atom_local

    @property
    def atom_local_label(self):
        return self._atom_local_label

    @property
    def atom_local_id(self):
        return self._atom_local_id

    @property
    def crd(self):
        return self._crd

    @property
    def frc(self):
        return self._frc

Sponge.DOMAIN_INFORMATION = DOMAIN_INFORMATION
Sponge.dd = None

class CONTROLLER:
    @property
    def MPI_rank(self):
        return Sponge._MPI_rank()

    def printf(self, *values, sep=" ", end="\n"):
        return Sponge._printf(sep.join([f"{i}" for i in values]) + end)

Sponge.CONTROLLER = CONTROLLER
Sponge.controller = CONTROLLER()

def _numpy_backend(dlpack_tensor):
    import ctypes
    import numpy as np

    class DLDevice(ctypes.Structure):
        _fields_ = [('device_type', ctypes.c_int), ('device_id', ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [('code', ctypes.c_uint8), ('bits', ctypes.c_uint8), ('lanes', ctypes.c_uint16)]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ('data', ctypes.c_void_p),
            ('device', DLDevice),
            ('ndim', ctypes.c_int),
            ('dtype', DLDataType),
            ('shape', ctypes.POINTER(ctypes.c_int64)),
            ('strides', ctypes.POINTER(ctypes.c_int64)),
            ('byte_offset', ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        _fields_ = [('dl_tensor', DLTensor), ('manager_ctx', ctypes.c_void_p), ('deleter', ctypes.c_void_p)]

    pycapsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    pycapsule_get_pointer.restype = ctypes.c_void_p
    pycapsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]

    capsule = dlpack_tensor.capsule
    ptr = pycapsule_get_pointer(capsule, b'dltensor')
    managed = ctypes.cast(ptr, ctypes.POINTER(DLManagedTensor)).contents
    tensor = managed.dl_tensor
    shape = tuple(tensor.shape[i] for i in range(tensor.ndim))
    if tensor.dtype.lanes != 1:
        raise TypeError(f'unsupported dlpack lanes: {tensor.dtype.lanes}')
    dtype_map = {
        (0, 32): (ctypes.c_int32, np.int32),
        (1, 8): (ctypes.c_uint8, np.uint8),
        (2, 32): (ctypes.c_float, np.float32),
    }
    key = (int(tensor.dtype.code), int(tensor.dtype.bits))
    if key not in dtype_map:
        raise TypeError(
            f'unsupported dlpack dtype: code={tensor.dtype.code}, '
            f'bits={tensor.dtype.bits}, lanes={tensor.dtype.lanes}'
        )
    ctype, np_dtype = dtype_map[key]
    size = int(np.prod(shape, dtype=np.int64))
    data_ptr = tensor.data + tensor.byte_offset
    buffer = (ctype * size).from_address(data_ptr)
    return np.ctypeslib.as_array(buffer).view(np_dtype).reshape(shape)

def _resolve_backend(name):
    backend = name.lower()
    if backend == "numpy":
        return _numpy_backend
    if backend == "jax":
        import jax.dlpack
        return jax.dlpack.from_dlpack
    if backend == "cupy":
        import cupy
        return cupy.from_dlpack
    if backend in ("pytorch", "torch"):
        import torch
        return torch.from_dlpack
    raise ValueError(
        "backend must be one of: 'numpy', 'jax', 'cupy', 'pytorch'"
    )

def _device_name(device_type):
    names = {
        1: "cpu",
        2: "cuda",
        10: "rocm",
    }
    return names.get(device_type, f"device_type={device_type}")

def set_backend(backend):
    import warnings

    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    if Sponge.backend_name is not None:
        raise RuntimeError("backend has already been set")
    backend_name = backend.lower()
    device_type = Sponge._backend
    if backend_name == "numpy" and device_type != 1:
        raise ValueError(
            f"backend 'numpy' requires CPU tensors, but current device is "
            f"{_device_name(device_type)}"
        )
    if backend_name == "cupy" and device_type == 1:
        raise ValueError(
            "backend 'cupy' requires GPU tensors, but current device is "
            f"{_device_name(device_type)}"
        )
    if backend_name == "jax":
        warnings.warn(
            "jax backend is read-only in PRIPS. "
            "Use numpy/cupy/pytorch if you need in-place writable tensors.",
            RuntimeWarning,
            stacklevel=2,
        )
    Sponge.backend = _resolve_backend(backend_name)
    Sponge.md_info._crd = Sponge.backend(Sponge.md_info._crd)
    Sponge.md_info._frc = Sponge.backend(Sponge.md_info._frc)
    Sponge.backend_name = backend_name

Sponge.set_backend = set_backend
Sponge.backend_name = None

)XYJ"))
        return;

    char buffer[4096];
    sprintf(buffer, "Sponge.fname = r'%s'", py_script_path.c_str());
    if (!RunPythonCode("Python script path", buffer)) return;
    if (!RunPythonCode("Python user module",
                       R"XYJ(sponge_pyplugin_path = Path(Sponge.fname)
spec = iu.spec_from_file_location('sponge_pyplugin', sponge_pyplugin_path)
if spec is None:
    Sponge.controller.printf(f"Module '{Sponge.fname}' is not found.")
    sponge_pyplugin = None
else:
    sponge_pyplugin = iu.module_from_spec(spec)
    spec.loader.exec_module(sponge_pyplugin)
    Sponge.controller.printf("        module '%s' imported."%(sponge_pyplugin_path.stem))
)XYJ"))
        return;

    is_initialized = 1;
    controller_printf("    end initializing pyplugin\n");
}

PLUGIN_API void Initial_Local_MP_V1(const SPONGE_LOCAL_MP_API_V1* api)
{
    if (runtime_error_latched) return;
    if (api == NULL || api->api_version != SPONGE_LOCAL_MP_API_VERSION ||
        api->prips_api == NULL ||
        api->prips_api->api_version != SPONGE_PRIPS_API_VERSION ||
        api->register_mlip_capabilities == NULL || api->get_pp_rank == NULL ||
        api->get_pp_size == NULL || api->get_device_index == NULL ||
        api->get_domain_epoch == NULL || api->get_need_potential == NULL ||
        api->get_need_pressure == NULL || api->get_cell == NULL ||
        api->get_mlip_neighbor_epoch == NULL ||
        api->get_mlip_neighbor_view == NULL ||
        (api->prips_api->device_type == kDLCUDA &&
         (api->exchange_on_stream == NULL ||
          api->submit_owned_outputs == NULL)))
    {
        RuntimeFailure(
            "Initial_Local_MP_V1",
            "PRIPS requires a complete local-MP API v1 with a valid v2 base.");
        return;
    }
    sponge_local_mp = api;
    Initial_Stable(api->prips_api);
}

PLUGIN_API void Initial(void* md, void* ctrl, void* nl, void* cv, void* cv_map,
                        void* cv_instance_map)
{
    (void)md;
    (void)ctrl;
    (void)nl;
    (void)cv;
    (void)cv_map;
    (void)cv_instance_map;
    RuntimeFailure("Initial",
                   "PRIPS requires SPONGE stable plugin API support. "
                   "Rebuild SPONGE with Initial_Stable loader support.");
}

PLUGIN_API void After_Initial()
{
    if (!is_initialized || runtime_error_latched) return;
    RunPythonCode("After_Initial", R"XYJ(
if sponge_pyplugin is not None and hasattr(sponge_pyplugin, "local_mp_adapter"):
    if Sponge.backend_name is None:
        Sponge.set_backend("pytorch")
    if Sponge.backend_name not in ("pytorch", "torch"):
        raise RuntimeError("local-MP requires the PyTorch backend")
if Sponge.neighbor_list is None:
    Sponge.neighbor_list = Sponge.NEIGHBOR_LIST()
if hasattr(sponge_pyplugin, "After_Initial"):
    sponge_pyplugin.After_Initial()
if sponge_pyplugin is not None and hasattr(sponge_pyplugin, "local_mp_adapter"):
    from prips.local_mp import ResidentLocalMPDriver
    Sponge._local_mp_driver = ResidentLocalMPDriver(
        Sponge, sponge_pyplugin.local_mp_adapter, sponge_pyplugin.atom_types)
    )XYJ");
}

// Optional plugin declaration, queried after all After_Initial callbacks.
// False is a whole-run promise covering the adapter and all executed code.
PLUGIN_API int Uses_Classic_Neighbor_List()
{
    if (!is_initialized || runtime_error_latched || !Py_IsInitialized())
        return 1;
    const PyGILState_STATE gil = PyGILState_Ensure();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* globals = main == nullptr ? nullptr : PyModule_GetDict(main);
    PyObject* result =
        globals == nullptr
            ? nullptr
            : PyRun_String(
                  "getattr(Sponge, '_local_mp_driver', None) is not None and "
                  "getattr(sponge_pyplugin, 'uses_classic_neighbor_list', "
                  "None) is False and "
                  "not any(hasattr(sponge_pyplugin, name) for name in "
                  "('After_Initial', 'Calculate_Force', "
                  "'Set_Domain_Information', 'Mdout_Print'))",
                  Py_eval_input, globals, globals);
    const int required = result == Py_True ? 0 : 1;
    Py_XDECREF(result);
    if (PyErr_Occurred()) PyErr_Clear();
    PyGILState_Release(gil);
    return required;
}

PLUGIN_API void Set_Domain_Information(void* domain_info)
{
    (void)domain_info;
    if (!is_initialized || runtime_error_latched) return;
    RunPythonCode("Set_Domain_Information", R"XYJ(
Sponge.dd = Sponge.DOMAIN_INFORMATION()
if hasattr(sponge_pyplugin, "Set_Domain_Information"):
    sponge_pyplugin.Set_Domain_Information()
if hasattr(Sponge, "_local_mp_driver"):
    Sponge._local_mp_driver.set_domain_information(Sponge.dd)
)XYJ");
}

PLUGIN_API void Calculate_Force()
{
    if (!is_initialized || runtime_error_latched) return;
    RunPythonCode("Calculate_Force", R"XYJ(
if hasattr(sponge_pyplugin, "Calculate_Force"):
    sponge_pyplugin.Calculate_Force()
if hasattr(Sponge, "_local_mp_driver"):
    Sponge._local_mp_driver.calculate_force()
    )XYJ");
}

PLUGIN_API void Mdout_Print()
{
    if (!is_initialized || runtime_error_latched) return;
    RunPythonCode("Mdout_Print", R"XYJ(
if hasattr(sponge_pyplugin, "Mdout_Print"):
    sponge_pyplugin.Mdout_Print()
    )XYJ");
}
