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

#include <cstdarg>
#include <cstdlib>
#include <sstream>

#include "Python.h"
#include "dlpack.h"
#include "sponge_abi.h"

static MD_INFORMATION* md_info = NULL;
static CONTROLLER* controller = NULL;
static NEIGHBOR_LIST* neighbor_list = NULL;
static DOMAIN_INFORMATION* dd = NULL;
static int is_initialized = 0;
static std::string py_script_path;
static DLDeviceType dlpack_device_type = kDLCPU;

static bool controller_command_exist(const char* key)
{
    return controller != NULL && controller->commands.count(key) > 0;
}

static const char* controller_command(const char* key)
{
    if (!controller_command_exist(key)) return NULL;
    return controller->commands.find(key)->second.c_str();
}

static void controller_printf(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    va_list args_copy;
    va_copy(args_copy, args);
    vprintf(fmt, args);
    if (controller != NULL && controller->mdinfo != NULL)
        vfprintf(controller->mdinfo, fmt, args_copy);
    va_end(args_copy);
    va_end(args);
}

static void fail_abi_check(const std::string& reason)
{
    fprintf(stderr, "        PRIPS ABI check failed: %s\n", reason.c_str());
    exit(1);
}

static int read_layout_version(const void* ptr, size_t offset)
{
    int value = 0;
    memcpy(&value, (const char*)ptr + offset, sizeof(int));
    return value;
}

PLUGIN_API std::string Name()
{
    return std::string("Python Runtime Interface Plugin");
}

PLUGIN_API std::string Version() { return std::string("2.0a0"); }

PLUGIN_API std::string Version_Check(int i)
{
    if (i != SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION)
    {
        return std::string(
            "Reason:\n\tPRIPS v2.0a0 expects SPONGE layout version " +
            std::to_string(SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION) +
            ", but got " + std::to_string(i));
    }
    return std::string();
}

static void delete_dltensor(DLManagedTensor* managed_tensor)
{
    DLTensor* tensor = &managed_tensor->dl_tensor;
    free(tensor->shape);
    if (tensor->strides != NULL) free(tensor->strides);
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
    tensor->device = {dlpack_device_type, 0};
    tensor->ndim = N_dim;
    tensor->dtype = {(uint8_t)type_code, bits, lanes};
    tensor->shape = (int64_t*)malloc(sizeof(int64_t) * N_dim);
    memcpy(tensor->shape, shape, sizeof(int64_t) * N_dim);
    if (strides != NULL)
    {
        tensor->strides = (int64_t*)malloc(sizeof(int64_t) * N_dim);
        memcpy(tensor->strides, strides, sizeof(int64_t) * N_dim);
    }
    PyObject* a = PyCapsule_New((void*)managed_tensor, "dltensor", NULL);
    return a;
}

// MD information
static PyObject* Atom_Numbers(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i", md_info->atom_numbers);
}

static PyObject* Steps(PyObject* self, PyObject* args)
{
    return Py_BuildValue("i", md_info->sys.steps);
}

static PyObject* Coordinate(PyObject* self, PyObject* args)
{
    int64_t shape[2] = {3, md_info->atom_numbers};
    return create_dltensor((void*)md_info->crd, 2, shape, NULL, kDLFloat);
}

static PyObject* Force(PyObject* self, PyObject* args)
{
    int64_t shape[2] = {3, md_info->atom_numbers};
    return create_dltensor((void*)md_info->frc, 2, shape, NULL, kDLFloat);
}

// Domain decomposition
static PyObject* Local_Atom_Numbers(PyObject* self, PyObject* args)
{
    if (dd == NULL) return Py_BuildValue("");
    return Py_BuildValue("i", dd->atom_numbers);
}

static PyObject* Local_Ghost_Numbers(PyObject* self, PyObject* args)
{
    if (dd == NULL) return Py_BuildValue("");
    return Py_BuildValue("i", dd->ghost_numbers);
}

static PyObject* Local_PP_Rank(PyObject* self, PyObject* args)
{
    if (dd == NULL) return Py_BuildValue("");
    return Py_BuildValue("i", dd->pp_rank);
}

static PyObject* Atom_Local(PyObject* self, PyObject* args)
{
    if (dd == NULL || dd->atom_local == NULL) return Py_BuildValue("");
    int64_t shape[1] = {dd->atom_numbers + dd->ghost_numbers};
    return create_dltensor((void*)dd->atom_local, 1, shape, NULL, kDLInt);
}

static PyObject* Atom_Local_Label(PyObject* self, PyObject* args)
{
    if (dd == NULL || dd->atom_local_label == NULL) return Py_BuildValue("");
    int64_t shape[1] = {dd->max_atom_numbers};
    return create_dltensor((void*)dd->atom_local_label, 1, shape, NULL, kDLUInt,
                           8);
}

static PyObject* Atom_Local_Id(PyObject* self, PyObject* args)
{
    if (dd == NULL || dd->atom_local_id == NULL) return Py_BuildValue("");
    int64_t shape[1] = {dd->max_atom_numbers};
    return create_dltensor((void*)dd->atom_local_id, 1, shape, NULL, kDLInt);
}

static PyObject* Local_Coordinate(PyObject* self, PyObject* args)
{
    if (dd == NULL || dd->crd == NULL) return Py_BuildValue("");
    int64_t shape[2] = {3, dd->atom_numbers + dd->ghost_numbers};
    return create_dltensor((void*)dd->crd, 2, shape, NULL, kDLFloat);
}

static PyObject* Local_Force(PyObject* self, PyObject* args)
{
    if (dd == NULL || dd->frc == NULL) return Py_BuildValue("");
    int64_t shape[2] = {3, dd->atom_numbers + dd->ghost_numbers};
    return create_dltensor((void*)dd->frc, 2, shape, NULL, kDLFloat);
}

// Neighbor List
static PyObject* Neighbor_List_Index(PyObject* self, PyObject* args)
{
    if (neighbor_list == NULL || neighbor_list->h_nl == NULL)
    {
        return Py_BuildValue("");
    }
    int64_t shape[2] = {md_info->atom_numbers,
                        neighbor_list->max_neighbor_numbers};
    return create_dltensor((void*)neighbor_list->h_nl->atom_serial, 2, shape,
                           NULL, kDLInt);
}

static PyObject* Neighbor_List_Numbers(PyObject* self, PyObject* args)
{
    if (neighbor_list == NULL || neighbor_list->h_nl == NULL)
    {
        return Py_BuildValue("");
    }
    PyObject* numbers = PyList_New(md_info->atom_numbers);
    for (int i = 0; i < md_info->atom_numbers; i++)
    {
        PyList_SET_ITEM(numbers, i,
                        PyLong_FromLong(neighbor_list->h_nl[i].atom_numbers));
    }
    return numbers;
}

static PyObject* Neighbor_List_Max_Numbers(PyObject* self, PyObject* args)
{
    if (neighbor_list == NULL)
    {
        return Py_BuildValue("");
    }
    return Py_BuildValue("i", neighbor_list->max_neighbor_numbers);
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
    return Py_BuildValue("i", controller->_MPI_rank);
}

static PyMethodDef SpongeMethods[] = {
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
    PyModule_AddObject(m, "Sponge", m0);
    return m;
}

PLUGIN_API void Set_Backend_Device_Type(int device_type)
{
    dlpack_device_type = static_cast<DLDeviceType>(device_type);
}

PLUGIN_API void Initial(MD_INFORMATION* md, CONTROLLER* ctrl, NEIGHBOR_LIST* nl,
                        void* cv, void* cv_map, void* cv_instance_map)
{
    md_info = md;
    controller = ctrl;
    neighbor_list = nl;
    const int controller_layout_version = read_layout_version(
        controller, SPONGE_PRIPS_CONTROLLER_LAST_MODIFY_DATE_OFFSET);
    if (controller_layout_version != SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION)
    {
        fail_abi_check("controller->last_modify_date=" +
                       std::to_string(controller_layout_version) +
                       ", expected=" +
                       std::to_string(SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION));
    }
    const int md_layout_version = read_layout_version(
        md_info, SPONGE_PRIPS_MD_INFO_LAST_MODIFY_DATE_OFFSET);
    if (md_layout_version != SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION)
    {
        fail_abi_check("md_info->last_modify_date=" +
                       std::to_string(md_layout_version) + ", expected=" +
                       std::to_string(SPONGE_PRIPS_EXPECTED_LAYOUT_VERSION));
    }
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
    PyImport_AppendInittab("prips", &PyInit_prips);
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        fprintf(stderr, "        Python Initialize Failed.\n");
        exit(1);
    }
    else
    {
        controller_printf("        Python Initialized\n");
    }

    wchar_t* temp_args[1] = {(wchar_t*)L"SPONGE"};
    PySys_SetArgv(1, temp_args);
    PyRun_SimpleString(R"XYJ(
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import importlib.util as iu

old_excepthook = sys.excepthook
def new_hook(exctype, value, traceback):
    old_excepthook(exctype, value, traceback)
    exit(1)
sys.excepthook = new_hook

from prips import Sponge

class SpongeDLPackTensor:
    def __init__(self, capsule):
        self.capsule = capsule

    def __dlpack__(self, *args, **kwargs):
        return self.capsule

    def __dlpack_device__(self):
        return (Sponge._backend, 0)

Sponge.SpongeDLPackTensor = SpongeDLPackTensor

class MD_INFORMATION:
    class system_information:
        @property
        def steps(self):
            return Sponge._steps()

    @property
    def atom_numbers(self):
        return Sponge._atom_numbers()

    @property
    def crd(self):
        return Sponge.backend(Sponge.SpongeDLPackTensor(Sponge._coordinate()))

    @property
    def frc(self):
        return Sponge.backend(Sponge.SpongeDLPackTensor(Sponge._force()))

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

    def __init__(self):
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
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
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
    Sponge.backend = _resolve_backend(backend_name)
    Sponge.backend_name = backend_name

Sponge.set_backend = set_backend
Sponge.backend_name = None

)XYJ");

    char buffer[CHAR_LENGTH_MAX];
    sprintf(buffer, "Sponge.fname = r'%s'", py_script_path.c_str());
    PyRun_SimpleString(buffer);
    PyRun_SimpleString(R"XYJ(sponge_pyplugin_path = Path(Sponge.fname)
spec = iu.spec_from_file_location('sponge_pyplugin', sponge_pyplugin_path)
if spec is None:
    Sponge.controller.printf(f"Module '{Sponge.fname}' is not found.")
    sponge_pyplugin = None
else:
    sponge_pyplugin = iu.module_from_spec(spec)
    spec.loader.exec_module(sponge_pyplugin)
    Sponge.controller.printf("        module '%s' imported."%(sponge_pyplugin_path.stem))
)XYJ");

    is_initialized = 1;
    controller_printf("    end initializing pyplugin\n");
}

PLUGIN_API void After_Initial()
{
    if (!is_initialized) return;
    PyRun_SimpleString(R"XYJ(
if Sponge.neighbor_list is None:
    Sponge.neighbor_list = Sponge.NEIGHBOR_LIST()
if hasattr(sponge_pyplugin, "After_Initial"):
    sponge_pyplugin.After_Initial()
    )XYJ");
}

PLUGIN_API void Set_Domain_Information(DOMAIN_INFORMATION* domain_info)
{
    dd = domain_info;
    if (!is_initialized) return;
    PyRun_SimpleString(R"XYJ(
Sponge.dd = Sponge.DOMAIN_INFORMATION()
)XYJ");
}

PLUGIN_API void Calculate_Force()
{
    if (!is_initialized) return;
    PyRun_SimpleString(R"XYJ(
if hasattr(sponge_pyplugin, "Calculate_Force"):
    sponge_pyplugin.Calculate_Force()
    )XYJ");
}

PLUGIN_API void Mdout_Print()
{
    if (!is_initialized) return;
    PyRun_SimpleString(R"XYJ(
if hasattr(sponge_pyplugin, "Mdout_Print"):
    sponge_pyplugin.Mdout_Print()
    )XYJ");
}
