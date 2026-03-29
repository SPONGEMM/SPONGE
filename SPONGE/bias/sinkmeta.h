#ifndef __SINKMETA_CUH__
#define __SINKMETA_CUH__

#include "../collective_variable/collective_variable.h"
#include "../common.h"
#include "../control.h"

struct MetaGrid
{
    int ndim = 0;
    int total_size = 0;
    // 每一维的网格点数（host/device）
    std::vector<int> num_points;
    int* d_num_points = NULL;
    // 每一维是否周期（仅 host 使用）
    std::vector<bool> is_periodic;
    // 网格下界（host/device）
    std::vector<float> lower;
    float* d_lower = NULL;
    // 网格上界（仅 host 使用）
    std::vector<float> upper;
    // 网格步长（host/device）
    std::vector<float> spacing;
    float* d_spacing = NULL;
    // 网格步长倒数（仅 host 使用）
    std::vector<float> inv_spacing;

    std::vector<float> potential;
    float* d_potential = NULL;
    std::vector<float> force;
    float* d_force = NULL;
    std::vector<float> normal_lse;
    float* d_normal_lse = NULL;
    std::vector<float> normal_force;
    float* d_normal_force = NULL;

    void Initial(const std::vector<int>& npts, const std::vector<float>& lo,
                 const std::vector<float>& up,
                 const std::vector<bool>& periodic);
    void Alloc_Device();
    void Sync_To_Device();
    void Sync_To_Host();
    int Get_Flat_Index(const std::vector<float>& values) const;
    std::vector<float> Get_Coordinates(int flat_index) const;
};

struct MetaScatter
{
    int ndim = 0;
    int num_points = 0;
    std::vector<std::vector<float>> coordinates;
    std::vector<float> coordinates_flat;
    float* d_coordinates = NULL;
    std::vector<float> periods;
    float* d_periods = NULL;

    std::vector<float> potential;
    float* d_potential = NULL;
    std::vector<float> force;
    float* d_force = NULL;

    void Initial(const std::vector<int>& npts, const std::vector<float>& period,
                 const std::vector<std::vector<float>>& coor);
    void Alloc_Device();
    void Sync_To_Device();
    void Sync_To_Host();
    int Get_Index(const std::vector<float>& values) const;
    std::vector<int> Get_Neighbor(const std::vector<float>& values,
                                  const float* cutoff) const;
    const std::vector<float>& Get_Coordinate(int index) const;
};

struct META
{
    using Gdata = std::vector<float>;
    using Axis = std::vector<float>;

    // ---- public interface ----
    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int last_modify_date = 20260326;
    CONTROLLER* controller = NULL;

    void Initial(CONTROLLER* controller,
                 COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                 char* module_name = NULL);
    void Do_Metadynamics(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                         LTMatrix3 rcell, int step, int need_potential,
                         int need_pressure, VECTOR* frc, float* d_potential,
                         LTMatrix3* d_virial, float sys_temp);
    void Step_Print(CONTROLLER* controller);
    void Write_Potential(void);
    void Write_Directly(void);

    // ---- internal type ----
    struct Hill
    {
        Hill(const Axis& centers, const Axis& inv_w, const Axis& period,
             const float& theight);
        const Gdata& Calc_Hill(const Axis& values);
        Axis centers_;
        Axis inv_w_;
        Axis periods_;
        float height;
        float potential;
        Axis dx_, df_;
        Gdata tder_;
    };

    // ---- CV configuration ----
    CV_LIST cvs;
    int ndim = 1;
    float* cv_mins;
    float* cv_maxs;
    float* cv_periods;
    float* cv_sigmas;
    int* n_grids;
    std::vector<float> sigmas;
    std::vector<float> periods;
    std::vector<float> cv_deltas;
    float* cutoff;

    // ---- grid / scatter storage ----
    MetaGrid* mgrid = NULL;
    MetaScatter* mscatter = NULL;

    // ---- hill storage ----
    std::vector<Hill> hills;
    Axis vsink;
    int history_freq = 0;

    // ---- flags ----
    bool usegrid = true;
    bool use_scatter = false;
    bool do_borderwall = false;
    bool do_cutoff = false;
    bool do_negative = false;
    bool subhill = false;
    bool kde = false;
    int mask = 0;
    int convmeta = 0;
    int grw = 0;
    bool has_edge_file_input = false;

    // ---- height / well-tempered parameters ----
    float height;
    float height_0;
    float dip = 0.0;
    float welltemp_factor = 1000000000.;
    int is_welltemp = 0;
    float temperature = 300;

    // ---- scatter parameters ----
    int scatter_size = 0;
    std::vector<float*> tcoor;

    // ---- border wall ----
    float border_potential_height = 1000.;
    std::vector<float> border_lower;
    std::vector<float> border_upper;

    // ---- runtime state ----
    float potential_local = 0.;
    float potential_backup = 0.;
    float potential_max = 0.;
    float* Dpotential_local = NULL;
    float sum_max = 0.0;
    float new_max = 0.;
    int max_index;
    float max_force = 0.1;
    float exit_tag;
    Axis est_values_;
    Gdata est_sum_force_;

    // ---- reweighting state ----
    float rct = 0.;
    float rbias = 0.;
    float bias = 0.;
    float minus_beta_f = 1.0;
    float minus_beta_f_plus_v = 0;

    // ---- device buffers ----
    float* d_hill_centers = NULL;
    float* d_hill_inv_w = NULL;
    float* d_hill_periods = NULL;
    float* d_cutoff = NULL;

    // ---- IO ----
    char read_potential_file_name[256];
    char write_potential_file_name[256];
    char write_directly_file_name[256];
    char edge_file_name[256];
    int potential_update_interval;
    int write_information_interval;

    // ---- internal methods ----
    void Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                           int need_potential,
                                           int need_pressure,
                                           float* d_potential,
                                           LTMatrix3* d_virial);
    void Potential_And_Derivative(const int need_potential);
    void Border_Derivative(float* upper, float* lower, float* cutoff,
                           float* Dpotential_local);
    void Set_Grid(CONTROLLER* controller);
    void Estimate(const Axis& values, const bool need_potential,
                  const bool need_force);
    void Add_Potential(float sys_temp, int steps);
    void Get_Height(const Axis& values);
    void Get_Reweighting_Bias(float temp);
    float Calc_V_Shift(const Axis& values);
    float Normalization(const Axis& values, float factor, bool do_normalise);
    void Read_Potential(CONTROLLER* controller);
    bool Read_Edge_File(const char* file_name, std::vector<float>& potential);
    void Pick_Scatter(const std::string fn);
    int Load_Hills(const std::string& fn);
    float Calc_Hill(const Axis& values, const int i);
    float Sum_Hills(int history_freq);
    void Edge_Effect(const int dim, const int size);
    Axis Rotate_Vector(const Axis& tang_vector);
    void Cartesian_To_Path(const Axis& Cartesian_values, Axis& Path_values);
    double Tang_Vector(Gdata& tang_vector, const Axis& values,
                       const Axis& neighbor);
    float Project_To_Path(const Gdata& tang_vector, const Axis& values,
                          const Axis& Cartesian);
};

#endif
