#ifndef __META_CUH__
#define __META_CUH__

#include "../../common.h"
#include "../../control.h"
#include "field/meta_grid.h"
#include "field/switch_function.h"
#include "../../collective_variable/collective_variable.h"

std::vector<float> normalize(const std::vector<float>& v);
std::vector<float> crossProduct(const std::vector<float>& a,
                                const std::vector<float>& b);
float determinant(const std::vector<std::vector<float>>& matrix);

struct META
{
    using Gdata = std::vector<float>;
    using Axis = std::vector<float>;

    char module_name[CHAR_LENGTH_MAX];
    int is_initialized = 0;
    int last_modify_date = 20260326;

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

    struct Hill
    {
        Hill(const Axis& centers, const Axis& inv_w, const Axis& period,
             const float& theight);
        const Gdata& CalcHill(const Axis& values);
        std::vector<GaussianSF> gsf;
        float height;
        float potential;
        Axis dx_, df_;
        Gdata tder_;
    };

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

    MetaGrid* mgrid = nullptr;
    MetaScatter* mscatter = nullptr;

    std::vector<Hill> hills;
    Axis vsink;
    int history_freq = 0;

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
    int catheter = 0;

    float height;
    float height_0;
    float dip = 0.0;

    float welltemp_factor = 1000000000.;
    int is_welltemp = 0;
    float temperature = 300;

    int scatter_size = 0;
    std::vector<float*> tcoor;
    std::vector<float> delta_sigma;
    float sigma_s;
    float sigma_r;

    float border_potential_height = 1000.;
    std::vector<float> border_lower;
    std::vector<float> border_upper;

    float potential_local = 0.;
    float potential_backup = 0.;
    float potential_max = 0.;
    float* Dpotential_local = nullptr;
    float sum_max = 0.0;
    float new_max = 0.;
    int max_index;
    float maxforce = 0.1;
    float exit_tag;

    Axis est_values_;
    Gdata est_sum_force_;

    float rct = 0.;
    float rbias = 0.;
    float bias = 0.;
    float minusBetaF = 1.0;
    float minusBetaFplusV = 0;

    char read_potential_file_name[256];
    char write_potential_file_name[256];
    char write_directly_file_name[256];
    char edge_file_name[256];
    int potential_update_interval;
    int write_information_interval;

    void Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                           int need_potential,
                                           int need_pressure,
                                           float* d_potential,
                                           LTMatrix3* d_virial);
    void Potential_and_derivative(const int need_potential);
    void Border_derivative(float* upper, float* lower, float* cutoff,
                           float* Dpotential_local);
    void Setgrid(CONTROLLER* controller);
    void Estimate(const Axis& values, const bool need_potential,
                  const bool need_force);
    void AddPotential(float sys_temp, int steps);
    void getHeight(const Axis& values);
    void getReweightingBias(float temp);
    float CalcVshift(const Axis& values);
    float Normalization(const Axis& values, float factor, bool do_normalise);
    void Read_Potential(CONTROLLER* controller);

    bool ReadEdgeFile(const char* file_name, std::vector<float>& potential);
    void PickScatter(const std::string fn);
    int LoadHills(const std::string& fn);
    float CalcHill(const Axis& values, const int i);
    float Sumhills(int history_freq);
    void EdgeEffect(const int dim, const int size);

    Axis RotateVector(const Axis& tang_vector, bool do_debug);
    void Cartesian2Path(const Axis& Cartesian_values, Axis& Path_values);
    double TangVector(Gdata& tang_vector, const Axis& values,
                      const Axis& neighbor);
    float ProjectToPath(const Gdata& tang_vector, const Axis& values,
                        const Axis& Cartesian);
};

#endif
