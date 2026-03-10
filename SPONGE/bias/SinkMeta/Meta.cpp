#include "Meta.h"

#ifdef USE_GPU
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>

float exp_added(float a, const float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}
using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

string GetTime(TimePoint& local_time)
{
    local_time = std::chrono::system_clock::now();
    time_t now_time = std::chrono::system_clock::to_time_t(local_time);
    string time_str(asctime(localtime(&now_time)));
    return time_str.substr(0, time_str.find('\n'));
}

string GetDuration(const TimePoint& late_time, const TimePoint& early_time,
                   float& duration)
{
    // Some constants.
    const auto elapsed = late_time - early_time;
    const size_t milliseconds = static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
    const size_t seconds = milliseconds / 1000000L;
    const size_t microseconds = milliseconds % 1000000L;
    const size_t Second2Day = 86400L;
    const size_t Second2Hour = 3600L;
    const size_t Second2Minute = 60L;
    const size_t day = seconds / Second2Day;
    const size_t hour = seconds % Second2Day / Second2Hour;
    const size_t min = seconds % Second2Day % Second2Hour / Second2Minute;
    const size_t second = seconds % Second2Day % Second2Hour % Second2Minute;
    // Calculate duration in second.
    const int BufferSize = 2048;
    char buffer[BufferSize];
    sprintf(buffer,
            "%lu days %lu hours %lu minutes %lu seconds %.1f milliseconds",
            static_cast<unsigned long>(day), static_cast<unsigned long>(hour),
            static_cast<unsigned long>(min), static_cast<unsigned long>(second),
            microseconds * 0.001);
    duration = milliseconds * 0.000001;  // From millisecond to second.
    return string(buffer);
}
struct Zdiff
{
    const float f1, f2;

    Zdiff(float f1, float f2) : f1(f1), f2(f2) {}

    __host__ __device__ float operator()(const float& x) const
    {
        return expf(f1 * x - f2);
    }
};
struct ExpMinusMax
{
    const float maxVal;

    ExpMinusMax(float maxVal) : maxVal(maxVal) {}

    __host__ __device__ float operator()(const float& x) const
    {
        return expf(x - maxVal);
    }
};
#ifdef USE_GPU
using DeviceFloatVector = thrust::device_vector<float>;
using DeviceDoubleVector = thrust::device_vector<double>;
#else
using DeviceFloatVector = std::vector<float>;
using DeviceDoubleVector = std::vector<double>;
#endif

float PartitionFunction(const float factor, float& i_max,
                        const DeviceFloatVector& values)
{
    if (values.empty() || factor < 0.0000001)
    {
        return 0.0;
    }
#ifdef USE_GPU
    i_max = *thrust::max_element(values.begin(), values.end());
    float maxVal = factor * i_max;

    float sum = thrust::transform_reduce(values.begin(), values.end(),
                                         Zdiff(factor, maxVal), 0.0,
                                         thrust::plus<float>());
#else
    i_max = *std::max_element(values.begin(), values.end());
    float maxVal = factor * i_max;
    float sum = 0.0f;
    for (const auto& x : values)
    {
        sum += std::exp(factor * x - maxVal);
    }
#endif
    return maxVal + logf(sum);
}
float logSumExp(const vector<float>& values)
{
    if (values.empty()) return -numeric_limits<float>::infinity();

#ifdef USE_GPU
    DeviceDoubleVector d_values(values.begin(), values.end());
    float maxVal = *thrust::max_element(d_values.begin(), d_values.end());

    float sum = thrust::transform_reduce(d_values.begin(), d_values.end(),
                                         ExpMinusMax(maxVal), 0.0,
                                         thrust::plus<float>());
#else
    float maxVal = *std::max_element(values.begin(), values.end());
    float sum = 0.0f;
    for (const auto& v : values)
    {
        sum += std::exp(v - maxVal);
    }
#endif
    return maxVal + logf(sum);
}

static int split_sentence(const std::string& line,
                          std::vector<std::string>& words)
{
    words.clear();
    std::istringstream iss(line);
    std::string word;
    while (iss >> word)
    {
        words.push_back(word);
    }
    return static_cast<int>(words.size());
}

static int split_sentence(const char* line, std::vector<std::string>& words)
{
    if (line == nullptr) return 0;
    return split_sentence(std::string(line), words);
}
static void Write_CV_Header(FILE* temp_file, int ndim, const CV_LIST& cvs)
{
    for (int i = 0; i < ndim; ++i)
    {
        const char* cv_name = nullptr;
        if (i < static_cast<int>(cvs.size()) && cvs[i] != nullptr &&
            cvs[i]->module_name[0] != '\0')
        {
            cv_name = cvs[i]->module_name;
        }
        if (cv_name != nullptr)
        {
            fprintf(temp_file, "%s\t", cv_name);
        }
        else
        {
            fprintf(temp_file, "cv%d\t", i + 1);
        }
    }
}
void hilllog(const string fn, const vector<float>& hillcenter,
             const vector<float>& hillheight)
{
    if (!fn.empty())
    {
        ofstream hillsout;
        hillsout.open(fn.c_str(), fstream::app);
        hillsout.precision(8);
        for (auto& gauss : hillcenter)
        {
            hillsout << gauss << "\t";
        }
        for (auto& hh : hillheight)
        {
            hillsout << hh << "\t";
        }
        hillsout << endl;
        hillsout.close();
    }
}
void showProgressBar(int progress, int total, int barWidth = 70)
{
    float percentage = static_cast<float>(progress) / total;
    int pos = barWidth * percentage;

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(percentage * 100.0) << " %\r";
    std::cout.flush();
}
// Function to normalize a vector
std::vector<float> normalize(const std::vector<float>& v)
{
    // printf("The size of vector is %d",v.size());
    float norm = 0.;
    for (auto vi : v)
    {
        norm += vi * vi;
    }
    if (norm == 0.0)
        throw std::runtime_error("Zero-length vector cannot be normalized.");
    vector<float> new_v;
    for (int i = 0; i < v.size(); ++i)
    {
        new_v.push_back(v[i] / sqrt(norm));
    }
    return new_v;
}
// Function to compute the cross product of two vectors
std::vector<float> crossProduct(const std::vector<float>& a,
                                const std::vector<float>& b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

// Function to compute the determinant of a square matrix
float determinant(const std::vector<std::vector<float>>& matrix)
{
    int n = matrix.size();
    if (n == 1)
    {
        return matrix[0][0];
    }

    float det = 0;
    for (int i = 0; i < n; ++i)
    {
        std::vector<std::vector<float>> submatrix(n - 1,
                                                  std::vector<float>(n - 1));

        for (int j = 1; j < n; ++j)
        {
            int subCol = 0;
            for (int k = 0; k < n; ++k)
            {
                if (k == i) continue;
                submatrix[j - 1][subCol] = matrix[j][k];
                subCol++;
            }
        }

        float subDet = determinant(submatrix);
        det += (i % 2 == 0 ? 1 : -1) * matrix[0][i] * subDet;
    }

    return det;
}
META::Axis META::RotateVector(const Axis& tang_vector, bool do_debug)
{
    if (do_debug)
    {
        printf("\nRotate Matrix:\n(%f", tang_vector[0]);
        for (int i = 1; i < ndim; ++i)
        {
            printf(" %f", tang_vector[i]);
        }
        printf(")\n");
    }
    vector<float> normal_vector;
    int reference_axis = 0;
    if (fabs(tang_vector[reference_axis]) > 0.99)
    {
        ++reference_axis;
    }
    for (int i = 0; i < ndim; ++i)
    {
        if (i == reference_axis)
        {
            normal_vector.push_back(1.);
        }
        else
        {
            normal_vector.push_back(0.);
        }
    }
    Axis jb;
    if (do_debug) printf("(");
    float i_min = tang_vector[reference_axis];
    float e1 = sqrtf(1 - i_min * i_min);
    float e2 = -i_min / e1;
    for (int i = 0; i < ndim; ++i)
    {
        if (i == reference_axis)
        {
            jb.push_back(e1);
        }
        else
        {
            jb.push_back(tang_vector[i] * e2);
        }
        if (do_debug)
        {
            float jbi = jb[i];
            printf(" %f", jbi);
        }
    }
    if (ndim == 2)  // right-hand rule is apply here!
    {
        vector<vector<float>> determinant_v = vector<Axis>{tang_vector, jb};
        float sign = determinant(determinant_v);
        return Axis{jb[0] * sign, jb[1] * sign};
    }
    return jb;
}
void META::Cartesian2Path(const Axis& Cartesian_values, Axis& Path_values)
{
    double cumulative_s = 0.0;
    double jacobian = 0.501;
    bool do_debug = false;
    Axis values, neighbor;
    Axis tang_vector(ndim, 0.);
    int index = scatter->GetIndex(Cartesian_values);
    if (index < scatter_size - 1)
    {
        values = scatter->GetCoordinate(index);
        neighbor = scatter->GetCoordinate(index + 1);
    }
    else  // last point, backward
    {
        values = scatter->GetCoordinate(index - 1);
        neighbor = scatter->GetCoordinate(index);
    }
    double segment = TangVector(tang_vector, values, neighbor);  // TANGENTIAL;
    double projected_last =
        ProjectToPath(tang_vector, neighbor, Cartesian_values);
    double other_s = cumulative_s + projected_last;
    Path_values.push_back(other_s);
    Axis normal_vector = RotateVector(tang_vector, do_debug);
    Path_values.push_back(
        ProjectToPath(normal_vector, values, Cartesian_values));
    if (ndim == 3)
    {
        Axis binormal_vector =
            normalize(crossProduct(tang_vector, normal_vector));
        Path_values.push_back(
            ProjectToPath(binormal_vector, values, Cartesian_values));
    }
    return;
}
void META::Setgrid(CONTROLLER* controller)  //
{
    std::vector<int> ngrid;
    std::vector<float> lower, upper, periodic;
    std::vector<bool> isperiodic;
    border_upper.resize(ndim);
    border_lower.resize(ndim);
    for (size_t i = 0; i < ndim; ++i)
    {
        ngrid.push_back(n_grids[i]);
        lower.push_back(cv_mins[i]);
        upper.push_back(cv_maxs[i]);
        periodic.push_back(cv_periods[i]);
        isperiodic.push_back(cv_periods[i] > 0 ? true : false);
    }
    normal_force = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
    // normal_factor = new Grid<float>(ngrid, lower, upper, isperiodic);
    normal_lse = new Grid<float>(ngrid, lower, upper, isperiodic);
    normal_force->data_ = vector<Gdata>(normal_force->size(), Gdata(ndim, 0.0));
    // Write Potential need potential_grid!!!!!!!!!!
    potential_grid = new Grid<float>(ngrid, lower, upper, isperiodic);
    potential_grid->data_ = vector<float>(potential_grid->size(), 0.0);
    if (usegrid)
    {
        grid = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
        grid->data_ =
            vector<Gdata>(grid->size(), Gdata(grid->GetDimension(), 0.0));
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        // normal_factor->data_ = vector<float>(normal_factor->size(),
        // normalization);
        normal_lse->data_ =
            vector<float>(normal_lse->size(), log(normalization));
        scatter = nullptr;
        potential_scatter = nullptr;
        // EdgeEffect(ndim, normal_lse->size());
        Sumhills(history_freq);
    }
    else if (use_scatter)
    {
        if (mask > 0)
        {
            grid = new Grid<Gdata>(ngrid, lower, upper, isperiodic);
            grid->data_ =
                vector<Gdata>(grid->size(), Gdata(grid->GetDimension(), 0.0));
        }
        else
        {
            grid = nullptr;
            potential_scatter = nullptr;
        }
        std::vector<int> nscatter;
        int oldsize = 1;  // cvs[0]->point->size();
        for (size_t i = 0; i < ndim; ++i)
        {
            nscatter.push_back(n_grids[i]);
            oldsize *= n_grids[i];
        }
        max_index = floor(scatter_size / 2);  /// initial at the middle point!
        if (oldsize < scatter_size)
        {
            printf("Error, scatter size %d larger than grid %d!\n",
                   scatter_size, oldsize);
            grid = nullptr;
            potential_scatter = nullptr;
            scatter = nullptr;
            potential_scatter = nullptr;
            controller->Throw_SPONGE_Error(spongeErrorConflictingCommand,
                                           "Meta::SetGrid()\n");
            return;
        }
        std::vector<std::vector<float>> coor;  //(oldsize);
        for (size_t j = 0; j < scatter_size; ++j)
        {
            std::vector<float> p;
            for (size_t i = 0; i < ndim; ++i)
            {
                // printf("Coordinate of (%d,%d) is %f\n",i,j,pp);
                p.push_back(tcoor[i][j]);
            }
            coor.push_back(p);
        }
        scatter = new Scatter<Gdata>(nscatter, periodic, coor);
        potential_scatter = new Scatter<float>(nscatter, periodic, coor);
        scatter->data_ =
            vector<Gdata>(scatter_size, Gdata(scatter->GetDimension(), 0.0));
        potential_scatter->data_ =
            vector<float>(potential_scatter->size(), 0.0);
        if (catheter)
        {
            // Method 3 use s and v only
            rotate_v = new Scatter<Gdata>(nscatter, periodic, coor);
            rotate_v->data_ = vector<Gdata>(scatter_size, Gdata(ndim, 0.0));
            for (size_t index = 0; index < scatter_size - 1;
                 ++index)  // : indices)
            {
                Axis values = rotate_v->GetCoordinate(index);
                Axis neighbor = rotate_v->GetCoordinate(index + 1);
                // Gdata data: Tangent Vector normalized as unit vector;
                Gdata& data = rotate_v->data()[index];
                double temp_s = TangVector(data, values, neighbor);
            }
            double temp_sp =
                TangVector(rotate_v->data()[scatter_size - 1],
                           rotate_v->GetCoordinate(scatter_size - 2),
                           rotate_v->GetCoordinate(scatter_size - 1));

            // Method 1 need R matrix
            rotate_matrix = new Scatter<Gdata>(nscatter, periodic, coor);
            rotate_matrix->data_ =
                vector<Gdata>(scatter_size, Gdata(ndim * ndim, 0.0));
            // Rotate matrix is special orthogonal matrix: R^{-1}=R^T
            for (size_t index = 0; index < scatter_size - 1;
                 ++index)  // : indices)
            {
                Gdata data;
                Axis values = rotate_matrix->GetCoordinate(index);
                Axis neighbor = rotate_matrix->GetCoordinate(index + 1);
                Axis tang_vector(ndim, 0.);  // TANGENTIAL
                double segment_s = TangVector(tang_vector, values, neighbor);
                for (auto t : tang_vector)
                {
                    data.push_back(t);
                }
                Axis normal_vector = RotateVector(tang_vector, false);
                for (auto n : normal_vector)
                {
                    data.push_back(n);
                }
                if (ndim == 3)
                {
                    Axis binormal_vector =
                        normalize(crossProduct(tang_vector, normal_vector));
                    for (auto b : binormal_vector)
                    {
                        data.push_back(b);
                    }
                }
                /*
                if (!CheckOrthogonal(data, ndim))
                {
                    controller->Throw_SPONGE_Error(spongeErrorValueErrorCommand,
                "METAD::SetGrid", "Rotate matrix is no an orthogonal matrix");
                }*/
                rotate_matrix->data()[index] = data;
            }
            rotate_matrix->data()[scatter_size - 1] =
                rotate_matrix->data()[scatter_size - 2];
        }
        // calculate normal_factor and print
        EdgeEffect(1, scatter_size);
        Sumhills(history_freq);
    }
    else
    {
        printf("Warning! No grid version is very slow\n");
        grid = nullptr;
        potential_scatter = nullptr;
        scatter = nullptr;
        potential_scatter = nullptr;
    }
}
META::Hill::Hill(const Axis& centers, const Axis& inv_w, const Axis& period,
                 const float& theight)
    : height(theight)
{
    for (int i = 0; i < centers.size(); ++i)
    {
        gsf.push_back(GaussianSF(centers[i], inv_w[i], period[i]));
    }
}
META::Gdata META::Hill::CalcHill(const Axis& values)
{
    const size_t& n = values.size();
    Axis dx(n, 0.0), df(n, 1.0);
    // Compute difference between grid point and current val.
    for (size_t i = 0; i < n; ++i)
    {
        GaussianSF g = gsf[i];
        dx[i] = g.Evaluate(values[i], df[i]);
    }

    Gdata tder(n, 1.0);  // Force
    potential = 1.0;     // hill.potential
    // Compute derivative.
    for (size_t i = 0; i < n; ++i)
    {
        potential *= dx[i];
        for (size_t j = 0; j < n; ++j)
        {
            if (j != i)
            {
                tder[i] *= dx[j];
            }
            else
            {
                tder[i] *= df[j];
            }
        }
    }
    return tder;
}
float META::ProjectToPath(const Gdata& tang_vector, const Axis& values,
                          const Axis& Cartesian)
{
    float projected_s = 0.;
    for (int i = 0; i < ndim; ++i)
    {
        projected_s += (Cartesian[i] - values[i]) * tang_vector[i];
    }
    return projected_s;
}
double META::TangVector(Gdata& tang_vector, const Axis& values,
                        const Axis& neighbor)
{
    double square = 0;
    for (int i = 0; i < ndim; ++i)
    {
        double distance = neighbor[i] - values[i];
        tang_vector[i] = distance;
        square += distance * distance;
    }
    double segment_s = sqrt(square);
    for (int i = 0; i < ndim; ++i)
    {
        tang_vector[i] /= segment_s;
    }
    return segment_s;
}
bool META::ReadEdgeFile(const char* file_name, vector<float>& potential)
{
    FILE* temp_file = NULL;
    int grid_size = 0;
    bool readsuccess = true;
    int total = normal_force->size();
    vector<Gdata> force_from_file;
    printf("Reading %d grid of edge effect\n", total);
    temp_file = fopen(file_name, "r+");
    if (temp_file != NULL)
    {
        fseek(temp_file, 0, SEEK_END);

        if (ftell(temp_file) == 0)
        {
            printf("Edge file %s is empty\n", file_name);
        }
        else
        {
            Open_File_Safely(&temp_file, file_name, "r");
            char temp_char[256] = " ";  // empty but not nullptr
            int scanf_ret = 0;
            char* grid_val = temp_char;
            while (grid_val != NULL)
            {
                grid_val = fgets(temp_char, 256, temp_file);  // grid line
                std::vector<std::string> words;
                int nwords = split_sentence(temp_char, words);
                Gdata force(ndim, 0.);
                potential.push_back(
                    logf(std::stof(words[ndim])));  /// log sum exp!!!!!
                if (nwords == 1 + ndim * 2)
                {
                    for (int i = 0; i < ndim; ++i)
                    {
                        force[i] = std::stof(words[1 + ndim + i]);
                    }
                }
                else
                {
                    printf(
                        "Error reading Edge file %s, line %d of %d:\n Format "
                        "should have %d while only %d have been read\n",
                        file_name, grid_size, total, 1 + 2 * ndim, nwords);
                    return false;
                }
                ++grid_size;
                force_from_file.push_back(force);
            }
        }
        fclose(temp_file);
    }
    if (grid_size - 1 != total)
    {
        printf("Error reading Edge file %s, line %d of %d\n", file_name,
               grid_size, total);
        return false;
    }
    // Now apply to the Normalization factor&force.
    // normal_factor->data_ = potential;
    normal_lse->data_ = potential;
    sum_max = *std::max_element(potential.begin(), potential.end());
    if (scatter_size < total && do_negative)
    {
        int it_progress = 0;
        for (Grid<Gdata>::iterator g_iter = normal_force->begin();
             g_iter != normal_force->end(); ++g_iter)
        {
            *g_iter = force_from_file[it_progress];
            ++it_progress;
        }
    }
    return readsuccess;
}
// Load hills from output file.
int META::LoadHills(const string& fn)  //, const vector<double>& widths)
{
    ifstream hillsin(fn.c_str(), ios::in);
    if (!hillsin.is_open())
    {
        // ErrorTermination("Cannot open Hills file \"%s\".", fn.c_str());
        printf("Warning, No record of hills\n");
        return 0;
    }
    const string file_content((istreambuf_iterator<char>(hillsin)),
                              istreambuf_iterator<char>());
    hillsin.close();

    const int& cvsize = ndim;
    istringstream iss(file_content);
    string tstr;
    vector<string> words;
    int num_hills = 0;
    while (getline(iss, tstr, '\n'))
    {
        Axis values;
        split_sentence(tstr, words);
        if (words.size() < cvsize + 1)
        {
            printf("The format of Hills file \"%s\" near \"%s\" is wrong.",
                   fn.c_str(), tstr.c_str());
        }
        for (int i = 0; i < cvsize; ++i)
        {
            float center = stof(words[i]);
            /*if (!Parser::ParseRealNumber(words[i], center))
            {
                printf("The format of Hills file \"%s\" near \"%s\" is wrong.",
            fn.c_str(), words[i].c_str());
            }
            else*/
            {
                values.push_back(center);
            }
        }
        float theight = stof(words[cvsize]);  // well-tempered height!
        /*if (!Parser::ParseRealNumber(words[cvsize], theight))
        {
            printf("The format of Hills file \"%s\" near \"%s\" is wrong.",
        fn.c_str(), words[cvsize].c_str());
        }*/
        if (do_negative || use_scatter)
        {
            float p_max = stof(words[cvsize + 1]);
            int p_id = stoi(words[cvsize + 2]);
            if (p_id < scatter_size)
            {
                float Phi_s = expf(normal_lse->at(values));
                float vshift =
                    (p_max + dip * CONSTANT_kB * temperature) *
                    expf(-normal_lse->at(scatter->GetCoordinate(p_id)));
                vsink.push_back(Phi_s * vshift);
            }
            else
            {
                printf("Error reading sink projecting id: %d\n", p_id);
                return -1;
            }
        }
        Hill newhill = Hill(values, sigmas, periods, theight);
        hills.push_back(newhill);
        ++num_hills;
    }
    return num_hills;
}
float META::CalcHill(const Axis& values, const int i)
{
    float potential = 0;
    // Axis prefactor;
    for (int j = 0; j < i; ++j)
    {
        Hill& hill = hills[j];
        // second loop, debug only
        Gdata tder = hill.CalcHill(values);
        potential += hill.potential * hill.height;
        /*
        float pregauss = 0.;
        Axis neighbor; // = potential_scatter->GetCoordinate(index);
        for (auto &gauss : hill.gsf)
        {
            neighbor.push_back(gauss.GetCenter());
        }
        for (int i = 0; i < ndim; ++i)
        {
            float diff = (values[i] - neighbor[i]);
            if (periods[i] != 0.0)
            {
                diff -= roundf(diff / periods[i]) * periods[i];
            }
            float distance = diff * sigmas[i];
            pregauss -= 0.5 * distance * distance;
        }
        prefactor.push_back(pregauss + logf(hill.height));
        */
    }
    return potential;  // expf(logSumExp(prefactor));
}
float META::Sumhills(int history_freq)  // const vector<float> heights)
{
    if (history_freq == 0)
    {
        return 0.;
    }
    TimePoint start_time, end_time;
    float duration;
    GetTime(start_time);
    int nhills = LoadHills("myhill.log");  // hills.size();
    FILE* temp_file = NULL;
    printf("\r\nLoad hills file successfully, now calculate RCT!!!\n");
    Open_File_Safely(&temp_file, "history.log", "w");
    // first loop: history
    float old_potential;
    minusBetaFplusV =
        1. / (welltemp_factor - 1.) / CONSTANT_kB / temperature;  /// 300K
    minusBetaF = welltemp_factor * minusBetaFplusV;
    float total_gputime = 0.;
    for (int i = 0; i < nhills; ++i)
    {
        showProgressBar(i, nhills);
        Hill& hill = hills[i];
        Axis values;
        for (auto& gauss : hill.gsf)
        {
            values.push_back(gauss.GetCenter());
        }
        old_potential = CalcHill(values, i);
        if (history_freq != 0 && (i % history_freq == 0))
        {
            potential_grid->data_ = vector<float>(potential_grid->size(), 0.);
            TimePoint tstart, tend;
            float gputime;
            GetTime(tstart);
            // RCT calculation
            DeviceFloatVector d;
            if (use_scatter)
            {
                for (int iter = 0; iter < scatter_size; ++iter)
                {
                    potential_scatter->data_[iter] =
                        CalcHill(potential_scatter->GetCoordinate(iter), i);  //
                }
#ifdef USE_GPU
                d = DeviceFloatVector(potential_scatter->data_.begin(),
                                      potential_scatter->data_.end());
#else
                d = potential_scatter->data_;
#endif
            }
            else  // use grid
            {
                for (Grid<float>::iterator g_iter = potential_grid->begin();
                     g_iter != potential_grid->end(); ++g_iter)
                {
                    *g_iter = CalcHill(g_iter.coordinates(), i);
                } /*
     for (int j = 0; j < i; ++j)
     {
         Hill &hill_ = hills[j];
         //second loop, debug only
                 for (Grid<float>::iterator g_iter = potential_grid->begin();
     g_iter != potential_grid->end(); ++g_iter)
                 {
                     Gdata tder = hill_.CalcHill(g_iter.coordinates());
                     *g_iter += hill_.potential * hill_.height;
                 }
     }*/
#ifdef USE_GPU
                d = DeviceFloatVector(potential_grid->data_.begin(),
                                      potential_grid->data_.end());
#else
                d = potential_grid->data_;
#endif
            }
            GetTime(tend);
            GetDuration(tend, tstart, gputime);
            total_gputime += gputime;
            float Z_0 = PartitionFunction(minusBetaF, potential_max, d);
            float Z_V = PartitionFunction(minusBetaFplusV, potential_max, d);
            rct = CONSTANT_kB * temperature * (Z_0 - Z_V);
            float rbias = old_potential - rct;
            fprintf(temp_file, "%f\t%f\t%f\t%f\n", old_potential, rbias, rct,
                    vsink[i]);
        }
    }
    fclose(temp_file);
    GetTime(end_time);
    GetDuration(end_time, start_time, duration);
    int hours = floor(duration / 3600);
    float nohour = duration - 3600 * hours;
    int mins = floor(nohour / 60);
    float seconds = nohour - 60 * mins;
    printf(
        "The RBIAS & RCT calculation cost %f of %f seconds: %d hour %d min %f "
        "second\n",
        total_gputime / duration, duration, hours, mins, seconds);
    return old_potential;
}
void META::EdgeEffect(const int dim, const int scatter_size)
{
    vector<float> potential_from_file;
    const char* file_name = "sumhill.log";

    int total = normal_lse->size();
    if (scatter_size == total)
    {
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi * 2);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
        // normal_factor->data_ = vector<float>(normal_factor->size(),
        // normalization);
        normal_lse->data_ =
            vector<float>(normal_lse->size(), log(normalization));
    }
    if (!ReadEdgeFile(file_name, potential_from_file))
    {
        int it_progress = 0;
        printf("Calculation the %d grid of edge effect\n", total);
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, file_name, "w+");
        // default 1-dimensional scatter, maybe slow for 3D-mask!
        Axis esigmas;
        float adjust_factor = 1.0;
        if (convmeta)
        {
            // adjust_factor = sqrtf(2.0);
        }
        for (int i = 0; i < ndim; ++i)
        {
            esigmas.push_back(sigmas[i] * adjust_factor);
        }
        for (Grid<float>::iterator g_iter = normal_lse->begin();
             g_iter != normal_lse->end(); ++g_iter)
        {
            showProgressBar(++it_progress, total);
            const Axis values = g_iter.coordinates();
            double sum_hills = 0.;
            vector<float> prefactor;
            if (catheter)
            {
                float R = sqrtf(sigma_s * sigma_s -
                                2.0 * delta_sigma[scatter->GetIndex(values)]);
                for (int i = 0; i < ndim; ++i)
                {
                    esigmas[i] = R;
                }
            }
            Hill hill = Hill(values, esigmas, periods, 1.0);
            vector<int> indices;
            if (do_cutoff)
            {
                indices = potential_scatter->GetNeighbor(values, cutoff);
            }
            else
            {
                indices = vector<int>(scatter_size);
                iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                Axis neighbor = potential_scatter->GetCoordinate(index);
                float pregauss = 0.;
                for (int i = 0; i < ndim; ++i)
                {
                    float diff = (values[i] - neighbor[i]);
                    if (periods[i] != 0.0)
                    {
                        diff -= roundf(diff / periods[i]) * periods[i];
                    }
                    float distance = diff * esigmas[i];
                    pregauss -= 0.5 * distance * distance;
                }
                if (do_negative)  // && !subhill)
                {
                    Gdata tder = hill.CalcHill(neighbor);
                    float hill_potential = hill.potential;
                    Gdata& data = normal_force->at(values);
                    if (catheter)  // also need logsumexp !!!!!
                    {
                        Gdata& v = rotate_v->data()[index];
                        float s = ProjectToPath(v, values, neighbor);
                        float dss = delta_sigma[index] * s * s;
                        pregauss -= dss;
                        float s_shrink = expf(-dss);
                        hill_potential *= s_shrink;
                        for (int i = 0; i < ndim; ++i)
                        {
                            float dx = values[i] - neighbor[i];
                            if (periods[i] != 0.0)
                            {
                                dx -= roundf(dx / periods[i]) * periods[i];
                            }
                            for (int j = 0; j < ndim; ++j)
                            {
                                float partial =
                                    2 * delta_sigma[index] * v[i] * v[j] * dx;
                                data[i] += partial * hill_potential;
                            }
                            data[i] += tder[i] * s_shrink;
                        }
                    }
                    else
                    {
                        for (int i = 0; i < ndim; ++i)
                        {
                            data[i] += tder[i];
                        }
                    }
                }
                prefactor.push_back(pregauss);
            }

            float logsumhills = logSumExp(prefactor);
            sum_max = fmaxf(logsumhills, sum_max);  ///<
            vector<float> sum_potential(1 + ndim, expf(logsumhills));
            // normal_lse->at(values) = logsumhills;
            *g_iter = logsumhills;
            if (do_negative)
            {  // print force!
                Gdata& data = normal_force->at(values);
                for (int i = 0; i < ndim; ++i)
                {
                    sum_potential[i + 1] = data[i];
                }
            }
            // hilllog(file_name, values, sum_potential); //(*g_iter));
            for (auto& v : values)
            {
                fprintf(temp_file, "%f\t", v);
            }
            for (auto& s : sum_potential)
            {
                fprintf(temp_file, "%f\t", s);
            }
            fprintf(temp_file, "\n");
        }
        fclose(temp_file);
    }
    if (dim == 1)
    {
        PickScatter("lnbias.dat", normal_lse);  //,sum_max);
    }
}

void META::PickScatter(const string fn, Grid<float>* data)
{
    ofstream hillsout;
    hillsout.open(fn.c_str(), fstream::out);
    hillsout.precision(8);
    // float sum_max = sqrtf(1.0) / sum_normal;
    // float ln_max = sum_max * logf(sum_max);
    for (int index = 0; index < scatter_size; ++index)  //  indices)
    {
        Axis neighbor = potential_scatter->GetCoordinate(index);
        float lnbias = data->at(neighbor);
        // hillsout << index<< "\t" << sum_max*logf(lnbias)-ln_max << "\t" <<
        // lnbias-sum_max << endl;
        hillsout << index << "\t" << lnbias << "\t" << exp(lnbias) << endl;
    }
    hillsout.close();
}
float META::Normalization(const Axis& values, float factor, bool do_normalise)
{
    if (do_normalise)
    {
        if (usegrid)
        {
            return factor * expf(-normal_lse->data_[0]);
        }
        if (convmeta)
        {
            return factor *
                   expf(-normal_lse->at(scatter->GetCoordinate(max_index)));
        }
        else
        {
            return factor * expf(-normal_lse->at(values));
        }
    }
    else
    {
        return factor;
    }
}
void META::Estimate(const Axis& values, const bool need_potential,
                    const bool need_force)
{
    potential_local = 0;
    potential_backup = 0;

    float shift = potential_max + dip * CONSTANT_kB * temperature;
    if (do_negative)
    {
        if (grw)
        {
            shift = (welltemp_factor + dip) * CONSTANT_kB * temperature;
        }
        new_max = Normalization(values, shift, true);
    }
    float force_max = 0.0;  // Add the edge's force
    float normalforce_sum = 0.0;
    Gdata sum_force(ndim, 0.);
    for (size_t i = 0; i < ndim; ++i)
    {
        Dpotential_local[i] = 0.0;
        force_max += fabs(normal_force->at(values)[i]);
    }
    if (force_max > maxforce && need_force && mask)
    {
        exit_tag += 1.0;
    }
    // Axis aaaaa =
    // potential_grid->GetCoordinates(potential_grid->GetIndices(values));
    Hill hill = Hill(values, sigmas, periods, 1.0);
    if (use_scatter)
    {
        if (subhill)
        {
            vector<Gdata> derivative;
            vector<int> indices;
            if (do_cutoff)
            {
                indices = potential_scatter->GetNeighbor(values, cutoff);
            }
            else
            {
                indices = vector<int>(scatter_size);
                iota(indices.begin(), indices.end(), 0);
            }
            for (auto index : indices)
            {
                Axis neighbor = potential_scatter->GetCoordinate(index);
                Gdata tder = hill.CalcHill(neighbor);
                normalforce_sum += hill.potential;
                float factor = (mask > 0) ? potential_grid->at(neighbor)
                                          : potential_scatter->data()[index];
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        sum_force[i] += tder[i];
                        Dpotential_local[i] -= (factor)*tder[i];
                    }
                }
                // ratio =
                // normal_lse->at(potential_scatter->GetCoordinate(potential_scatter->GetIndex(neighbor)))-normal_lse->at(neighbor);
                // potential_local += (factor- new_max) * hill.potential;
                potential_backup += factor * hill.potential;
            }
            /*
            if (convmeta)
            {
                potential_local = potential_backup - shift *
            expf(normal_lse->at(scatter->GetCoordinate(values))-normal_lse->at(scatter->GetCoordinate(max_index)));
            /// sum_max;
            }
            else
            {
                potential_local = potential_backup - shift *
            (normal_lse->at(scatter->GetCoordinate(values))-  sum_max);
            }*/
        }
        else
        {
            potential_backup = (mask > 0) ? potential_grid->at(values)
                                          : potential_scatter->at(values);
            potential_local = potential_backup - CalcVshift(values);
            if (need_force)
            {
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        (mask > 0)
                            ? grid->at(values)[i]
                            : scatter->at(values)
                                  [i];  // -
                                        // potential_local*normal_force->at(values)[i];
                }
            }
        }
        /*if (do_borderwall)
        {
            vector<float> coordinate = potential_scatter->GetCoordinate(values);
            for (size_t i = 0; i < ndim; ++i)
            {
                border_upper[i] = coordinate[i] + 0.2 / cv_sigmas[i];
                border_lower[i] = coordinate[i] - 0.2 / cv_sigmas[i];
            }
        }*/
    }
    else if (usegrid)
    {
        if (subhill)
        {
            Axis vminus, vplus;
            for (size_t i = 0; i < ndim; ++i)
            {
                float lower = values[i] - cutoff[i];
                float upper = values[i] + cutoff[i] + 0.000001;
                if (periods[i] > 0)
                {
                    vminus.push_back(
                        lower);  // - round(lower / periods[i]) * periods[i]);
                    vplus.push_back(
                        upper);  // - round(lower / periods[i]) * periods[i]);
                }
                else
                {
                    vminus.push_back(std::fmax(lower, cv_mins[i]));
                    vplus.push_back(std::fmin(upper, cv_maxs[i]));
                }
            }
            Axis loop_flag = vminus;
            int index = 0;
            while (index >= 0)
            {
                //++sum_count;
                Gdata tder = hill.CalcHill(loop_flag);
                float factor = potential_grid->at(loop_flag);
                potential_backup += factor * hill.potential;
                // potential_local += (factor-new_max) * hill.potential;
                if (need_force)
                {
                    for (size_t i = 0; i < ndim; ++i)
                    {
                        Dpotential_local[i] -= (factor - new_max) * tder[i];
                    }
                }
                // another dimension!
                index = ndim - 1;
                while (index >= 0)
                {
                    loop_flag[index] += cv_deltas[index];
                    if (loop_flag[index] > vplus[index])
                    {
                        loop_flag[index] = vminus[index];
                        --index;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else
        {
            potential_backup = potential_grid->at(values);
            if (need_force)
            {
                for (int i = 0; i < cvs.size(); ++i)
                {
                    Dpotential_local[i] +=
                        grid->at(values)
                            [i];  // -
                                  // potential_local*normal_force->at(values)[i];;
                }
            }
        }
        if (do_borderwall)
        {
            for (size_t i = 0; i < ndim; ++i)
            {
                border_upper[i] = cv_maxs[i] - cutoff[i];  // 1.0/cv_sigmas[i] ;
                border_lower[i] = cv_mins[i] + cutoff[i];  // 1.0/cv_sigmas[i] ;
            }
        }
    }
    if (need_potential)
    {
        potential_local = potential_backup - CalcVshift(values);
    }
    if (need_force)  // && !subhill)
    {
        if (subhill)
        {
            float f0 = new_max * normal_force->at(values)[0];
            if (convmeta)
            {
                new_max =
                    shift *
                    expf(-normal_lse->at(scatter->GetCoordinate(max_index)));
            }
            else
            {
                new_max = shift / normalforce_sum;
            }
            float f1 = new_max * sum_force[0];
            if (fabs(f0 - f1) > shift)
            {
                printf("The shift, kde & histogram:%f: %f vs %f\n", shift, f1,
                       f0);
            }
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * sum_force[i];
            }
        }
        else
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += new_max * normal_force->at(values)[i];
            }
        }
    }
    /* original meta without grid!!!!
    for (auto &hill_ : hills)
    {
        Gdata tder = hill_.CalcHill(values);
        if (need_force)
        {
            for (int i = 0; i < cvs.size(); ++i)
            {
                Dpotential_local[i] += hill_.height * tder[i];
            }
        }
        if (need_potential)
        {
            potential_local += hill_.potential * hill_.height;
        }
    }
    potential_backup = potential_local;
    */
    return;
}
void META::Initial(CONTROLLER* controller,
                   COLLECTIVE_VARIABLE_CONTROLLER* cv_controller,
                   char* module_name)
{
    if (module_name == NULL)
    {
        strcpy(this->module_name, "meta");
    }
    else
    {
        strcpy(this->module_name, module_name);
    }
    if (!cv_controller->Command_Exist(this->module_name, "CV"))
    {
        controller->printf("META IS NOT INITIALIZED\n\n");
        return;
    }
    else
    {
        std::vector<std::string> cv_str =
            cv_controller->Ask_For_String_Parameter(this->module_name, "CV",
                                                    ndim);
        std::string cvv =
            std::accumulate(cv_str.begin(), cv_str.end(), std::string(""));
        printf("%s contains %d dimension META\n", cvv.c_str(), ndim);
    }
    if (cv_controller->Command_Exist(this->module_name, "dip"))
    {
        dip = cv_controller->Ask_For_Float_Parameter(this->module_name, "dip",
                                                     1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "welltemp_factor"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "welltemp_factor");
        welltemp_factor = temp_value[0];
        free(temp_value);
        if (welltemp_factor > 1)
        {
            is_welltemp = 1;
        }
        else
        {
            printf("\tWell-tempered Factor must larger than 1!\n");
            getchar();
        }
    }
    cvs = cv_controller->Ask_For_CV(this->module_name, -1);
    if (cv_controller->Command_Exist(this->module_name, "Ndim"))
    {
        ndim = *cv_controller->Ask_For_Int_Parameter(this->module_name, "Ndim");
        if (ndim != cvs.size())
        {
            controller->printf("%d D-META IS NOT CONSISTANT CV size %d\n\n",
                               ndim, cvs.size());
            return;
        }
    }
    else
    {
        ndim = cvs.size();
    }
    controller->printf("START INITIALIZING %dD-META:\n", ndim);
    // initialize Dpotential_local
    Malloc_Safely((void**)&Dpotential_local, sizeof(float) * ndim);
    sprintf(read_potential_file_name, "Meta_Potential.txt");
    sprintf(write_potential_file_name, "Meta_Potential.txt");
    if (controller->Command_Exist("default_in_file_prefix"))
    {
        sprintf(read_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_in_file_prefix"));
    }
    else
    {
        sprintf(read_potential_file_name, "Meta_Potential.txt");
    }
    if (controller->Command_Exist("default_out_file_prefix"))
    {
        sprintf(write_potential_file_name, "%s_Meta_Potential.txt",
                controller->Command("default_out_file_prefix"));
    }
    else
    {
        sprintf(write_potential_file_name, "Meta_Potential.txt");
    }
    sprintf(write_directly_file_name, "Meta_directly.txt");
    if (cv_controller->Command_Exist(this->module_name, "subhill"))
    {
        subhill = true;
        printf("reading subhill for meta: 1\n");
    }
    if (cv_controller->Command_Exist(this->module_name, "kde"))
    {
        int kde_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "kde", 1)[0];
        if (kde_dim)
        {
            kde = true;
            subhill = true;
            printf("reading kde's subhill for meta: %d\n", kde_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "mask"))
    {
        mask = cv_controller->Ask_For_Int_Parameter(this->module_name, "mask",
                                                    1)[0];
        if (mask)
        {
            printf("reading mask dimension meta: %d\n", mask);
            if (cv_controller->Command_Exist(this->module_name, "maxforce"))
            {
                maxforce = cv_controller->Ask_For_Float_Parameter(
                    this->module_name, "maxforce", 1)[0];
            }
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sink"))
    {
        int sub_dim = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                           "sink", 1)[0];
        if (sub_dim > 0)
        {
            do_negative = true;
            printf("reading sink/submarine dimension for meta: %d\n", sub_dim);
        }
    }
    if (cv_controller->Command_Exist(this->module_name, "sumhill_freq"))
    {
        history_freq = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "sumhill_freq", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "catheter"))
    {
        usegrid = false;
        use_scatter = true;
        do_negative = true;
        catheter =
            3;  // cv_controller->Ask_For_Int_Parameter(this->module_name,
                // "catheter", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "convmeta"))
    {
        // use_scatter = true;
        do_negative = true;
        convmeta = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                        "convmeta", 1)[0];
    }
    if (cv_controller->Command_Exist(this->module_name, "grw"))
    {
        // use_scatter = true;
        do_negative = true;
        grw = cv_controller->Ask_For_Int_Parameter(this->module_name, "grw",
                                                   1)[0];
    }
    cv_periods = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_period", cvs.size(), 1, false);
    cv_sigmas = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                       "CV_sigma", cvs.size());
    cutoff = cv_controller->Ask_For_Float_Parameter(
        this->module_name, "CV_sigma", cvs.size(), 1, false, 0., -3);
    if (cv_controller->Command_Exist(this->module_name, "cutoff"))
    {
        do_cutoff = true;
        cutoff = cv_controller->Ask_For_Float_Parameter(this->module_name,
                                                        "cutoff", cvs.size());
    }
    for (int i = 0; i < cvs.size(); i++)
    {
        if (cv_sigmas[i] <= 0)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                "CV_sigma should always be greater than 0");
        }
        if (!do_cutoff)
        {
            cutoff[i] = 3 * cv_sigmas[i];
        }
        if (kde)
        {
            cv_sigmas[i] =
                1.414f / cv_sigmas[i];  ///  inverted sigma!!!!!!!!!!!!!!!!!!!
        }
        else
        {
            cv_sigmas[i] =
                1.0 / cv_sigmas[i];  /// inverted sigma!!!!!!!!!!!!!!!!!!!
        }
    }
    float sqrtpi = sqrtf(CONSTANT_Pi * 2);
    for (int i = 0; i < ndim; i++)
    {
        sigmas.push_back(cv_sigmas[i]);
        periods.push_back(cv_periods[i]);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_in_file"))
    {
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else if (cv_controller->Command_Exist(this->module_name, "scatter_in_file"))
    {
        usegrid = false;
        use_scatter = true;
        printf("Use %d scatter point for CV!\n", scatter_size);
        strcpy(read_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "scatter_in_file")[0]
                   .c_str());
        if (usegrid || use_scatter)
        {
            Read_Potential(controller);
        }
    }
    else
    {
        if (cv_controller->Command_Exist(this->module_name, "scatter"))
        {
            scatter_size = *(cv_controller->Ask_For_Int_Parameter(
                this->module_name, "scatter", 1));
            if (scatter_size > 0)
            {
                usegrid = false;
                use_scatter = true;
                printf("Use %d scatter point for CV!\n", scatter_size);
                for (int i = 0; i < cvs.size(); i++)
                {
                    tcoor.push_back(cv_controller->Ask_For_Float_Parameter(
                        cvs[i]->module_name, "CV_point", scatter_size, 1,
                        false));
                }
            }
            else
            {
                printf("Not using scatter point for CV\n");
                use_scatter = false;
            }
        }

        cv_mins = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_minimal", cvs.size());
        cv_maxs = cv_controller->Ask_For_Float_Parameter(
            this->module_name, "CV_maximum", cvs.size());
        n_grids = cv_controller->Ask_For_Int_Parameter(this->module_name,
                                                       "CV_grid", cvs.size());
        for (int i = 0; i < cvs.size(); ++i)
        {
            if (cv_maxs[i] <= cv_mins[i])
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_maximum should always be greater than CV_minimal");
            }
            if (n_grids[i] <= 1)
            {
                controller->Throw_SPONGE_Error(
                    spongeErrorValueErrorCommand, "METADYNAMICS::Initial",
                    "CV_grid should always be greater than 1");
            }
            cv_deltas.push_back((cv_maxs[i] - cv_mins[i]) / n_grids[i]);
        }
        Setgrid(controller);
    }
    height_0 = 1.0;
    if (cv_controller->Command_Exist(this->module_name, "height"))
    {
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "height");
        height_0 = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "wall_height"))
    {
        do_borderwall = true;
        float* temp_value =
            cv_controller->Ask_For_Float_Parameter("meta", "wall_height");
        border_potential_height = temp_value[0];
        free(temp_value);
    }
    if (cv_controller->Command_Exist(this->module_name, "potential_out_file"))
    {
        strcpy(write_potential_file_name,
               cv_controller
                   ->Ask_For_String_Parameter(this->module_name,
                                              "potential_out_file")[0]
                   .c_str());
    }
    bool has_potential_update_interval = false;
    if (cv_controller->Command_Exist(this->module_name,
                                     "potential_update_interval"))
    {
        int* temp_value = cv_controller->Ask_For_Int_Parameter(
            this->module_name, "potential_update_interval");
        potential_update_interval = temp_value[0];
        has_potential_update_interval = true;
        free(temp_value);
    }
    if (controller->Command_Exist("write_information_interval"))
    {
        write_information_interval =
            atoi(controller->Command("write_information_interval"));
    }
    else
    {
        write_information_interval = 1000;
    }
    if (write_information_interval <= 0)
    {
        write_information_interval = 1000;
    }
    if (!has_potential_update_interval)
    {
        controller->printf(
            "    Potential update interval is set to "
            "write_information_interval by default\n");
        potential_update_interval = write_information_interval;
    }
    if (potential_update_interval <= 0)
    {
        potential_update_interval = 1000;
    }
    controller->Step_Print_Initial("meta", "%f");
    controller->Step_Print_Initial("rbias", "%f");
    controller->Step_Print_Initial("rct", "%f");
    controller->printf("    potential output file: %s\n",
                       write_potential_file_name);
    is_initialized = 1;
    controller->printf("END INITIALIZING META\n\n");
}

void META::Write_Potential(void)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_potential_file_name, "w");
        /*fprintf(temp_file, "%dD-Meta X %f\n", ndim, sum_normal);
        for (int i = 0; i < ndim; ++i)
        {
            fprintf(temp_file, "%f\t%f\t%f\n", cv_mins[i], cv_maxs[i],
        cv_deltas[i]);
        }
        int gridsize = 1;
        for (int i = 0; i < ndim; ++i)
        {
            int num_grid = round((cv_maxs[i] - cv_mins[i]) / cv_deltas[i]);
            if (periods[i] == 0)
            {
                ++num_grid; //  numpoint+1 for non-periodic condition
            }
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        fprintf(temp_file, "%d\n", gridsize);
            */
        if (subhill || (!usegrid && !use_scatter))
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_local\tpotential_backup");
            if (!kde)
            {
                fprintf(temp_file, "\tpotential_raw");
            }
            fprintf(temp_file, "\n");
            vector<float> loop_flag(ndim, 0);
            vector<float> loop_floor(ndim, 0);
            for (int i = 0; i < ndim; ++i)
            {
                loop_floor[i] = cv_mins[i] + 0.5 * cv_deltas[i];
                loop_flag[i] = loop_floor[i];
            }
            int i = 0;
            while (i >= 0)
            {
                Estimate(loop_flag, true, false);  // get potential
                ostringstream ss;
                for (const float& v : loop_flag)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f", ss.str().c_str(),
                        potential_local, potential_backup);
                if (!kde)
                {
                    if (potential_grid != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                potential_grid->at(loop_flag));
                    }
                    else if (potential_scatter != nullptr)
                    {
                        fprintf(temp_file, "\t%f",
                                potential_scatter->at(loop_flag));
                    }
                }
                fprintf(temp_file, "\n");
                //  iterate over any dimensions
                i = ndim - 1;
                while (i >= 0)
                {
                    loop_flag[i] += cv_deltas[i];
                    if (loop_flag[i] > cv_maxs[i])
                    {
                        loop_flag[i] = loop_floor[i];
                        --i;
                    }
                    else
                    {
                        break;
                    }
                }
            }
        }
        else if (potential_grid != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\tvshift\n");
            for (Grid<float>::iterator g_iter = potential_grid->begin();
                 g_iter != potential_grid->end(); ++g_iter)
            {
                ostringstream ss;
                const Axis coor = g_iter.coordinates();
                float vshift = CalcVshift(coor);
                for (const float& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(), *g_iter,
                        *g_iter - vshift, vshift);
            }
        }
        // In case of pure scattering point!
        else if (potential_scatter != nullptr)
        {
            fprintf(temp_file, "# ");
            Write_CV_Header(temp_file, ndim, cvs);
            fprintf(temp_file, "potential_raw\tpotential_shifted\n");
            // fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                const Axis coor = potential_scatter->GetCoordinate(iter);
                float vshift = CalcVshift(coor);
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t%f\n", ss.str().c_str(),
                        potential_scatter->data_[iter],
                        potential_scatter->data_[iter] - vshift);
            }
        }
        fclose(temp_file);
    }
}
void META::Write_Directly(void)
{
    if (!is_initialized || !(use_scatter || usegrid))
    {
        return;
    }
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        FILE* temp_file = NULL;
        Open_File_Safely(&temp_file, write_directly_file_name, "w");
        string meta_type;
        if (do_negative)
        {
            string pm = to_string(potential_max);
            meta_type += "sink(kcal): " + pm;
        }
        if (mask)
        {
            meta_type += " mask ";
        }
        if (subhill)
        {
            meta_type += " subhill ";
        }
        else
        {
            meta_type += " d_force";
        }

        fprintf(temp_file, "%dD-Meta X %s\n", ndim, meta_type.c_str());
        for (int i = 0; i < ndim; ++i)
        {
            fprintf(temp_file, "%f\t%f\t%f\n", cv_mins[i], cv_maxs[i],
                    cv_deltas[i]);
        }
        int gridsize = 1;
        for (int i = 0; i < ndim; ++i)
        {
            int num_grid = round((cv_maxs[i] - cv_mins[i]) / cv_deltas[i]);
            /*if (periods[i] == 0)
            {
                ++num_grid; //  numpoint+1 for non-periodic condition
            }*/
            fprintf(temp_file, " %d\t", num_grid);
            gridsize *= num_grid;
        }
        if (potential_scatter != nullptr)
        {
            // printf("Directly print the %d scatter points to
            // %s\n",scatter_size,write_directly_file_name);
            fprintf(temp_file, "%d\n", scatter_size);
            for (int iter = 0; iter < scatter_size; ++iter)
            {
                ostringstream ss;
                vector<float> coor = potential_scatter->GetCoordinate(iter);
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                if (subhill)
                {
                    fprintf(temp_file, "%s%f\t%f\t%f\n", ss.str().c_str(),
                            potential_local, potential_backup,
                            potential_scatter->data_[iter]);
                }
                else  // restart of catheter will replace the result!
                {
                    float result;
                    result = potential_local;
                    fprintf(temp_file, "%s%f\t", ss.str().c_str(), result);
                    Gdata& data = scatter->data_[iter];
                    for (int i = 0; i < ndim; ++i)
                    {
                        fprintf(temp_file, "%f\t", data[i]);
                    }
                    fprintf(temp_file, "%f\n", potential_scatter->data_[iter]);
                }
            }
        }
        else if (potential_grid != nullptr)
        {
            /*for (int i = 0; i < ndim; ++i)
            {
                fprintf(temp_file, " %d\t", n_grids[i]);
            }*/
            fprintf(temp_file, "%zu\n", potential_grid->size());
            for (Grid<float>::iterator g_iter = potential_grid->begin();
                 g_iter != potential_grid->end(); ++g_iter)
            {
                ostringstream ss;
                vector<float> coor = g_iter.coordinates();
                Estimate(coor, true, false);  // get potential
                for (auto& v : coor)
                {
                    ss << v << "\t";
                }
                fprintf(temp_file, "%s%f\t", ss.str().c_str(),
                        potential_local);  // potential_grid->data_[index]);
                Gdata& data = grid->at(coor);
                for (int i = 0; i < ndim; ++i)
                {
                    fprintf(temp_file, "%f\t", data[i]);
                }
                fprintf(temp_file, "%f\n", *g_iter);
            }
        }
        fclose(temp_file);
    }
}
void META::Read_Potential(CONTROLLER* controller)
{
    FILE* temp_file = NULL;
    Open_File_Safely(&temp_file, read_potential_file_name, "r");
    char temp_char[256];
    int scanf_ret = 0;
    char* get_val = fgets(temp_char, 256, temp_file);  // title line
    Malloc_Safely((void**)&cv_mins, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_maxs, sizeof(float) * ndim);
    Malloc_Safely((void**)&cv_deltas, sizeof(float) * ndim);
    Malloc_Safely((void**)&n_grids, sizeof(float) * ndim);
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%f %f %f\n", &cv_mins[i], &cv_maxs[i],
                           &cv_deltas[i]);
        if (scanf_ret != 3)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
        controller->printf(
            "    CV_minimal = %f\n    CV_maximum = %f\n    dCV = %f\n",
            cv_mins[i], cv_maxs[i], cv_deltas[i]);
    }
    for (int i = 0; i < ndim; ++i)
    {
        scanf_ret = fscanf(temp_file, "%d", &n_grids[i]);
        if (scanf_ret != 1)
        {
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file\n");
        }
    }
    scanf_ret = fscanf(temp_file, "%d\n", &scatter_size);
    // Scatter points coordinate
    for (int i = 0; i < ndim; ++i)
    {
        float* ttoorr;
        Malloc_Safely((void**)&ttoorr, sizeof(float) * scatter_size);
        tcoor.push_back(ttoorr);
    }
    vector<float> potential_from_file;
    vector<Gdata> force_from_file;
    sigma_s = cv_sigmas[0];
    for (int j = 0; j < scatter_size; ++j)
    {
        char* grid_val = fgets(temp_char, 256, temp_file);  // grid line
        /*std::string command = string_strip(temp_char);
        std::vector<std::string> words
         = string_split(command, " ");*/
        std::vector<std::string> words;
        int nwords = split_sentence(temp_char, words);
        Gdata force(ndim, 0.);
        if (nwords < ndim)
        {
            controller->printf("size %d not match %d\n", nwords, ndim);
            controller->Throw_SPONGE_Error(
                spongeErrorBadFileFormat, "META::Read_Potential",
                "Reason:\n\tbad potential input file \n");
        }
        else if (nwords < ndim + 2)
        {
            potential_from_file.push_back(0.);
        }
        else if (subhill && nwords >= ndim + 2)
        {
            potential_from_file.push_back(std::stof(words[nwords - 1]));
            // printf("Success reading line %d\n",j);
        }
        else if (nwords == 2 * ndim + 2)
        {
            potential_from_file.push_back(
                std::stof(words[2 * ndim + 1]));  // raw hill before sink
            if (!subhill)
            {
                for (int i = 0; i < ndim; ++i)
                {
                    force[i] = std::stof(words[1 + ndim + i]);
                }
            }
        }
        for (int i = 0; i < ndim; ++i)
        {
            tcoor[i][j] = std::stof(words[i]);  // coordinate!
        }
        force_from_file.push_back(force);
        if (catheter)
        {
            sigma_r = std::stof(words[ndim]);
            float sr_inv = 1.0 / sigma_r;
            delta_sigma.push_back(0.5 * (sigma_s * sigma_s - sr_inv * sr_inv));
        }
    }
    fclose(temp_file);
    Setgrid(controller);
    vector<float>::iterator max_it =
        max_element(potential_from_file.begin(), potential_from_file.end());
    potential_max = *max_it;
    if (usegrid)
    {
        potential_grid->data_ = potential_from_file;  // potential
        // calculate derivative force dpotential
        if (!subhill)
        {
            int index = 0;
            for (Grid<float>::iterator it = potential_grid->begin();
                 it != potential_grid->end(); ++it)
            {
                for (int i = 0; i < ndim; ++i)
                {
                    Axis coord = it.coordinates();
                    grid->at(coord)[i] = force_from_file[index][i];
                }
                /*
                for (int i = 0; i < ndim; ++i)
                {
                    vector<int> shift(ndim, 0);
                    shift[i] = 1;
                    // float just = *it;
                    auto shit = it;
                    shit += shift;
                    if (shit != potential_grid->end()) // do not compute
                edge !
                    {
                        grid->at(it.GetIndices())[i] = (*shit - *it) /
                cv_deltas[i];
                    }
                }*/
                ++index;
            }
        }
    }
    else if (use_scatter)
    {
        potential_scatter->data_ = potential_from_file;
        if (convmeta)
        {
            max_index = distance(potential_from_file.begin(), max_it);
            // sum_normal =
            // expf(-normal_lse->at(scatter->GetCoordinate(max_index)));
        }
        if (!subhill)
        {
            scatter->data_ = force_from_file;
        }
        if (mask)
        {
            for (int index = 0; index < potential_scatter->size(); ++index)
            {
                Axis coor = potential_scatter->GetCoordinate(index);
                potential_grid->at(coor) = potential_from_file[index];

                for (int i = 0; i < ndim; ++i)
                {
                    grid->at(coor)[i] = force_from_file[index][i];
                }
            }
        }
    }
}

#ifdef USE_GPU
static __global__ void Add_Frc(const int atom_numbers, VECTOR* frc,
                               VECTOR* cv_grad, float dheight_dcv)
{
    for (int i = blockIdx.x + blockDim.x * threadIdx.x; i < atom_numbers;
         i += gridDim.x * blockDim.x)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static __global__ void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static __global__ void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                                  const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#else
static void Add_Frc(const int atom_numbers, VECTOR* frc, VECTOR* cv_grad,
                    float dheight_dcv)
{
#pragma omp parallel for
    for (int i = 0; i < atom_numbers; i++)
    {
        frc[i] = frc[i] - dheight_dcv * cv_grad[i];
    }
}

static void Add_Potential(float* d_potential, const float to_add)
{
    d_potential[0] += to_add;
}

static void Add_Virial(LTMatrix3* d_virial, const float dU_dCV,
                       const LTMatrix3* cv_virial)
{
    d_virial[0] = d_virial[0] - dU_dCV * cv_virial[0];
}
#endif

void META::Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR* frc,
                                             int need_potential,
                                             int need_pressure,
                                             float* d_potential,
                                             LTMatrix3* d_virial)
{
    if (!is_initialized)
    {
        return;
    }
    Potential_and_derivative(need_potential);
    if (do_borderwall)
    {
        Border_derivative(border_upper.data(), border_lower.data(), cutoff,
                          Dpotential_local);
    }

    for (int i = 0; i < cvs.size(); ++i)
    {
        Launch_Device_Kernel(Add_Frc, (atom_numbers + 31 / 32), 32, 0, NULL,
                             atom_numbers, frc, cvs[i]->crd_grads,
                             Dpotential_local[i]);
        if (need_pressure)
        {
#ifdef USE_GPU
            Launch_Device_Kernel(Add_Virial, 1, 1, 0, NULL, d_virial,
                                 Dpotential_local[i], cvs[i]->virial);
#else
            Launch_Device_Kernel(Add_Virial, 1, 1, 0, NULL, d_virial,
                                 Dpotential_local[i], cvs[i]->virial);
#endif
        }
    }
    if (need_potential)
    {
        Launch_Device_Kernel(Add_Potential, 1, 1, 0, NULL, d_potential,
                             potential_local);  // device to host add!!!
    }
}

void META::Potential_and_derivative(const int need_potential)
{
    if (!is_initialized)
    {
        return;
    }
    // d_potential
    Axis values;  // cvs.size;
    for (int i = 0; i < cvs.size(); ++i)
    {
        values.push_back(cvs[i]->value);
        Dpotential_local[i] = 0.f;
    }
    Estimate(values, need_potential, true);
}

void META::Border_derivative(float* border_upper, float* border_lower,
                             float* cutoff, float* Dpotential_local)
{
    for (int i = 0; i < cvs.size(); ++i)
    {
        float h_cv = cvs[i]->value;
        if (h_cv - border_lower[i] < cutoff[i])
        {
            float distance = border_lower[i] - h_cv;
            if (periods[i] > 0)
            {
                distance -= roundf(distance / cv_periods[i]) * cv_periods[i];
            }
            Dpotential_local[i] =
                Dpotential_local[i] - border_potential_height * expf(distance);
        }
        else if (border_upper[i] - h_cv < cutoff[i])
        {
            float distance = h_cv - border_upper[i];
            if (periods[i] > 0)
            {
                distance -= roundf(distance / cv_periods[i]) * cv_periods[i];
            }
            Dpotential_local[i] =
                Dpotential_local[i] + border_potential_height * expf(distance);
        }
    }
}

/*
static __device__ float log_add_log(float a, float b)
{
    return fmaxf(a, b) + logf(1.0 + expf(-fabsf(a - b)));
}
static __global__ void Add_Exp(const int gridsize, float *d_grid, float
*expsum, float *minusbeta)
{
    // int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < gridsize)
    {
        expsum[0] = log_add_log(expsum[0], d_grid[i] * minusbeta[0]);
    }
}
*/
void META::getHeight(const Axis& values)
{
    Estimate(values, true, false);
    height = height_0;
    /*if(use_scatter && scatter != nullptr)
    {
            float ratio =
    normal_lse->at(scatter->GetCoordinate(scatter->GetIndex(values)))-normal_lse->at(values);
            //height = height_0 * expf(-ratio);
            if(abs(ratio) > 2.5 )
            {
                //height = 0;   /// For convergence, don't add this hill!
                //printf("Warning! The ratio of (%f,%f) is
    %f\n",values[0],values[1],expf(ratio));
            }
    }*/
    if (temperature < 0.00001 || welltemp_factor > 60000)
    {
        return;  // avoid /0 = nan
    }
    if (is_welltemp == 1)
    {
        // height_welltemp
        height = height_0 * expf(-potential_backup / (welltemp_factor - 1) /
                                 CONSTANT_kB / temperature);
    }
}

float META::CalcVshift(const Axis& values)
{
    if (!do_negative)
    {
        return 0.;
    }
    /*Axis values = coordinate;
    if (use_scatter)
    {
            values = scatter->GetCoordinate(coordinate);
    }*/
    if (convmeta)
    {
        return new_max *
               expf(normal_lse->at(values));  // normal_factor->at(values);
    }
    else  // GRW
    {
        return new_max * (normal_lse->at(values) - sum_max) *
               expf(normal_lse->at(values));
    }
}
void META::getReweightingBias(float temp)
{
    if (temperature < 0.00001)
    {
        return;  // avoid /0 = nan
    }
    float beta = 1.0 / CONSTANT_kB / temperature;
    minusBetaFplusV = beta / (welltemp_factor - 1.);  /// 300K
    minusBetaF = welltemp_factor * minusBetaFplusV;
    bias = potential_local;
    rbias = potential_backup;  // use original hill for reweighting
    float Z_0 = 0.;            // proportional to the integral of exp(-beta*F)
    float Z_V = 0.;  // proportional to the integral of exp(-beta*(F+V))
    float Z_0_sink =
        0.;  // proportional to the integral of exp(-beta*(F+V_{sink}))
    float Z_V_sink =
        0.;  // proportional to the integral of exp(-beta*(F+V_{sink}))
             /*if (!subhill)
            {
                DeviceFloatVector d;
                if (use_scatter)
                {
                    d = potential_scatter->data_;
                }
                else // use grid
                {
                    d = potential_grid->data_;
                }
                Z_0 = PartitionFunction(minusBetaF, potential_max, d);
                Z_V = PartitionFunction(minusBetaFplusV, potential_max, d);
            }
            else*/
    if (potential_scatter != nullptr)
    {
        for (int iter = 0; iter < scatter_size; ++iter)
        {
            Axis coor = potential_scatter->GetCoordinate(iter);
            Estimate(coor, true, false);  // calculate local potential!!!!

            Z_0 = exp_added(Z_0, minusBetaF * potential_local);
            Z_V = exp_added(Z_V, minusBetaFplusV * potential_local);
            Z_0_sink = exp_added(Z_0_sink, minusBetaF * potential_backup);
            // Calculate the shift potential
            Z_V_sink = exp_added(Z_V_sink, minusBetaFplusV * potential_backup +
                                               beta * CalcVshift(coor));
            // potential_max = max(potential_max, potential_backup);

            if (potential_backup > potential_max)
            {
                max_index = iter;
                potential_max = potential_backup;
            }
        }
    }
    else if (potential_grid != nullptr)
    {
        for (Grid<float>::iterator g_iter = potential_grid->begin();
             g_iter != potential_grid->end(); ++g_iter)
        {
            Estimate(g_iter.coordinates(), true, false);  //
            Z_0 = exp_added(Z_0, minusBetaF * potential_backup);
            Z_V = exp_added(Z_V, minusBetaFplusV * potential_backup);
            potential_max = max(potential_max, potential_backup);
        }
    }
    // sink meta:
    rct = CONSTANT_kB * temperature * (Z_0_sink - Z_V_sink);  // sink meta's rct
    /*
    float rct_sink = CONSTANT_kB * temperature * (Z_0 - Z_V);
    printf("The extra rct of sink is %f\n", rct_sink);
    bias -= rct_sink;
    */
    // use original hill for reweighting
    rbias -= rct + temp;
}

void META::AddPotential(float temp, int steps)
{
    if (!is_initialized)
    {
        return;
    }
    if (potential_update_interval <= 0)
    {
        return;
    }
    if (steps % potential_update_interval == 0)
    {
        Axis values;  //, cvalues;
        for (int i = 0; i < cvs.size(); ++i)
        {
            values.push_back(cvs[i]->value);
        }
        getHeight(values);
        float vshift;
        if (use_scatter)
        {
            Axis projected_values =
                scatter->GetCoordinate(scatter->GetIndex(values));
            vshift = CalcVshift(projected_values);
        }
        else
        {
            vshift = CalcVshift(values);
        }
        getReweightingBias(vshift);
        if (catheter)
        {
            float R = sqrtf(sigma_s * sigma_s -
                            2.0 * delta_sigma[scatter->GetIndex(values)]);
            for (int i = 0; i < ndim; ++i)
            {
                sigmas[i] = R;
            }
        }
        Hill hill = Hill(values, sigmas, periods, height);
        hills.push_back(hill);
        Axis hillinfo;  ///< myhill.log, after n-dim Axis column
        hillinfo.push_back(height);
        if (do_negative)
        {
            hillinfo.push_back(potential_max);
            hillinfo.push_back(max_index);
        }
        if (scatter != nullptr)
        {
            hillinfo.push_back(scatter->GetIndex(values));
            if (mask)  // output path coordinate! The first coulumn is path
                       // s, the last is hill
            {
                // Cartesian2Path(values, hillinfo);
                // hillinfo.push_back(exit_tag);
            }
        }
        hilllog("myhill.log", values, hillinfo);
        exit_tag = 0.0;
        if (!kde && subhill)
        {
            Gdata tder = hill.CalcHill(values);
            if (potential_grid != nullptr)
            {
                potential_grid->at(values) += height * hill.potential;
            }
            else if (potential_scatter != nullptr)
            {
                potential_scatter->at(values) += height * hill.potential;
            }
            return;
        }
        float factor = Normalization(values, height,
                                     kde);  // height with normalized factor
        // vector<int> myindices;
        vector<int> indices;
        if (use_scatter)
        {
            if (do_cutoff)
            {
                indices = potential_scatter->GetNeighbor(values, cutoff);
            }
            else
            {
                indices = vector<int>(scatter_size);
                iota(indices.begin(), indices.end(), 0);
            }
            // myindices = potential_scatter->GetNeighbor(values,
            // cv_deltas.data()); // The write potential cutoff should be
            // cv_delta?
            for (auto index : indices)
            {
                Axis coord = scatter->GetCoordinate(index);
                //  Add to scatter.
                Gdata& data = scatter->data()[index];
                Gdata tder = hill.CalcHill(coord);
                if (1)  //! subhill)
                {
                    if (catheter == 3)
                    {
                        Gdata& v = rotate_v->data()[index];
                        float s = ProjectToPath(v, coord, values);
                        float dss = delta_sigma[index] * s * s;
                        for (int i = 0; i < ndim; ++i)
                        {
                            for (int j = 0; j < ndim; ++j)
                            {
                                data[i] += factor * 2 * (values[i] - coord[i]) *
                                           delta_sigma[index] * v[i] * v[j] *
                                           hill.potential * expf(-dss);
                            }
                            data[i] += factor * tder[i];
                        }
                    }
                    else if (catheter == 2)
                    {
                        Axis values2, coord2;
                        Cartesian2Path(values, values2);
                        Cartesian2Path(coord, coord2);
                        Hill hill2 = Hill(values2, sigmas, periods, height);
                        Gdata tder2 = hill2.CalcHill(coord2);
                        Gdata& R = rotate_matrix->data()[index];
                        ////////////////////////Debug method
                        /// 3/////////////////////////////////////
                        Gdata& v = rotate_v->data()[index];         //
                        float s = ProjectToPath(v, coord, values);  //
                        float dss = delta_sigma[index] * s * s;     //
                        ///////////////////////////////////////////////////////////////////////
                        for (int i = 0; i < ndim; ++i)
                        {
                            float delta1 = 0.0;
                            float delta2 =
                                tder[i] + 2 * delta_sigma[index] * s * v[i] *
                                              hill.potential * expf(-dss);
                            float delta3 = tder[i];
                            float dx = values[i] - coord[i];
                            for (int j = 0; j < ndim; ++j)
                            {
                                float R_ij = R[i + j * ndim];
                                delta1 += R_ij * tder2[j];
                                delta3 += 2 * delta_sigma[index] * v[i] * v[j] *
                                          dx * hill.potential * expf(-dss);
                                // data[i] += factor * R_ij * tder2[j];
                            }
                            data[i] += factor * delta3;
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < ndim; ++i)
                        {
                            data[i] += factor * tder[i];
                        }
                    }
                }
                float potential_temp = factor * hill.potential;
                if (potential_scatter != nullptr)
                {
                    potential_scatter->data_[index] += potential_temp;
                }
            }
        }
        // If exist, add bias onto grid.
        if (potential_grid != nullptr)
        {
            int index = 0;
            int nindex = indices.size();
            for (auto it = potential_grid->begin(); it != potential_grid->end();
                 ++it)
            {
                // Add to grid.
                Gdata tder = hill.CalcHill(it.coordinates());
                if (usegrid || nindex)  // mask within neigtbor of cuoff!
                {
                    *it += factor * hill.potential;
                    if (!subhill && grid != nullptr)  // buggy for mask!!!!!!
                    {
                        Gdata& data = grid->data_[index];
                        for (size_t i = 0; i < grid->GetDimension(); ++i)
                        {
                            data[i] += factor * tder[i];
                        }
                        ++index;
                    }
                }
            }
        }
    }
}

void META::Do_Metadynamics(int atom_numbers, VECTOR* crd, LTMatrix3 cell,
                           LTMatrix3 rcell, int step, int need_potential,
                           int need_pressure, VECTOR* frc, float* d_potential,
                           LTMatrix3* d_virial, float sys_temp)
{
    if (this->is_initialized)
    {
        int need = CV_NEED_GPU_VALUE | CV_NEED_CRD_GRADS;
        if (need_pressure)
        {
            need |= CV_NEED_VIRIAL;
        }

        for (int i = 0; i < cvs.size(); i = i + 1)
        {
            this->cvs[i]->Compute(atom_numbers, crd, cell, rcell, need, step);
        }
        temperature = sys_temp;
        Meta_Force_With_Energy_And_Virial(atom_numbers, frc, need_potential,
                                          need_pressure, d_potential, d_virial);
        AddPotential(sys_temp, step);
    }
}

void META::Step_Print(CONTROLLER* controller)
{
    if (!is_initialized)
    {
        return;
    }
    if (CONTROLLER::MPI_size == 1 && CONTROLLER::PM_MPI_size == 1)
    {
        controller->Step_Print(this->module_name, potential_local);
        return;
    }
#ifdef USE_MPI
    if (CONTROLLER::MPI_rank == CONTROLLER::MPI_size - 1)
    {
        MPI_Send(&potential_local, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    if (CONTROLLER::MPI_rank == 0)
    {
        MPI_Recv(&potential_local, 1, MPI_FLOAT, CONTROLLER::MPI_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        controller->Step_Print(this->module_name, potential_local);
    }
#endif
}
