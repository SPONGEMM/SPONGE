#include "../meta.h"
#include "../util.h"
using sinkmeta::split_sentence;

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
            values.push_back(center);
        }
        float theight = stof(words[cvsize]);
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
    for (int j = 0; j < i; ++j)
    {
        Hill& hill = hills[j];
        Gdata tder = hill.CalcHill(values);
        potential += hill.potential * hill.height;
    }
    return potential;
}
float META::Sumhills(int history_freq)
{
    if (history_freq == 0)
    {
        return 0.;
    }
    TimePoint start_time, end_time;
    float duration;
    GetTime(start_time);
    int nhills = LoadHills("myhill.log");
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
                }
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
    const char* file_name = edge_file_name;

    int total = normal_lse->size();
    if (scatter_size == total)
    {
        float normalization = 1.0;
        float sqrtpi = sqrtf(CONSTANT_Pi * 2);
        for (int i = 0; i < ndim; i++)
        {
            normalization /= cv_deltas[i] * sigmas[i] / sqrtpi;
        }
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
            sum_max = fmaxf(logsumhills, sum_max);
            vector<float> sum_potential(1 + ndim, expf(logsumhills));
            *g_iter = logsumhills;
            if (do_negative)
            {  // print force!
                Gdata& data = normal_force->at(values);
                for (int i = 0; i < ndim; ++i)
                {
                    sum_potential[i + 1] = data[i];
                }
            }
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
    for (int index = 0; index < scatter_size; ++index)
    {
        Axis neighbor = potential_scatter->GetCoordinate(index);
        float lnbias = data->at(neighbor);
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
void META::getHeight(const Axis& values)
{
    Estimate(values, true, false);
    height = height_0;
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
    float Z_0_sink = 0.;
    float Z_V_sink = 0.;
    if (potential_scatter != nullptr)
    {
        for (int iter = 0; iter < scatter_size; ++iter)
        {
            Axis coor = potential_scatter->GetCoordinate(iter);
            Estimate(coor, true, false);
            Z_0 = exp_added(Z_0, minusBetaF * potential_local);
            Z_V = exp_added(Z_V, minusBetaFplusV * potential_local);
            Z_0_sink = exp_added(Z_0_sink, minusBetaF * potential_backup);
            // Calculate the shift potential
            Z_V_sink = exp_added(Z_V_sink, minusBetaFplusV * potential_backup +
                                               beta * CalcVshift(coor));
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
            Estimate(g_iter.coordinates(), true, false);
            Z_0 = exp_added(Z_0, minusBetaF * potential_backup);
            Z_V = exp_added(Z_V, minusBetaFplusV * potential_backup);
            potential_max = max(potential_max, potential_backup);
        }
    }
    rct = CONSTANT_kB * temperature * (Z_0_sink - Z_V_sink);
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
        Axis values;
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
        Axis hillinfo;
        hillinfo.push_back(height);
        if (do_negative)
        {
            hillinfo.push_back(potential_max);
            hillinfo.push_back(max_index);
        }
        if (scatter != nullptr)
        {
            hillinfo.push_back(scatter->GetIndex(values));
            if (mask)
            {
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
            for (auto index : indices)
            {
                Axis coord = scatter->GetCoordinate(index);
                //  Add to scatter.
                Gdata& data = scatter->data()[index];
                Gdata tder = hill.CalcHill(coord);
                if (1)
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
                if (usegrid || nindex)
                {
                    *it += factor * hill.potential;
                    if (!subhill && grid != nullptr)
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
