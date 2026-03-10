/*
 * Copyright 2021 Gao's lab, Peking University, CCME. All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __META_CUH__
#define __META_CUH__

// clang-format off
#include "../../common.h"
#include "../../control.h"
#include "vector"
#include "Grid.cpp"
#include "Scatter.cpp"
#include "SwitchFunction.h"
#include "../../collective_variable/collective_variable.h"
//#define Grid Scatter

using namespace std;
struct META
{
	using Gdata = std::vector<float>;     ///< Grid's data storage as field.
	Grid<Gdata> *normal_force = nullptr;  ///< do_negative force need normalization
	//Grid<float> *normal_factor = nullptr; ///< do_negative need sum as normalization
	Grid<float> *normal_lse = nullptr; ///< do_negative need Logsumexp of normalization
	Grid<Gdata> *grid = nullptr;
	Grid<float> *potential_grid = nullptr;
	Scatter<Gdata> *rotate_v = nullptr;	 ///< Anisotropic, path-dependent width of Gaussian
	Scatter<Gdata> *rotate_matrix = nullptr; ///< Cartesian to path coordinate rotation, right hand rule
	Scatter<Gdata> *scatter = nullptr;
	Scatter<float> *potential_scatter = nullptr;
	using Axis = std::vector<float>; ///< Axis with upper/lower boundary.
	struct Hill
	{
		/**
		 * @brief  Constructs a multidimensional Hill (Gaussian)
		 * @param[in] center Hill center.
		 * @param[in] inv_w Hill width(inverted).
		 * @param[in] period periodic box length.
		 * @param[in] height Hill height.
		 */
		Hill(const Axis &centers, const Axis &inv_w, const Axis &period, const float &theight);
		/**
		 * @brief Evaluate the Gaussian hill
		 * The F_old will be changed
		 * @param[in] values CV values for Hill as axis.
		 */
		Gdata CalcHill(const Axis &values);
		vector<GaussianSF> gsf; ///< Hill center and width.
		float height;		///< Hill height.
		float potential;	///< Hill potential.
	};
	vector<Hill> hills;
	bool usegrid = true;	    // false; ///< True or false.
	bool use_scatter = false;   ///< True or false.
	bool do_borderwall = false; ///< do exponetial wall at border
	bool do_cutoff = false;	    ///< do cutoff for faster loop
	bool do_negative = false;   ///< do negative hill in force: sink metad
	bool subhill = false;	    ///< add subhill to get gaussian
	bool kde = false;	    ///< Use Kernal density estimation(KDE)
	float dip = 0.0;	    ///< submarine extra dip for local minimum
	float sum_normal = 1.0;	    ///< do_negative need sum with normalization \Phi(s(V_max))
	float sum_max = 0.0;
	vector<float> delta_sigma;  ///< \frac{1}{2sigma_s^2} - \frac{1}{2sigma_r^2}
	float sigma_s;		    ///< The path sigma
	float sigma_r;		    ///< The wide sigma
	float exit_tag;		    ///< label exit in mask version(n-dim area)
	float new_max=0.;
	int max_index;
	float maxforce = 0.1;	    ///< Edge's force critiria for exit label
	int convmeta = 0;	    ///< ConvolutionMeta: constant normalization factor
	int grw = 0;
	int catheter = 0;	    ///< anisotropic Gaussian in catheter metad.
	int scatter_size = 0;
	int history_freq = 0;      ///< Calculate Rbias/RCT using sumhill at the begining
	std::vector<float *> tcoor; //(ndim, scatter_size);
	Axis vsink;
	Axis RotateVector(const Axis &tang_vector, bool do_debug);
	void Cartesian2Path(const Axis &Cartesian_values, Axis &Path_values);
	/**
	 * @brief unit vector along the path
	 * @param[out] normalized unit vector
	 * @param[in]
	 * @param[in]
	 * @return double s, normalized factor
	 */
	double TangVector(Gdata &tang_vector, const Axis &values, const Axis &neighbor);
	/**
	 * @brief Project Cartesian Coordinate to Path Coordinate
	 * @param[in] First unit vector of Rotation Matrix
	 * @param[in] Projected point in Path
	 * @param[in] CV's Cartesian
	 * @return   The Path value s of the tang_vector
	 */
	float ProjectToPath(const Gdata &tang_vector, const Axis &values, const Axis &Cartesian);
	void Setgrid(CONTROLLER *controller);
	void Estimate(const Axis &values, const bool need_potential, const bool need_force);
	bool ReadEdgeFile(const char *file_name, vector<float> &potential);
	void PickScatter(const string fn, Grid<float> *data);//, const float sum_max);
	int LoadHills(const string& fn);//, const vector<double>& widths)
	float CalcHill(const Axis & values, const int i);
	float  Sumhills(int history_freq);//,const vector<float> heights)
	void EdgeEffect(const int dim, const int size);
	float CalcVshift(const Axis &values);
	float Normalization(const Axis &values, float factor, bool do_normalise); // Normalization
	int is_initialized = 0;

	//作用的CV
	//   COLLECTIVE_VARIABLE_PROTOTYPE *cv;
	CV_LIST cvs;
	int ndim = 1; ///< default 1D metadynamics
	int mask = 0; ///< default 1D mask
	// ndim-Meta的边界和细度记录
	float border_potential_height = 1000.; // cv边界近似无限高势垒
	vector<float> border_lower;
	vector<float> border_upper;
	vector<float> cv_period;
	float *cv_mins;
	float *cv_maxs;
	vector<float> sigmas;
	vector<float> periods;
	vector<float> cv_deltas;
	float *cv_periods;
	float *cv_sigmas;
	int *n_grids;
	float *d_grid;

	//多次运行SPONGE相关的记录存储操作
	char read_potential_file_name[256];
	char write_potential_file_name[256];
	char write_directly_file_name[256]; ///<  Use scatter points, directly
	void Step_Print(CONTROLLER *controller);
	void Read_Potential(CONTROLLER *controller);
	void Write_Potential(void);
	void Write_Directly(void);

	float *cutoff;	//查表的范围（通常是3sigma）
	float height;	//通过well-temperature修正的reweight过后的势场高度
	float height_0; //初始势场高度

	float welltemp_factor = 1000000000.; // Biasfactor无限大时，即为普通的Meta无时间衰减，因此预留一个大值。
	int is_welltemp = 0;
	float temperature = 300;

    void Meta_Force_With_Energy_And_Virial(int atom_numbers, VECTOR *frc, int need_potential, int need_pressure, float *d_potential, LTMatrix3 *d_virial);
    void Do_Metadynamics(int atom_numbers, VECTOR *crd, LTMatrix3 cell, LTMatrix3 rcell,
                         int step, int need_potential, int need_pressure, VECTOR *frc, float *d_potential, LTMatrix3 *d_virial, float sys_temp);

	char module_name[CHAR_LENGTH_MAX];
	void Initial(CONTROLLER *controller, COLLECTIVE_VARIABLE_CONTROLLER *cv_controller, char *module_name = NULL);
	void AddPotential(float sys_temp, int steps);
	void getHeight(const Axis & values);
	void getReweightingBias(float temp);

	float factor_cv_grid_sum = 0.; //归一化因子

	void Potential_and_derivative(const int need_potential);				    //?��?????Potential??DPotential
	void Border_derivative(float *upper, float *lower, float *cutoff, float *Dpotential_local); //????Upper_Wall??��??

	float potential_local = 0.;	   //存储float Potential()
	float potential_backup = 0.;	   //存储float Potential()
	float potential_max = 0.;	   //存储float Potential()
	float *Dpotential_local = nullptr; //存储float* DPotential()
	int potential_update_interval;
	int write_information_interval;
	float rct = 0.;
	float rbias = 0.;	   // Rw_Potential：rbias=potential(x)-rct;
	float bias = 0.;	   // bias;
	float minusBetaF = 1.0;	   // non-welltemperd minusBetaF
	float minusBetaFplusV = 0; // non-welltemperd

};
#endif // META
