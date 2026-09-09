#pragma once

struct trajectory_output
{
    MD_INFORMATION* md_info = NULL;
    int current_crd_synchronized_step = 0;
    bool print_zeroth_frame = false;
    bool print_virial = false;
    float last_density = NAN;
    int write_trajectory_interval = 1000;    // 打印轨迹内容的所隔步数
    int write_mdout_interval = 1000;         // 打印能量信息的所隔步数
    int write_restart_file_interval = 1000;  // restart文件重新创建的所隔步数
    FILE* crd_traj = NULL;
    FILE* box_traj = NULL;
    SpongeH5OutputPlan::ResolvedOutputPlan h5_output_plan;
    std::string h5_output_identity_uuid;
    std::unique_ptr<SpongeH5MD::WriterBackend> h5_trajectory_backend;
    std::unique_ptr<SpongeH5MD::TrajectoryH5Writer> h5_trajectory_writer;
    std::unique_ptr<SpongeH5MD::WriterBackendFactory> h5_vds_backend_factory;
    std::unique_ptr<SpongeH5MD::VdsTrajectoryH5Writer> h5_vds_trajectory_writer;
    bool h5_trajectory_enabled = false;
    bool h5_trajectory_vds_enabled = false;
    bool h5_trajectory_velocity_enabled = false;
    bool h5_trajectory_force_enabled = false;
    std::vector<std::string> h5_observable_names;
    std::unique_ptr<SpongeH5MD::WriterBackend> h5_observable_backend;
    std::unique_ptr<SpongeH5MD::ObservableH5Writer> h5_observable_writer;
    bool h5_observable_enabled = false;
    std::vector<std::string> h5_observable_only_names;
    bool h5_restart_enabled = false;
    int64_t h5_restart_generation = 0;
    bool h5_nhc_observable_enabled = false;
    std::size_t h5_nhc_chain_length = 0;
    bool h5_sits_nk_enabled = false;
    std::string h5_sits_module_name;
    std::size_t h5_sits_k_count = 0;
    bool h5_metadynamics_scalar_enabled = false;
    bool h5_reaxff_energy_enabled = false;
    bool h5_reaxff_eeq_snapshot_enabled = false;
    std::size_t h5_reaxff_eeq_atom_count = 0;
    std::vector<std::string> h5_reaxff_terms;
    double h5_trajectory_finalize_elapsed_s = 0.0;
    double h5_observable_finalize_elapsed_s = 0.0;
    double h5_restart_finalize_elapsed_s = 0.0;
    std::vector<std::string> h5_output_failures;
    char restart_name[CHAR_LENGTH_MAX];
    void Initial(CONTROLLER* controller, MD_INFORMATION* md_info);
    void Initial_H5_Trajectory(CONTROLLER* controller);
    void Initial_H5_Observable(CONTROLLER* controller);
    void Initial_H5_Restart(CONTROLLER* controller);
    void Write_H5_Legacy_Sidecar_Provenance(
        CONTROLLER* controller, SpongeH5MD::TrajectoryH5Writer* writer,
        const char* context);
    void Write_H5_Legacy_Sidecar_Provenance(
        CONTROLLER* controller, SpongeH5MD::VdsTrajectoryH5Writer* writer,
        const char* context);
    void Write_H5_Legacy_Sidecar_Provenance(
        CONTROLLER* controller, SpongeH5MD::ObservableH5Writer* writer,
        const char* context);
    void Write_H5_Legacy_Sidecar_Provenance(CONTROLLER* controller,
                                            SpongeH5MD::RestartH5Writer* writer,
                                            const char* context);
    void Initial_H5_Nose_Hoover_Chain(CONTROLLER* controller,
                                      std::size_t chain_length);
    void Initial_H5_Sits_Nk(CONTROLLER* controller, const char* module_name,
                            std::size_t k_count);
    void Initial_H5_Metadynamics(CONTROLLER* controller, int is_initialized);
    void Initial_H5_Reaxff(CONTROLLER* controller, int is_initialized,
                           std::size_t eeq_atom_count);
    void Prepare_H5_Swmr_Layout(CONTROLLER* controller,
                                const char* metadynamics_module_name);
    void Start_H5_Swmr(CONTROLLER* controller);
    void Append_H5_Trajectory_Frame(CONTROLLER* controller);
    void Append_H5_Observable_Frame(CONTROLLER* controller);
    void Append_H5_Observable_Only_Frame(CONTROLLER* controller);
    void Append_H5_Nose_Hoover_Chain_Frame(CONTROLLER* controller,
                                           const float* coordinates,
                                           const float* velocities,
                                           std::size_t chain_length);
    void Append_H5_Sits_Nk_Frame(CONTROLLER* controller,
                                 const char* module_name, const float* values,
                                 std::size_t k_count);
    void Append_H5_Metadynamics_Scalar_Frame(CONTROLLER* controller,
                                             double meta, double rbias,
                                             double rct);
    void Append_H5_Reaxff_Frame(CONTROLLER* controller);
    void Write_H5_Reaxff_Eeq_Charge_Snapshot(CONTROLLER* controller,
                                             const float* values,
                                             std::size_t atom_count);
    void Write_H5_Metadynamics_Diagnostic_File(CONTROLLER* controller,
                                               const char* module_name,
                                               const char* component,
                                               const char* file_name);
    void Publish_Output(CONTROLLER* controller);
    void Publish_H5_Output(CONTROLLER* controller);
    void Finalize_H5_Trajectory(CONTROLLER* controller);
    void Finalize_H5_Observable(CONTROLLER* controller);
    void Record_H5_Output_Failure(const char* family, const char* phase,
                                  const std::string& reason);
    std::string H5_Output_Failure_Summary() const;
    void Export_H5_Restart_File(
        CONTROLLER* controller, const float* nhc_coordinates = NULL,
        const float* nhc_velocities = NULL, std::size_t nhc_chain_length = 0,
        const SpongeH5MD::RestartSitsState* sits_state = NULL,
        const char* metad_module_name = NULL,
        const SpongeH5MD::RestartMetadynamicsState* metad_state = NULL,
        const char* metad_hills_file_name = NULL,
        const char* metad_history_file_name = NULL,
        const char* metad_edge_file_name = NULL,
        const char* metad_potential_file_name = NULL,
        const char* metad_direct_file_name = NULL,
        const char* restraint_name = NULL,
        const float* restraint_reference_coordinates = NULL,
        std::size_t restraint_atom_count = 0,
        const std::map<std::string, std::vector<float>>* cv_references = NULL,
        const SpongeH5MD::RestartDynamicState* dynamic_state = NULL);
    bool Should_Write_Legacy_Restart(CONTROLLER* controller);
    void Export_Restart_File(const char* rst7_name = NULL);
    void Append_Crd_Traj_File(FILE* fp = NULL);
    void Append_Box_Traj_File(FILE* fp = NULL);
    // 20210827用于输出速度和力
    int is_frc_traj = 0, is_vel_traj = 0;
    int restart_export_count = 0;
    int max_restart_export_count = 1;
    FILE* frc_traj = NULL;
    FILE* vel_traj = NULL;
    void Append_Frc_Traj_File(FILE* fp = NULL);
    void Append_Vel_Traj_File(FILE* fp = NULL);
    bool Check_Mdout_Step();
    bool Check_Force_Step();
    bool Check_Trajectory_Step();
    bool Check_Restart_Step();
};
