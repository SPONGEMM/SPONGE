set(SPONGE_SOURCES
    ${PROJECT_ROOT_DIR}/SPONGE/main.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/common.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/control.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/xponge/xponge.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/MD_core/MD_core.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Domain_decomposition/Domain_decomposition.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/neighbor_list.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/neighbor_list/full_neighbor_list.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/Lennard_Jones_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/LJ_soft_core.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/Lennard_Jones_force/solvent_LJ.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Middle_Langevin_MD.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Andersen_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Berendsen_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Bussi_thermostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/thermostat/Nose_Hoover_Chain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/barostat/pressure_based_barostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/barostat/MC_barostat.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/PM_force/PM_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/angle/angle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/angle/Urey_Bradley_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bond/bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/cmap/cmap.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/dihedral/dihedral.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/dihedral/improper_dihedral.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/nb14/nb14.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/restrain/restrain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/constrain.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/settle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/constrain/shake.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/virtual_atoms/virtual_atoms.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/collective_variable.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/RMSD.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/combine.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/simple_cv.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/collective_variable/tabulated.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/plugin/plugin.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/steer.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/restrain_cv.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/Meta1D.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/SinkMeta/Meta.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/SinkMeta/Grid.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/SinkMeta/Scatter.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/bias/SinkMeta/SwitchFunction.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/SITS/SITS.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/custom_force/listed_forces.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/custom_force/pairwise_force.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/wall/soft_wall.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/wall/hard_wall.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/quantum_chemistry/quantum_chemistry.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/Coulomb_Force_No_PBC.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/Lennard_Jones_force_No_PBC.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/NO_PBC/generalized_Born.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/sw.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/edip.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/eam.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/tersoff.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/eeq.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/bond_order.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/bond.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/vdw.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/over_under.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/valence_angle.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/torsion.cpp
    ${PROJECT_ROOT_DIR}/SPONGE/manybody/reaxff/hydrogen_bond.cpp)

set(SOURCES ${SPONGE_SOURCES})

find_package(tomlplusplus CONFIG REQUIRED)
add_library(sponge_toml STATIC
            ${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp)
set_source_files_properties(${PROJECT_ROOT_DIR}/SPONGE/third_party/toml/toml.cpp
                            PROPERTIES LANGUAGE CXX)
target_include_directories(sponge_toml PUBLIC ${PROJECT_ROOT_DIR}/SPONGE)
target_link_libraries(sponge_toml PUBLIC tomlplusplus::tomlplusplus)

add_executable(${CURRENT_TARGET} ${SOURCES})
target_link_libraries(${CURRENT_TARGET} PRIVATE sponge_toml)
install(TARGETS ${CURRENT_TARGET} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
