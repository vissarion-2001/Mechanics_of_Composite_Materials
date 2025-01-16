# Mechanics_of_Composite_Materials

This code provides a python script for running Laminated composites Analysis 

## Classes 

- Material: This is the main class of the program providing several methods for analyzing the composite

# complaince_matrix = material.complaince_matrix() 

-heights = Material.construct_h()
    # heights_for_stresses = material.heights_for_stresses()
    # angles = material.convert_deg_to_rad()
    # A = material.abd_matrix()[0]
    # B = material.abd_matrix()[1]
    # D = material.abd_matrix()[2]
    # ABD = material.abd_matrix()[3]
    # Q_per_lamina = material.abd_matrix()[4]
    # moments = material.moments()
    # d = material.disp_pressure_calculations()
    # material.threed_plotting_displacement_pressure()
    # stresses = material.stresses_calc()
    # stresses_t = material.thermal_stress_calculations()
    # q = material.q_forces_calculations
    # material.plotting_moments()
    # material.plotting_stresses(5)
    # material.plotting_q_forces()
    # s = material.shear_stresses()
    # material.plotting_strains(50, 50)
    # material.tsai_wu_fpf(stresses)
    # fs_all_mechanical = material.max_criterion_failure(stresses)
    # fs_all_thermal = material.max_criterion_failure(stresses_t)
    # sft_m = material.safety_of_margin_calculations(stresses)
    # sft_t = material.safety_of_margin_calculations(stresses_t)
    
