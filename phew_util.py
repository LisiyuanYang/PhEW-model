import numpy as np
from astropy import constants, units

MHYDR = constants.m_p.cgs.value #proton mass in g
BOLTZMANN = constants.k_B.cgs.value #Boltzmann constant in cgs
XH = 0.76 #mass fraction of hydrogen
XHE = (1.0 - XH) / (4.0 * XH) #n(He) / n(H)
mu_ionized = (1 + 4 * XHE) / (2 + 3 * XHE) #Assumes full ionization
GAMMA = 1.66667

Coulomb_log_constant = 30 #In the PhEW model ln(Lambda) is always assumed to be 30

conductive_massloss_factor = 6.1e-7 * 8 * np.pi * mu_ionized * MHYDR / (15 * GAMMA * BOLTZMANN)

def cloud_crushing_timescale(density_cloud, density_amb, r_cloud, v_rel):
    '''
    Cloud crushing timescale of a cloud-crushing problem.
    '''
    return np.sqrt(density_cloud / density_amb) * (r_cloud / v_rel)

def Coulomb_log(n_e, T_e):#n_e should be in cm**-3
    return 15.88 + np.log(T_e / np.sqrt(n_e)) #Not going to be used in the PhEW model

@np.vectorize(excluded=['variable_log'])
def kappa_h(T_h, n_e=None, variable_log=False):
    '''
    Spitzer conduction coefficient
    '''
    if variable_log:
        assert n_e != None
        log = Coulomb_log(n_e, T_h)
    else:
        log = 30
    return 1.83e-5 * T_h ** 2.5 / log

def sound_speed_2(temperature):
    '''
    Square of the speed of sound as a function of temperature; assumes full ionization.
    '''
    return GAMMA * BOLTZMANN * temperature / (mu_ionized * MHYDR)

@np.vectorize(excluded=['q'])
def jp_conduction_density(q, Mach):
    '''
    Conductive shock jump condition for the ambient density. Taken from the Gizmo implementation.
    '''
    if Mach <= 1:
        return 1.
    else:
        beta = 1 / (GAMMA * Mach ** 2)
        y = np.sqrt(9 + 16 * q + 5 * beta * (5 * beta - 6))
        return 8 / (5 * (1 + beta) - y)
    
@np.vectorize(excluded=['q'])
def jp_conduction_temperature(q, Mach):
    '''
    Conducive shock jump condition for the ambient temperature. Taken from the Gizmo implementation.
    '''
    if Mach <= 1:
        return 1.
    else:
        x = 1 / jp_conduction_density(q, Mach)
        beta = 1 / (GAMMA * Mach ** 2)
        return max((1 + beta - x) / beta * x, 1.) #Make sure that the post-shock temperature is not lower than the original

@np.vectorize
def khi_suppression(rho_cloud, T_cloud, r_cloud, T_amb, Mach, f_s=0.1, q=0.9):
    '''
    The suppression factor for KHI due to conduction. Everything should be in cgs.
    '''
    tau_s = jp_conduction_temperature(q, Mach)
    if Mach > 1:
        T_ps = max(1, tau_s) * T_amb
    else:
        T_ps = T_amb
    
    rho_ps = rho_cloud * T_cloud / T_ps
    n_ps = rho_ps / (mu_ionized * MHYDR)
    lkh = 1.7745e5 * f_s * np.sqrt(rho_cloud/rho_ps) * T_ps ** 2 / n_ps
    return lkh / r_cloud

def lambda_KHI(rho_cloud, rho_ps, T_ps, n_ps, f_s=0.1):
    '''
    KHI length scale. Taken from the Gizmo implementation.
    '''
    return 1.7745e5 * f_s * np.sqrt(rho_cloud/rho_ps) * T_ps ** 2 / n_ps

def evaporative_rhs(T_star, n_ps, T_ps, R_c, T_c, PhEWFRadius=1.):
    '''
    The rhs of equation 36. Its root determines the saturated zone for conduction.
    '''
    #if T_star == 0:
    #    T_star = T_ps
    a = 2.4e4 * (T_ps ** 2.5 - T_star ** 2.5) * (T_c / T_star) ** 1.53
    b = n_ps * T_ps / np.sqrt(T_star) * R_c * PhEWFRadius
    return a / b - 1

def evaporative_mlr_classic(T_star, T_ps, L_cloud, f_m=3.5, f_s=0.1):
    '''
    Evaporative mass loss rate with classical conduction. In practice, we first solve for T_* using equation 36 (replacing T_c with T_*), and then use
    classic conduction to calculate the mass loss rate between the surface with a temperature of T_* and T_ps. Equation 30 of paper I.
    '''
    return conductive_massloss_factor * (T_ps ** 2.5 - T_star ** 2.5) * L_cloud / f_m * f_s
