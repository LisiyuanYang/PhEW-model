import numpy as np
from scipy.optimize import bisect
import copy
from matplotlib import pyplot as plt
from astropy import constants, units
from phew_util import MHYDR, mu_ionized, BOLTZMANN, sound_speed_2, cloud_crushing_timescale, \
    jp_conduction_density, jp_conduction_temperature, lambda_KHI, evaporative_rhs, evaporative_mlr_classic

#PhEWFsCond = 0.1
PhEWFsCond = 0.1
PhEWQs = 0.9
PhEWFKHI = 30.0
PhEWFm = 3.5

#Do everything in cgs
class cloud:
    '''
    A cloud and its ambient environment. For simplicity, the core temperature of the cloud is always assumed to be 1e4 K. This assumption is also made
    in most of the subroutines of the Gizmo implementation, even though it does evolve the cloud temperature. All parameters should be in cgs units.
    '''
    def __init__(self, mass, radius, L_cloud, T_cloud, density, T_amb, density_amb, v_rel, wind_clock):
        '''
        Set up an unshocked cloud. All parameters should be in cgs units.
        '''
        self.wind_clock = wind_clock #Need this for the expansion velocity
        self.mass = mass
        self.L_cloud = L_cloud
        self.T_cloud = T_cloud
        self.density = density
        self.density_prev = density #The density at the previous timestep
        self.radius = np.cbrt(3 * mass / (4 * np.pi * density))
        self.radius_prev = self.radius
        self.T_amb = T_amb
        self.density_amb = density_amb
        self.n_amb = density_amb / (mu_ionized * MHYDR)
        self.T_ps = T_amb
        self.density_ps = density_amb #Post shock ambient temperature and density
        self.n_ps = self.density_ps / (mu_ionized * MHYDR)
        self.v_rel = v_rel
        self.Mach = v_rel / np.sqrt(sound_speed_2(T_amb))
        self.tcc = cloud_crushing_timescale(density, density_amb, self.radius, v_rel)
    
    def shock(self, inplace=True, qs=PhEWQs):
        '''
        Post-shock conditions of the cloud. Does not really do anything if the cloud is subsonic.
        '''
        if inplace:
            new_cloud = self
        else:
            new_cloud = copy.copy(self)
        new_cloud.T_ps = new_cloud.T_amb * jp_conduction_temperature(qs, new_cloud.Mach)
        new_cloud.density_ps = new_cloud.density_amb * jp_conduction_density(qs, new_cloud.Mach)
        new_cloud.n_ps = new_cloud.density_ps / (mu_ionized * MHYDR)
        new_cloud.density_prev = new_cloud.density #Need this because phew.c somehow uses this to calculate the KHI mass loss rate; shouldn't matter as long as the time step is small enough
        new_cloud.density = new_cloud.density_ps * new_cloud.T_ps / new_cloud.T_cloud
        new_cloud.radius_prev = new_cloud.radius
        new_cloud.radius = np.cbrt(new_cloud.mass / (2.0 * np.pi * new_cloud.density)) #Implicitly assumes a cylinder with L=2R_c
        if inplace == False:
            return new_cloud
    
    def expand_velocity(self, f_growth=1.0):
        '''
        Calculates how fast the tail of the cloud expands
        '''
        factor_1 = 4.46e-15 * self.T_amb ** 2.5 * self.wind_clock / (self.density * self.radius ** 2) / f_growth
        factor_2 = self.T_amb * self.density_amb / (self.T_ps * self.density_ps)
        c_c = np.sqrt(sound_speed_2(self.T_cloud))
        v_exp = -c_c * np.log(max(factor_1, factor_2))
        if v_exp < 0:
            v_exp = 0
        return v_exp
    
    def saturation_temperature(self):
        '''
        Solves for T* by equating equation 35 to 1
        '''
        if evaporative_rhs(self.T_cloud, self.n_ps, self.T_ps, self.radius, self.T_cloud) < 0:
            return 1e4
        else:
            to_solve = lambda T_star: evaporative_rhs(T_star, self.n_ps, self.T_ps, self.radius, self.T_cloud)
            T_star = bisect(to_solve, self.T_cloud, self.T_ps) #Using bisect to match the Gizmo implementation.
            return T_star
    
    def evaporation_timescale(self, f_m=PhEWFm, f_s=PhEWFsCond):
        '''
        The evaporation time scale of the cloud. Equation 37.
        '''
        T_star = self.saturation_temperature()
        evaporative_mlr = evaporative_mlr_classic(T_star, self.T_ps, self.L_cloud, f_m=f_m, f_s=f_s)
        if evaporative_mlr > 0:
            return self.mass / evaporative_mlr
        else:
            return 1e50
    
    def khi_timescale(self, f_s=PhEWFsCond, f_khi=PhEWFKHI):
        '''
        Kelvin-Helmholtz time scale of the cloud. Equation 24.
        '''
        tau_cc = cloud_crushing_timescale(self.density_prev, self.density_amb, self.radius, self.v_rel)
        tau_khi = f_khi * tau_cc * np.sqrt(1 + self.Mach)
        lkh = lambda_KHI(self.density_prev, self.density_ps, self.T_ps, self.n_ps, f_s=f_s)
        return tau_khi, lkh
    
    def get_mass_loss_timescale(self, f_m=PhEWFm, f_s=PhEWFsCond, f_khi=PhEWFKHI):
        '''
        Calculates the overall mass loss rate of the cloud using a combination of supressed KHI and evaporation. Equation 43 with the typo fixed.
        '''
        tau_evap = self.evaporation_timescale(f_m=f_m, f_s=f_s)
        tau_khi, lkh = self.khi_timescale(f_s=f_s, f_khi=f_khi)
        return 1 / (1 / tau_evap + np.exp(-lkh / self.radius) / tau_khi)
    
    def hydro_acceleration(self, f_accel=1.0):
        '''
        Ram pressure accleration. Equation 44.
        '''
        if self.Mach > 1:
            P_ps = self.n_ps * self.T_ps * BOLTZMANN
        else:
            P_ps = self.n_amb * self.T_amb * BOLTZMANN * (1 + self.Mach / 3) ** 2.5
        return -P_ps * np.pi * self.radius ** 2 / self.mass * np.sign(self.v_rel) * f_accel
    
    def step(self, dtime, inplace=True, qs=PhEWQs, f_m=PhEWFm, f_s=PhEWFsCond, f_khi=PhEWFKHI, f_growth=1.0, f_accel=1.0):
        '''
        Evolve the whole cloud by one time step. If inplace=True, the cloud itself is changed. Otherwise sets up a new cloud instance.
        '''
        if inplace:
            new_cloud = self
        else:
            new_cloud = copy.copy(self)
        new_cloud.shock(qs=qs)
        tau_disrupt = new_cloud.get_mass_loss_timescale(f_m=f_m, f_s=f_s, f_khi=f_khi)
        f_disrupt = 1 - np.exp(-dtime / tau_disrupt)
        v_exp = new_cloud.expand_velocity(f_growth=f_growth)
        hydro_accel = new_cloud.hydro_acceleration(f_accel=f_accel)
        new_cloud.mass *= 1 - f_disrupt
        new_cloud.L_cloud += v_exp * dtime
        new_cloud.v_rel += hydro_accel * dtime
        new_cloud.Mach = new_cloud.v_rel / np.sqrt(sound_speed_2(new_cloud.T_amb))
        new_cloud.wind_clock += dtime

        if inplace == False:
            return new_cloud
        
def make_peq_cloud(mass, density, T_amb, v_rel, T_cloud=1e4, qs=PhEWQs):
    '''
    Sets up the initial (post-shock) condition for a PhEW particle. Makes a new instance of cloud.
    '''
    density_amb = density * T_cloud / T_amb #Assumes pressure equilibrium
    new_cloud = cloud(mass, 0, 0, T_cloud, density, T_amb, density_amb, v_rel, 0.)
    new_cloud.shock(qs=qs)
    new_cloud.L_cloud = 4 * new_cloud.radius
    return new_cloud

if __name__ == '__main__':
    '''
    An example of setting up and evolving a cloud.
    '''
    cloud_mass_cgs = (1e5 * units.solMass).cgs.value
    cloud_density_cgs = 1e-26 #already in g/cm^3
    T_amb = 3e6 #chi_0=300
    v_rel = 1700 * 1e5 #in cm/s

    cloud_sample = make_peq_cloud(cloud_mass_cgs, cloud_density_cgs, T_amb, v_rel)
    t_final = 100 * cloud_sample.tcc
    step_size = 0.01 * cloud_sample.tcc
    time_list = np.arange(0, t_final + 0.5 * step_size, step_size)
    mass_list = np.zeros_like(time_list)
    vel_list = np.zeros_like(time_list)
    mass_list[0] = cloud_mass_cgs
    vel_list[0] = v_rel

    for i in range(1, len(time_list)):
        if cloud_sample.mass > 0:
            cloud_sample.step(step_size)
            mass_list[i] = cloud_sample.mass
            vel_list[i] = cloud_sample.v_rel
        else:
            break
    
    time_list = time_list[: i - 1]
    mass_list = mass_list[: i - 1]
    vel_list = vel_list[: i - 1]
    
