import pandas as pd
import numpy as np
import scipy # must be at least scipy >=  version 1.4.X
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from accessoryfxns import antiporter, eff_perm, nernst

'''
Handle File Path
'''
import sys
sys.path.append('../')
data_path = sys.path[0].replace('/scripts', '/data')

# Load csv into pandas df, generate data dictionary for parameter values
filename = 'parameters.csv'
df_parameters = pd.read_csv(data_path + '/'+ filename, index_col = 'Parameters')
parameter_dict = df_parameters.to_dict('index')

def calculatePotentials(y, parameter_dict):
	# Voltage potential drive ion gradients across the cell

	# Unpack variables
	bi, bl, ci, cl, ni = y[0], y[1], y[2], 160 - y[2], y[3]

	# # Load intracellular parameters from dictionary
	# bi = parameter_dict['bi']['Value']
	# ci = parameter_dict['ci']['Value']
	# ni = parameter_dict['ni']['Value']

	# # Load luminal parameters from dictionary
	# bl = parameter_dict['bl']['Value']
	# cl = parameter_dict['cl']['Value']

	# Load basolateral parameters from dictionary
	bb = parameter_dict['bb']['Value']
	nb = parameter_dict['nb']['Value']

	# Sodium bicarbonate channel (NBC) voltage potential
	enbc = nernst((bi**2*ni), (bb**2*nb))

	# Bicarbonate voltage potential
	eb = nernst(bi, bl)

	# Chloride voltage potential
	ec = nernst(ci, cl)

	# Potassium voltage potential
	ek = parameter_dict['ek']['Value']

	# Sodium voltage potential
	ena = nernst(nb, ni)

	return enbc, eb, ec, ek, ena 

def calculatePermeabilities(y, parameter_dict, gcftr):
	# Permeabilities change if CFTR channel is open or closed

	# Unpack variables
	bi, bl, ci, cl = y[0], y[1], y[2], 160 - y[2]

	# # Load intracellular parameters from dictionary
	# bi = parameter_dict['bi']['Value']
	# ci = parameter_dict['ci']['Value']

	# # Load luminal parameters from dictionary
	# bl = parameter_dict['bl']['Value']
	# cl = parameter_dict['cl']['Value']

	# Load conductances from dictionary
	g_cl = parameter_dict['g_cl']['Value']
	g_bi = parameter_dict['g_bi']['Value']

	# Effective chloride CFTR permeability
	kccf = eff_perm(ci,cl)*gcftr*g_cl

	# Effective bicarbonate CFTR permeability
	kbcf = eff_perm(bi, bl)*gcftr*g_bi

	# Effective sodium bicarbonate channel (NBC) CFTR permeability
	knbc = parameter_dict['gnbc']['Value']

	return kccf, kbcf, knbc

def calculateOverallVoltage(parameter_dict, enbc, eb, ec, ek, ena, knbc, kbcf, kccf):
	
	# Load necessary parameters
	gk = parameter_dict['gk']['Value']
	gnaleak = parameter_dict['gnaleak']['Value']

	# Load into NP arrays for multiplication & summation
	voltage_array = np.array([enbc, eb, ec, ek, ena])
	permeability_array_with_leak = np.array([knbc, kbcf, kccf, gk, gnaleak])
	permeability_array_no_leak = np.array([knbc, kbcf, kccf, gk])

	# Calculate voltage
	voltage = np.sum(voltage_array * permeability_array_with_leak) / np.sum(permeability_array_no_leak)

	return voltage

def calculateVoltageDependentFluxes(v, enbc, eb, ec, knbc, kbcf, kccf):

	# Bicarbonate flux through sodium bicarbonate channel
	jnbc = knbc * (v - enbc)

	# Bicarbonate flux through CFTR channel
	jbcftr = kbcf * (v - eb)

	# Chloride flux through CFTR channel
	jccftr = kccf * (v - ec)

	return jnbc, jbcftr, jccftr

def calculateAntiporterDependentFluxes(y, parameter_dict, luminal_antiporter_status, basolateral_antiporter_status):

	# Unpack variables
	bi, bl, ci, cl = y[0], y[1], y[2], 160 - y[2]

	# Load parameters from dictionary
	gapl = parameter_dict['gapl']['Value']
	kbi = parameter_dict['kbi']['Value']
	kcl = parameter_dict['kcl']['Value']
	bb = parameter_dict['bb']['Value']
	cb = parameter_dict['cb']['Value']

	# Flux through luminal antiporter
	if luminal_antiporter_status:
		japl = antiporter(bl, bi, cl, ci, kbi, kcl) * gapl
	else:
		japl = 0

	# Flux through basolateral antiporter
	if basolateral_antiporter_status:
		japbl = antiporter(bb, bi, cb, ci, kbi, kcl) * gapbl
	else:
		japbl = 0

	return japl, japbl

def calculateIonicFluxes(parameter_dict, jbcftr, japl, jccftr, japbl):

	# Load parameters from dictionary
	jac = parameter_dict['jac']['Value']
	rat = parameter_dict['rat']['Value']
	vr = parameter_dict['vr']['Value']

	# Luminal bicarbonate flux
	jbl = (-jbcftr - japl) / vr + jac * rat

	# Intracellular chloride flux
	jci =  jccftr - japl - japbl

	# Luminal chloride flux
	jcl = (-jccftr + japl) / vr + jac

	return jbl, jci, jcl

def calculateScaledLuminalFluxes(y, parameter_dict, jcl, jbl, v, ena):

	# Unpack variables
	ni = y[3]

	# Load parameters from dictionary
	ionstr = parameter_dict['ionstr']['Value']
	epump = parameter_dict['epump']['Value']
	np0 = parameter_dict['np0']['Value']
	gnak = parameter_dict['gnak']['Value']
	gnaleak = parameter_dict['gnaleak']['Value']

	# Luminal flux
	jlum = (jcl + jbl) / ionstr

	# Sodium / potassium channel flux
	jnak = gnak * (v - epump) * (ni / np0)**3

	# Sodium leak flux
	jnaleak = gnaleak  * (v - ena)

	return jlum, jnak, jnaleak

def calculateFlow(parameter_dict, jlum):
	# Calculate flow of ions through luminal side
	return jlum * parameter_dict['ionstr']['Value']

def calculateDifferentialEqs(y, parameter_dict, jbcftr, japl, japbl, jnbc, jbl, jlum, jci, jnak, jnaleak):

	# Unpack variables
	bi, bl = y[0], y[1]

	# Load parameters from dictionary
	zeta = parameter_dict['zeta']['Value']
	buf = parameter_dict['buf']['Value']
	bi0 = parameter_dict['bi0']['Value']
	chi = parameter_dict['chi']['Value']

	# Intracellular bicarbonate concentration
	dbi_dt = zeta * chi * (jbcftr + japl + japbl + buf * (bi0 - bi) + 2 * jnbc)

	# Luminal bicarbonate concentration
	dbl_dt = zeta * (jbl - jlum * bl)

	# Intracellular chloride concentration
	dci_dt = zeta * jci

	# Intracellular sodium concentration
	dni_dt = zeta * (jnbc - jnak - jnaleak)

	return dbi_dt, dbl_dt, dci_dt, dni_dt


def model(t, y, parameter_dict, gcftr, luminal_antiporter_status, basolateral_antiporter_status):

	# Calculate Nernst potentials
	enbc, eb, ec, ek, ena = calculatePotentials(y, parameter_dict)

	# Calculate effective CFTR permeabilities
	kccf, kbcf, knbc = calculatePermeabilities(y, parameter_dict, gcftr)

	# Calculate overall voltage of cell
	v = calculateOverallVoltage(parameter_dict, enbc, eb, ec, ek, ena, knbc, kbcf, kccf)

	# Calculate voltage-dependent fluxes
	jnbc, jbcftr, jccftr = calculateVoltageDependentFluxes(v, enbc, eb, ec, knbc, kbcf, kccf)

	# Calculate the antiporter-dependent fluxes
	japl, japbl = calculateAntiporterDependentFluxes(y, parameter_dict, luminal_antiporter_status, basolateral_antiporter_status)

	# Calculate overall ionic fluxes
	jbl, jci, jcl = calculateIonicFluxes(parameter_dict, jbcftr, japl, jccftr, japbl)

	# Calculate scaled luminal fluxes
	jlum, jnak, jnaleak = calculateScaledLuminalFluxes(y, parameter_dict, jcl, jbl, v, ena)

	# Calculate flow
	flow = calculateFlow(parameter_dict, jlum)

	dbi_dt, dbl_dt, dci_dt, dni_dt = calculateDifferentialEqs(y, parameter_dict, jbcftr, japl, japbl, jnbc, jbl, jlum, jci, jnak, jnaleak)

	return [dbi_dt, dbl_dt, dci_dt, dni_dt]

def bundleInitialConditions(parameter_dict):

	# Load parameters from dictionary
	bi = parameter_dict['bi']['Value']
	bl = parameter_dict['bl']['Value']
	ci = parameter_dict['ci']['Value']
	ni = parameter_dict['ni']['Value']

	# Package into array to pass into model
	y0 = [bi, bl, ci, ni]

	return y0

def runBaseModel(parameters):

	# Bundle Initial Conditions
	y0 = bundleInitialConditions(parameters)
	t_on, t_off, t_end = 1500, 5000, 8000
	cftr_closed = (parameters, 0.00007, False, False)
	cftr_open = (parameters, 1, False, False)

	# Solve initial state
	state_0 = solve_ivp(fun = model, t_span = (0, t_on),
						 y0 = y0, args = cftr_closed)

	# Gather last elements of solution array to pass to next state
	y1 = [state_0.y[0][-1], state_0.y[1][-1], state_0.y[2][-1], state_0.y[3][-1]]

	# Solve open CFTR state
	state_1 = solve_ivp(fun = model, t_span = (t_on, t_off),
						 y0 = y1, args = cftr_open)

	# Gather last elements of solution array to pass to next state
	y2 = [state_1.y[0][-1], state_1.y[1][-1], state_1.y[2][-1], state_1.y[3][-1]]

	# Solve closed CFTR state
	state_2 = solve_ivp(fun = model, t_span = (t_off, t_end),
						 y0 = y2, args = cftr_closed)

	# Package output array to graph or analyze
	output, concentrations = dict(), ['bi', 'bl', 'ci', 'ni']
	output['time'] = np.concatenate([state_0.t, state_1.t, state_2.t])
	for i in range(len(concentrations)):
		output[concentrations[i]] = np.concatenate([state_0.y[i], state_1.y[i], state_2.y[i]])
	output['cl'] = np.asarray([160 - output['bl']])

	return output

def graphModel(output):

	# Graph change in concentration for bicarbonate, chloride, and sodium
	for key in output:
		if key != 'time':
			plt.plot(output['time'], output[key].transpose())

	# Add legend
	plt.legend(['bi', 'bl', 'ci', 'ni', 'cl'])

	plt.show()

	return

# graphModel(runBaseModel(parameter_dict))











