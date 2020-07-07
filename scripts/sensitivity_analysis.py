# Sensitivity Analysis for CFTR Duct Model

# Import SA modules
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

# Import Duct Model Functions
from model import runBaseModel, parameter_dict

# Import pandas
import pandas as pd

# Must define model inputs

def problemParameters(parameter_dict, scaling_factor):

	# Initialize dictionary to store problem parameters and acceptable bounds
	problem = {'num_vars': 0, 'names': [], 'bounds': []}

	# Handle each element in parameter dictionary and pass to output dictionary
	for key in parameter_dict:

		# Add name to list
		problem['names'].append(key)

		# Increase number of variable seen by one
		problem['num_vars'] += 1

		# Generate upper and lower bounds
		original_value = parameter_dict[key]['Value']
		upper_bound = scaling_factor * original_value
		lower_bound = (1/scaling_factor) * original_value

		# Avoid "Bounds are not legal error" when multiplying negative numbers
		if upper_bound < lower_bound:
			upper_bound, lower_bound = lower_bound, upper_bound

		# Add bounds to list
		problem['bounds'].append([lower_bound, upper_bound])

	return problem

def runModelManyTimes(parameters, scaling_factor):
	# Collect problem parameters, names, and bounds
	problem = problemParameters(parameters, scaling_factor)

	# Generate samples
	param_values = saltelli.sample(problem, 100)

	# Create empty array of same shape to store values
	Y = np.zeros([param_values.shape[0]])

	print(Y.shape)

	# Store results of model in Y array
	for i, option in enumerate(param_values):

		# Put parameters back into dictionary needed for duct model function
		option = dict()
		for j, parameter in enumerate(problem['names']):
			var_dict = dict()
			var_dict[parameter] = dict()
			var_dict[parameter]['Value'] = param_values[i][j]
			option[parameter] = var_dict[parameter]

		# Store peak bicarbonate secretion as measure of model performance
		Y[i] = np.max(runBaseModel(option)['bl'])
		
		print(i, Y[i])


	return problem, Y

def performAnalysis(problem, Y):

	Si = sobol.analyze(problem, Y, print_to_console = True)
	
	pd.DataFrame.from_dict(Si, orient = 'index').to_csv('sa_results.csv')

	return


problem, Y = runModelManyTimes(parameter_dict, 1.3)
performAnalysis(problem, Y)






