import numpy as np
from scipy import stats

# Mean Absolute Error (average L1 distance).
def compute_MAE(Prediction_vector, Target_vector):
	MAE = abs(Prediction_vector - Target_vector).sum() / Target_vector.size
	return MAE

# Mean Squared Error (average L2 distance).
def compute_MSE(Prediction_vector, Target_vector):
	MSE = ((Prediction_vector - Target_vector)**2).sum() / Target_vector.size
	return MSE

# Peak Signal-to-Noice Ratio.
def compute_PSNR(Prediction_vector, Target_vector):
	MSE = ((Prediction_vector - Target_vector)**2).sum() / Target_vector.size
	MaxI = np.max(Target_vector) - np.min(Target_vector)
	PSNR = 10 * np.log10(MaxI**2 / MSE)
	return PSNR

# Normalized Cross Correlation.
def compute_NCC(Prediction_vector, Target_vector):
	NCC = (np.multiply(Prediction_vector, Target_vector) / (np.std(Prediction_vector) * np.std(Target_vector))).sum() / Target_vector.size
	return NCC

# Pearson Correlation: Measure the linear relationship between two datasets
def compute_PR(Prediction_vector, Target_vector):
	# First value is coefficient, second value is p value
	PR = stats.pearsonr(Prediction_vector, Target_vector)
	return PR

# Spearman Correlation 
def compute_SR(Prediction_vector, Target_vector):
	SR = stats.spearmanr(Prediction_vector, Target_vector, nan_policy='omit')
	return SR

# ROI-based Correlation Coefficient.