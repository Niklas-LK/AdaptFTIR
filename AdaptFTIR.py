#################################
# # # # # # AdaptFTIR # # # # # #
#################################

import numpy as np
from typing import Union

class AdaptFTIR:
    def __init__(self, spectra: np.ndarray, subject_ids: np.array, calibration_set: np.ndarray, calibration_ids: np.array ,other_labels: Union[None, np.ndarray]=None, seed:int=0):

        '''
        Initializes the AdaptFTIR class with given parameters.
        
        input:
        -----------
        spectra:
            np.ndarray with observations as rows and covariates as columns
            (observations with missing spectrum values will be removed)
        subject_ids:
            np.ndarray or list with ids corresponding to the spectra
            (number of spectra must match number of subject ids)
        calibration_set: 
            np.ndarray: observations as rows and covariates as columns
            float between 0 and 1: ratio to split of from the spectra_dataset to use as calibration set (selection will be random)
        calibration_ids:
            None (default): used if the calibration set is split of from the spectra_dataset
            np.array or list: if a seperate calibration set is provided
        other_labels:
            None (default): ignored
            np.ndarray: 
                observations as rows and covariates as columns
                if label does not change within a subject id: copy label for the simulations (e.g. sex, place_of_birth, ...)
                if label does change within a subject id: ignore label (e.g. age, sample id, ...)
        seed: 
            int: fixates the seed for reproducibility

        '''
        self.spectra = spectra
        self.subject_ids = np.array(subject_ids)
        self.calibration_set = calibration_set
        self.calibration_ids = np.array(calibration_ids)
        self.other_labels = other_labels
        self.seed = seed

        self.validate_inputs()

    def validate_inputs(self):
        '''Validates inputs to ensure they meet expected types and values.'''
        # spectra
        if not isinstance(self.spectra, np.ndarray):
            raise ValueError('"spectra" must be a NumPy array')
        
        # subject_ids
        if not isinstance(self.subject_ids, np.ndarray) and not isinstance(self.subject_ids, list):
            raise ValueError('"subject_ids" must be provided as a 1-d list or NumPy array')
        
        # calibration_set
        if not isinstance(self.calibration_set, (np.ndarray, float)):
            raise ValueError('"calibration_set" must be provided as a NumPy array or float')

        # calibration_ids
        if not isinstance(self.calibration_ids, np.ndarray) and not isinstance(self.calibration_ids, list):
            raise ValueError('"calibration_ids" must be provided as a 1-d list or NumPy array')

        # other labels
        if self.other_labels is not None and not isinstance(self.other_labels, (np.ndarray, list)):
            raise ValueError('other_labels must be provided as a np.ndarray')

        # seed
        if not isinstance(self.seed, int):
            raise ValueError('"seed" must be an integer')


    def reshape_data(self, X, ids):
        '''Reshapes data into a list of arrays where each corresponds to one subject_id.'''
        unique_ids = np.unique(ids)
        reshaped_data = [X[ids == uid] for uid in unique_ids]
        return reshaped_data, unique_ids


    def get_sigma_bar(self, X):
        '''Calculates the averaged within-person covariance matrix.'''
        if len(X) != 0:
            summed_cov = np.zeros((X[0].shape[1], X[0].shape[1]))
            for i in range(len(X)):
                summed_cov += np.cov(X[i], rowvar=False)
            Sigma_bar = summed_cov / len(X)
            return Sigma_bar
        else:
            return None


    def mvg(self, n_to_generate_per_id, p, sigma, p_ids):
        '''Simulates new spectra using the element-wise averaged covariance matrix.'''
        np.random.seed(self.seed)
        p_mean = [np.mean(p[i], axis=0) for i in range(len(p))]

        sim_x = []
        for i in range(len(p)):
            sim_x.append(np.random.multivariate_normal(p_mean[i], sigma, n_to_generate_per_id))
        
        if type(self.other_labels) == None:
            sim_y = np.repeat(p_ids, n_to_generate_per_id)
        else:
            sim_y = self.handle_other_labels(n_to_generate_per_id)
        return np.vstack(sim_x), sim_y


    def combine_input_and_sim(self, inp_x, inp_y, sim_x, sim_y):
        '''Combines input data with simulated data.'''
        aug_x = np.vstack([inp_x, sim_x])
        if type(self.other_labels) == None:
            aug_y = np.hstack([inp_y, sim_y])
        else:
            aug_y = np.vstack([inp_y, sim_y])
        return aug_x, aug_y

    def handle_other_labels(self, n_to_generate_per_id):
        # Extract unique IDs
        unique_ids = np.unique(self.subject_ids)
        # Create a list to store the resulting rows
        result = []

        for uid in unique_ids:
            # Filter rows belonging to the current ID
            rows = self.other_labels[self.subject_ids == uid]
            # Initialize a list for storing the processed columns
            processed_rows = []
            
            for col_idx in range(self.other_labels.shape[1]):
                # Extract the column values for the current ID
                col_values = rows[:, col_idx]
                unique_elements = np.unique(col_values)
                
                if len(unique_elements) == 1:
                    processed_rows.append([unique_elements[0]] * n_to_generate_per_id)
                else:
                    processed_rows.append([None] * n_to_generate_per_id)
            
            # Construct the final processed rows for the current ID
            for i in range(n_to_generate_per_id):
                result_row = [uid]
                for col_idx in range(self.other_labels.shape[1]):
                    result_row.append(processed_rows[col_idx][i])
                result.append(result_row)
        
        # Convert the result list to a NumPy array
        return np.array(result, dtype=object)


    def run(self, n_to_generate_per_id=1, return_type='augmentation', shuffle_output=False):
        np.random.seed(self.seed)
        '''Executes the AdaptFTIR process and returns the appropriate data based on return_type.'''

        # return_type check
        if return_type not in ['augmentation', 'simulation']:
            raise ValueError('"return_type" must be in {"augmentation", "simulation"}')
        
        # shuffle check
        if not isinstance(shuffle_output, bool):
            raise ValueError('"shuffle" must be a boolean')

        P, P_ids = self.reshape_data(self.spectra, self.subject_ids)
        C, _ = self.reshape_data(self.calibration_set, self.calibration_ids)

        C_cov = self.get_sigma_bar(C)
        sim_spectra, sim_ids = self.mvg(n_to_generate_per_id, P, C_cov, P_ids)

        # if only simulations should be returned:
        if return_type == 'simulation':
            # shuffle Yes/No?
            if shuffle_output == False:
                return sim_spectra, sim_ids
            else:
                perm = np.random.permutation(len(sim_spectra))
                return sim_spectra[perm], sim_ids[perm]
        # if simulations and input data should be returned
        else:
            # if other labels exist:
            if type(self.other_labels) == None:
                # shuffle Yes/No?
                if shuffle_output == False:
                    return self.combine_input_and_sim(self.spectra, self.subject_ids, sim_spectra, sim_ids)
                else:
                    aug_spectra, aug_ids = self.combine_input_and_sim(self.spectra, self.subject_ids, sim_spectra, sim_ids)
                    perm = np.random.permutation(len(aug_spectra))
                    return aug_spectra[perm], aug_ids[perm]
            else:
                # shuffle Yes/No?
                if shuffle_output == False:
                    return self.combine_input_and_sim(self.spectra, np.hstack([self.subject_ids.reshape(-1, 1), self.other_labels]), sim_spectra, sim_ids)
                else:
                    aug_spectra, aug_ids = self.combine_input_and_sim(self.spectra, np.hstack([self.subject_ids.reshape(-1, 1), self.other_labels]), sim_spectra, sim_ids)
                    perm = np.random.permutation(len(aug_spectra))
                    return aug_spectra[perm], aug_ids[perm]
                

def split_calibration_set(spectra: np.ndarray, subject_ids: np.array, device_ids: np.array, calibration_split: float, seed:int=0):
    '''
    Splits the spectra dataset into a calibration set and a leftover set,
    maintaining the ratio of device IDs in both sets.
    
    Parameters:
    -----------
    spectra: np.ndarray
        Observations as rows and covariates as columns.
    subject_ids: np.array
        Array of subject IDs corresponding to the spectra.
    device_ids: np.array
        Array of device IDs corresponding to the spectra.
    calibration_split: float
        Ratio to split off from the spectra dataset to use as calibration set (0 < calibration_split < 1).

    Returns:
    --------
    calibration_set: np.ndarray
        Calibration set spectra.
    calibration_subject_ids: np.array
        Calibration set subject IDs.
    calibration_device_ids: np.array
        Calibration set device IDs.
    left_over_spectra: np.ndarray
        Leftover spectra.
    left_over_subject_ids: np.array
        Leftover subject IDs.
    left_over_device_ids: np.array
        Leftover device IDs.
    '''
    np.random.seed(seed)
    
    unique_subject_ids = np.unique(subject_ids)
    unique_device_ids = np.unique(device_ids)
    num_subjects = len(unique_subject_ids)
    num_calibration_subjects = int(np.floor(num_subjects * calibration_split))
    
    # Create a dictionary to hold the number of subjects needed per device ID
    device_id_to_subjects = {device: [] for device in unique_device_ids}
    
    # Map each subject to their device ID
    subject_to_device = {subject: device_ids[np.where(subject_ids == subject)[0][0]] for subject in unique_subject_ids}

    # Calculate the number of subjects to sample per device ID
    device_counts = {device: np.sum(device_ids == device) for device in unique_device_ids}
    device_proportions = {device: count / len(device_ids) for device, count in device_counts.items()}
    device_subject_counts = {device: int(np.floor(num_calibration_subjects * proportion * len(unique_device_ids))) for device, proportion in device_proportions.items()}

    # Randomly shuffle unique subject IDs
    np.random.shuffle(unique_subject_ids)

    # Select subjects for the calibration set based on device proportions
    selected_subjects = []
    for subject in unique_subject_ids:
        device_id = subject_to_device[subject]
        if len(device_id_to_subjects[device_id]) < device_subject_counts[device_id]:
            device_id_to_subjects[device_id].append(subject)
            selected_subjects.append(subject)
        if len(selected_subjects) >= num_calibration_subjects:
            break

    # Split the data based on selected subjects
    calibration_indices = np.isin(subject_ids, selected_subjects)
    left_over_indices = ~calibration_indices

    calibration_set = spectra[calibration_indices]
    calibration_subject_ids = subject_ids[calibration_indices]
    calibration_device_ids = device_ids[calibration_indices]

    left_over_spectra = spectra[left_over_indices]
    left_over_subject_ids = subject_ids[left_over_indices]
    left_over_device_ids = device_ids[left_over_indices]

    return (calibration_set, calibration_subject_ids, calibration_device_ids), (left_over_spectra, left_over_subject_ids, left_over_device_ids)
