import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_movie(movie):
    """Normalize the range of gray levels in a movie"""
    norm_movie = movie.astype(float)
    norm_movie -= norm_movie.min()
    if not np.isclose(norm_movie.max(), 0):
        norm_movie /= norm_movie.max()
    return norm_movie

def normalize_movie_neg_pos_1(movie):
    """Normalize the range of gray levels in a movie"""
    movie = normalize_movie(movie)
    movie = (movie - 0.5) * 2
    return movie

def get_prod_up_to_square(arr):
    res = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            res.append(arr[i]*arr[j])
    return np.array(res)

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

class MouseDatasetSegNewBehav(Dataset):
    
    def __init__(self, file_id, seg_idx, segment_num=10, data_split="train", vid_type="vid_shift_mean", 
                 seq_len=1, predict_offset=1, behav_mode="orig", norm_mode="01"):
        
        """
        Constructs the mouse dataset (that is already split into train/test set) for a certain experiment.
        Works with the data in this folder: /hdd/aiwenxu/mouse-data-3-segment-split-70-30-48ms
        More info on the data split see notebook may_1_data_split.ipynb

        Parameters
        ----------
            file_id : str
                can be one of the following: "070921_J553RT", "110421_J569LT", "101521_J559NC"
            seg_idx : int
                can be 0, 1, ..., segment_num - 1
            segment_num : int
                can be 3, 5, 8 or 10, defaults to 10
            data_split : str
                can be either "train" or "test", defaults to "train"
            vid_type : str
                can be either "vid_mean", "vid_middle" (head cam video) 
                or "vid_shift_mean", "vid_shift_middle" (eye shifted video)
            seq_len : int
                how many timesteps of images and behavioral variable is contained in one sample, defaults to 1
            predict_offset: int
                index offset (in timesteps) of the neural activity compared to the ending index of the images 
                and behavioral, defaults to 1
                combined with seq_len, this means that
                images and behavorial variable: [start_idx, start_idx + 1, ..., start_idx + seq_len - 1]
                neural spikes: [start_idx + seq_len - 1 + predict_offset)
            behav_prod : str
                can be either "orig", "orig_prod", "velo", "velo_prod", "all", "all_prod"
                if "orig", return all behavioral variables (6, gz is not loaded into the dataset)
                if "orig_prod", return "orig" and their combinations products, up to 2nd power (6 + 15 = 21)
                if "velo", return speed and the first derivative of all the other behavioral variables (speed, pitch',
                roll', phi', theta', eyerad') (6)
                if "velo_prod", return "velo" and their combinations products, up to 2nd power (6 + 15 = 21)
                if "all", return (speed, pitch, roll, phi, theta, eyerad, pitch', roll', phi', theta', eyerad') (11)
                if "all_prod", return (speed, pitch, roll, phi, theta, eyerad, pitch', roll', phi', theta', eyerad')
                and their combinations products, up to 2nd power (11 + 55 = 66)
            norm_mode : str
                can be either "01" or "-1+1"
                if "01", normalize images to [0, 1].
                if "-1+1", normalize images to [-1, +1]
        """

        self.seq_len = seq_len
        self.predict_offset = predict_offset
        if behav_mode not in ["orig", "orig_prod", "velo", "velo_prod", "all", "all_prod"]:
            raise ValueError("behav_mode must be one of orig, orig_prod, velo, velo_prod, all, all_prod")
        self.behav_mode = behav_mode
        
        ROOT_DIR_SEG = "/hdd/aiwenxu/mouse-data-{}-segment-split-70-30-48ms".format(segment_num)
        data_dir = "{}/{}".format(ROOT_DIR_SEG, file_id)
        
        # firing rate below 3 Hz
        bad_neuron_index_dict = {"070921_J553RT": [1,  20,  21,  25,  27,  30,  31,  35,  39,  45,  46,  49,  50,  
                                                   51,  52,  53,  54,  55,  57,  58,  59,  64,  65,  68,  75,  77,  
                                                   79,  81,  83,  84,  87,  88,  90,  91,  94,  95,  97, 104, 105, 107], 
                                 "101521_J559NC": [1,  4,  9, 22, 25, 27, 28, 29, 43, 45, 46, 47, 51, 53], 
                                 "110421_J569LT": [0,  1,  2,  5,  6,  8, 11, 12, 13, 16, 18, 22, 24, 26, 28, 30, 41, 
                                                   44, 47, 49]}
        
        all_nsp = np.load("{}/{}_nsp_seg_{}.npy".format(data_dir, data_split, seg_idx))
        good_nsp = []
        for i in range(all_nsp.shape[1]):
            if i not in bad_neuron_index_dict[file_id]:
                good_nsp.append(all_nsp[:, i])
        self.nsp = np.transpose(np.array(good_nsp))
            
        self.images = np.load("{}/{}_{}_seg_{}.npy".format(data_dir, data_split, vid_type, seg_idx))
        if norm_mode == "01":
            self.images = normalize_movie(self.images)
        elif norm_mode == "-1+1":
            self.images = normalize_movie_neg_pos_1(self.images)
        else:
            raise ValueError("norm_mode must be either 01 or -1+1")
        self.images = np.expand_dims(self.images, axis=1)
        
        behav_key_list = ['speed', 'pitch', 'roll', 'phi', 'th', 'eyerad']
        behavior_var_list = [np.load("{}/{}_{}_seg_{}.npy".format(data_dir, data_split, behav_key, seg_idx)) 
                             for behav_key in behav_key_list]
        self.behavior_var = np.stack(behavior_var_list, axis=1)
        self.behavior_var = np.nan_to_num(self.behavior_var)
        
        behav_diff_key_list = ['pitch', 'roll', 'phi', 'th', 'eyerad']
        behavior_var_diff_list = [np.diff(np.nan_to_num(np.load("{}/{}_{}_seg_{}.npy".format(
            data_dir, data_split, behav_key, seg_idx))), prepend=0) for behav_key in behav_diff_key_list]
        self.behavior_var_diff = np.stack(behavior_var_diff_list, axis=1)

    def __len__(self):
        len_block = self.seq_len + self.predict_offset
        return self.images.shape[0] - len_block + 1

    def __getitem__(self, idx):
        
#         print("input: start_index", idx, "end_index", idx+self.seq_len-1)
#         print("pred: ", idx+self.seq_len-1+self.predict_offset)
        
        current_frame = torch.tensor(self.images[idx:(idx+self.seq_len)], dtype=torch.float)
        
        if self.behav_mode == "orig":
            current_behavior_var = torch.tensor(self.behavior_var[idx:(idx+self.seq_len)], dtype=torch.float)
        elif self.behav_mode == "orig_prod":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                prods = get_prod_up_to_square(self.behavior_var[i])
                behav_frame = np.concatenate([self.behavior_var[i], prods])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "velo":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                behav_frame = np.concatenate([self.behavior_var[i, 0:1], self.behavior_var_diff[i]])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == 'velo_prod':
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                before_prod = np.concatenate([self.behavior_var[i, 0:1], self.behavior_var_diff[i]])
                prod = get_prod_up_to_square(before_prod)
                behav_frame = np.concatenate([before_prod, prod])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "all":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                behav_frame = np.concatenate([self.behavior_var[i], self.behavior_var_diff[i]])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "all_prod":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                before_prod = np.concatenate([self.behavior_var[i], self.behavior_var_diff[i]])
                prod = get_prod_up_to_square(before_prod)
                behav_frame = np.concatenate([before_prod, prod])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
            
        neural_spikes = torch.tensor(self.nsp[idx+self.seq_len-1+self.predict_offset], dtype=torch.float)
        
        return current_frame, current_behavior_var, neural_spikes
    
class MouseDatasetSegNewBehavNanInterp(Dataset):
    
    def __init__(self, file_id, seg_idx, segment_num=10, data_split="train", vid_type="vid_shift_mean", 
                 seq_len=1, predict_offset=1, behav_mode="orig", norm_mode="01"):
        
        """
        Constructs the mouse dataset (that is already split into train/test set) for a certain experiment.
        Works with the data in this folder: /hdd/aiwenxu/mouse-data-3-segment-split-70-30-48ms
        More info on the data split see notebook may_1_data_split.ipynb

        Parameters
        ----------
            file_id : str
                can be one of the following: "070921_J553RT", "110421_J569LT", "101521_J559NC"
            seg_idx : int
                can be 0, 1, ..., segment_num - 1
            segment_num : int
                can be 3, 5, 8 or 10, defaults to 10
            data_split : str
                can be either "train" or "test", defaults to "train"
            vid_type : str
                can be either "vid_mean", "vid_middle" (head cam video) 
                or "vid_shift_mean", "vid_shift_middle" (eye shifted video)
            seq_len : int
                how many timesteps of images and behavioral variable is contained in one sample, defaults to 1
            predict_offset: int
                index offset (in timesteps) of the neural activity compared to the ending index of the images 
                and behavioral, defaults to 1
                combined with seq_len, this means that
                images and behavorial variable: [start_idx, start_idx + 1, ..., start_idx + seq_len - 1]
                neural spikes: [start_idx + seq_len - 1 + predict_offset)
            behav_prod : str
                can be either "orig", "orig_prod", "velo", "velo_prod", "all", "all_prod"
                if "orig", return all behavioral variables (6, gz is not loaded into the dataset)
                if "orig_prod", return "orig" and their combinations products, up to 2nd power (6 + 15 = 21)
                if "velo", return speed and the first derivative of all the other behavioral variables (speed, pitch',
                roll', phi', theta', eyerad') (6)
                if "velo_prod", return "velo" and their combinations products, up to 2nd power (6 + 15 = 21)
                if "all", return (speed, pitch, roll, phi, theta, eyerad, pitch', roll', phi', theta', eyerad') (11)
                if "all_prod", return (speed, pitch, roll, phi, theta, eyerad, pitch', roll', phi', theta', eyerad')
                and their combinations products, up to 2nd power (11 + 55 = 66)
            norm_mode : str
                can be either "01" or "-1+1"
                if "01", normalize images to [0, 1].
                if "-1+1", normalize images to [-1, +1]
        """

        self.seq_len = seq_len
        self.predict_offset = predict_offset
        if behav_mode not in ["orig", "orig_prod", "velo", "velo_prod", "all", "all_prod"]:
            raise ValueError("behav_mode must be one of orig, orig_prod, velo, velo_prod, all, all_prod")
        self.behav_mode = behav_mode
        
        ROOT_DIR_SEG = "/hdd/aiwenxu/mouse-data-{}-segment-split-70-30-48ms".format(segment_num)
        data_dir = "{}/{}".format(ROOT_DIR_SEG, file_id)
        
        # firing rate below 3 Hz
        bad_neuron_index_dict = {"070921_J553RT": [1,  20,  21,  25,  27,  30,  31,  35,  39,  45,  46,  49,  50,  
                                                   51,  52,  53,  54,  55,  57,  58,  59,  64,  65,  68,  75,  77,  
                                                   79,  81,  83,  84,  87,  88,  90,  91,  94,  95,  97, 104, 105, 107], 
                                 "101521_J559NC": [1,  4,  9, 22, 25, 27, 28, 29, 43, 45, 46, 47, 51, 53], 
                                 "110421_J569LT": [0,  1,  2,  5,  6,  8, 11, 12, 13, 16, 18, 22, 24, 26, 28, 30, 41, 
                                                   44, 47, 49]}
        
        all_nsp = np.load("{}/{}_nsp_seg_{}.npy".format(data_dir, data_split, seg_idx))
        good_nsp = []
        for i in range(all_nsp.shape[1]):
            if i not in bad_neuron_index_dict[file_id]:
                good_nsp.append(all_nsp[:, i])
        self.nsp = np.transpose(np.array(good_nsp))
            
        self.images = np.load("{}/{}_{}_seg_{}.npy".format(data_dir, data_split, vid_type, seg_idx))
        if norm_mode == "01":
            self.images = normalize_movie(self.images)
        elif norm_mode == "-1+1":
            self.images = normalize_movie_neg_pos_1(self.images)
        else:
            raise ValueError("norm_mode must be either 01 or -1+1")
        self.images = np.expand_dims(self.images, axis=1)
        
        behav_key_list = ['speed', 'pitch', 'roll', 'phi', 'th', 'eyerad']
        behavior_var_list = []
        for behav_key in behav_key_list:
            y = np.load("{}/{}_{}_seg_{}.npy".format(data_dir, data_split, behav_key, seg_idx))
            nans, x = nan_helper(y)
            y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            behavior_var_list.append(y)
        self.behavior_var = np.stack(behavior_var_list, axis=1)
        
        behav_diff_key_list = ['pitch', 'roll', 'phi', 'th', 'eyerad']
        behavior_var_diff_list = []
        for behav_key in behav_diff_key_list:
            y = np.load("{}/{}_{}_seg_{}.npy".format(data_dir, data_split, behav_key, seg_idx))
            nans, x = nan_helper(y)
            y[nans]= np.interp(x(nans), x(~nans), y[~nans])
            behavior_var_diff_list.append(np.diff(y, prepend=0))
        self.behavior_var_diff = np.stack(behavior_var_diff_list, axis=1)

    def __len__(self):
        len_block = self.seq_len + self.predict_offset
        return self.images.shape[0] - len_block + 1

    def __getitem__(self, idx):
        
#         print("input: start_index", idx, "end_index", idx+self.seq_len-1)
#         print("pred: ", idx+self.seq_len-1+self.predict_offset)
        
        current_frame = torch.tensor(self.images[idx:(idx+self.seq_len)], dtype=torch.float)
        
        if self.behav_mode == "orig":
            current_behavior_var = torch.tensor(self.behavior_var[idx:(idx+self.seq_len)], dtype=torch.float)
        elif self.behav_mode == "orig_prod":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                prods = get_prod_up_to_square(self.behavior_var[i])
                behav_frame = np.concatenate([self.behavior_var[i], prods])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "velo":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                behav_frame = np.concatenate([self.behavior_var[i, 0:1], self.behavior_var_diff[i]])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == 'velo_prod':
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                before_prod = np.concatenate([self.behavior_var[i, 0:1], self.behavior_var_diff[i]])
                prod = get_prod_up_to_square(before_prod)
                behav_frame = np.concatenate([before_prod, prod])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "all":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                behav_frame = np.concatenate([self.behavior_var[i], self.behavior_var_diff[i]])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
        elif self.behav_mode == "all_prod":
            current_behavior_var = []
            for i in range(idx, idx+self.seq_len):
                before_prod = np.concatenate([self.behavior_var[i], self.behavior_var_diff[i]])
                prod = get_prod_up_to_square(before_prod)
                behav_frame = np.concatenate([before_prod, prod])
                current_behavior_var.append(behav_frame)
            current_behavior_var = np.array(current_behavior_var)
            current_behavior_var = torch.tensor(current_behavior_var, dtype=torch.float)
            
        neural_spikes = torch.tensor(self.nsp[idx+self.seq_len-1+self.predict_offset], dtype=torch.float)
        
        return current_frame, current_behavior_var, neural_spikes