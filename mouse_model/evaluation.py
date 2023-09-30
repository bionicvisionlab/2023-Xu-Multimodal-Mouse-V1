import numpy as np

def correlation(pred, label, chunk_len=100, flatten_neuron=False):
    
    '''
    pred, label: arrays of shape (num_timesteps, num_neurons)
    
    return an array of correlation coefficients
    
    if flatten_neuron = True, the returned array is of shape 
    (num_timesteps//chunk_len), where each entry is the correlation
    coefficient between corresponding pred and label narrays of shape 
    (chunk_len * num_neurons)
    
    if flatten_neuron = False, the returned array is of shape 
    (num_timesteps//chunk_len, num_neurons), where each entry is the 
    correlation between corresponding pred and label arrays of shape
    (chunk_len)
    '''
    
    res = []
    for i in range(0, pred.shape[0], chunk_len):
        if flatten_neuron:
            pred_current_chunk = pred[i:i+chunk_len].flatten()
            label_current_chunk = label[i:i+chunk_len].flatten()
            cor_coef = np.corrcoef(pred_current_chunk, label_current_chunk)[0][1]
            res.append(cor_coef)
        else:
            res_per_neuron = []
            for j in range(pred.shape[1]):
                cor_coef = np.corrcoef(pred[i:i+chunk_len, j], label[i:i+chunk_len, j])[0][1]
                res_per_neuron.append(cor_coef)
            res.append(res_per_neuron)
    return np.array(res)

# correlation in time, averaged with neurons
def cor_in_time(pred, label):
    return correlation(pred, label, chunk_len=pred.shape[0], flatten_neuron=False)

# correlation in neurons, averaged with trials
def cor_in_neurons(pred, label):
    return correlation(pred, label, chunk_len=1, flatten_neuron=True)