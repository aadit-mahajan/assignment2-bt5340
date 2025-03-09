import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

plt.style.use('seaborn-paper')

SIGMA = 0.001
R = 0.008314

def partition_function(kappa, n=15):

    # define the transfer matrix
    mat = [
        [1, SIGMA*kappa], 
        [1, kappa]
    ]
    mat = np.array(mat)

    # eigendecomposition of the matrix
    w, s = np.linalg.eig(mat)
    s_inv = np.linalg.inv(s)

    generator = s @ np.diag(np.power(w, n)) @ s_inv
    pre_mult_mat = np.array([1, SIGMA*kappa])
    post_mult_mat = np.array([1, 1])
    partition_function = pre_mult_mat @ generator @ post_mult_mat

    return partition_function

def seq_sw(sequence, kappa):
    nucleation = 0
    n_helix = 0
    if sequence[0] == 1:
        nucleation += 1
 
    for i in range(1, len(sequence)):
        if sequence[i] == 1 and sequence[i-1] == 0:
            nucleation+=1

    for i in sequence:
        if i == 1:
            n_helix += 1

    sw = np.power(SIGMA, nucleation) * np.power(kappa, n_helix)

    return sw

def generate_sequences(n=15):
    # generate all possible sequences of length n

    seqs = []
    for i in range(n):
        for j in range(i, n):
            if j-i >= 0:
                seqs.append([1]*(j-i) + [0]*(n-(j-i)))
            for k in range(j, n):
                for l in range(k, n):
                    if j-i >= 0 and k-j > 1 and l-k >= 0:
                        seqs.append([1]*(j-i) + [0]*(k-j) + [1]*(l-k) + [0]*(n-(l-k)))
    
    return seqs

def generate_sequences_ssa(n=15):
    # generate all possible sequences of length n with only one nucleation site
    seqs = []
    for i in range(n):
        for j in range(i, n):
            if j-i >= 0:
                seqs.append([1]*(j-i) + [0]*(n-(j-i)))

    return seqs

def generate_sequences_dsa(n=15):
    # generate all possible sequences of length n with two nucleation sites
    seqs = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                for l in range(k, n):
                    if j-i >= 0 and k-j > 1 and l-k >= 0:
                        seqs.append([1]*(j-i) + [0]*(k-j) + [1]*(l-k) + [0]*(n-(l-k)))

    return seqs

def free_energy(sequences, partition_function, kappa, temp, n=15):
    free_energies = np.zeros(n)   # this stores the free energy of i helical residues at the ith index
    z = partition_function

    for seq in sequences:
        n_helix = 0
        for i in seq:
            if i == 1:
                n_helix += 1
        
        sw = seq_sw(seq, kappa)
        if sw == 0:
            sw = 1 

        prob = sw / z
        free_energies[n_helix] += -R*temp*np.log(prob)
    free_energies = free_energies/1000  # convert to kJ/mol
    
    return free_energies

def plot_free_energy(temps, free_energies, title, output_dir, type, kappa):
    plt.figure()

    for i in range(len(temps)):
        data = free_energies.iloc[:, i]
        num_helices = np.arange(1, 16, 1)
        plt.plot(num_helices, data, label=f'Temperature: {temps[i]} K', color=plt.cm.Reds(i/len(temps)))
    plt.xlabel('Number of helical residues')
    plt.ylabel('Free Energy (kJ/mol)')
    plt.xticks(np.arange(1, 16, 1))
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.savefig(f'{output_dir}/{type}_kappa_{kappa :.3f}.png')
    plt.close()


def workflow(kappa, type = 'ssa'):
    temp_min = 278
    temp_max = 368
    temp_step = 10
    plot_output_dir = f'./fe_output/{type}'
    os.makedirs(plot_output_dir, exist_ok=True)
    z = partition_function(kappa)
    
    if type == 'ssa':
        seq = generate_sequences_ssa()
    elif type == 'dsa':
        seq = generate_sequences_dsa()
    elif type == 'all':
        seq = generate_sequences()

    fe_temps = {}
    temps = np.arange(temp_min, temp_max+temp_step, temp_step)
    for temp in temps:
        fe = free_energy(seq, z, kappa, temp)
        fe_temps[temp] = fe
    
    fe_df = pd.DataFrame(fe_temps)
    plot_free_energy(temps, fe_df, f'Free Energy vs Number of Helical Residues ({type}), $\kappa = {kappa :.3f}$', plot_output_dir, type, kappa)

if __name__ == '__main__':
    
    kappa_values_low = np.arange(0.01, 0.11, 0.02)
    kappa_values_high = np.arange(10, 60, 10)

    for kappa in kappa_values_low:
        workflow(kappa, 'ssa')
        workflow(kappa, 'dsa')
        workflow(kappa, 'all')

    for kappa in kappa_values_high:
        workflow(kappa, 'ssa')
        workflow(kappa, 'dsa')
        workflow(kappa, 'all')





    






        








