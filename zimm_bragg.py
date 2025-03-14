import numpy as np
import matplotlib.pyplot as plt
import math

plt.style.use('seaborn-paper')

T_REF = 298
SIGMA = 0.001
R = 8.314
S_REF = 1.5
NRES = 15
TSTART = 200
TEND = 450
TSTEP = 20

S_HI = 1.5
S_LO = 0.5

def get_s_temp(temp, s_ref=S_REF):
    s = s_ref * math.exp((T_REF-temp)/temp)
    return s

def get_all_sequences():
    total_combs = 2**NRES
    sequences = []
    for i in range(total_combs):
        bin_str = list(bin(i))[2:]
        seq = (NRES-len(bin_str))*['0'] + bin_str
        
        seq = [int(i) for i in seq]
        sequences.append(seq)

    return sequences

def get_nu_nhelix(seq):
    num_nuc = 0
    if seq[0] == 1:
        num_nuc += 1
    for i in range(1, len(seq)):
        if seq[i] == 1 and seq[i-1] == 0:
            num_nuc += 1

    nhelix = sum(seq)
    return num_nuc, nhelix

def FE(temp, seqs):
    z_tot = 0

    s = get_s_temp(temp, S_REF)

    sw_dict = {} 
    for seq in seqs:
        num_nuc, nhelix = get_nu_nhelix(seq)
        sw = math.pow(SIGMA, num_nuc) * math.pow(s, nhelix)
        # print(nhelix, num_nuc, sw, seq, math.pow(SIGMA, num_nuc), math.pow(s, nhelix), sep='\t')
        z_tot += sw
        if nhelix not in sw_dict.keys():
            sw_dict[nhelix] = sw
        else:
            sw_dict[nhelix] += sw

    probs = [sw_dict[i]/z_tot for i in range(NRES+1)]
    # print('z_total', z_tot)
    fe = -R*temp*np.log(probs)
    # print(sw_dict, temp)
    fe = fe/1000
    return fe, probs

def get_s_cont(seq, s_hi, s_lo, s1=3, e1=7, s2=10, e2=14):
    s_cont = 1
    helix_res = list(range(s1-1, e1)) + list(range(s2-1, e2))
    for i in range(len(seq)):
        if seq[i] == 1 and i in helix_res:
            s_cont *= s_hi
        elif seq[i] == 1 and i not in helix_res:
            s_cont *= s_lo

    return s_cont

def FE_two_seq(temp, seqs):
    z_tot = 0

    s_hi = get_s_temp(temp, s_ref=S_HI)
    s_lo = get_s_temp(temp, s_ref=S_LO)

    sw_dict = {} 
    for seq in seqs:
        num_nuc, nhelix = get_nu_nhelix(seq)
        s_cont = get_s_cont(seq, s_hi, s_lo)
        sw = math.pow(SIGMA, num_nuc) * s_cont
        # print(nhelix, num_nuc, sw, seq, math.pow(SIGMA, num_nuc), math.pow(s, nhelix), sep='\t')
        z_tot += sw
        if nhelix not in sw_dict.keys():
            sw_dict[nhelix] = sw
        else:
            sw_dict[nhelix] += sw

    probs = [sw_dict[i]/z_tot for i in range(NRES+1)]
    # print('z_total', z_tot)
    fe = -R*temp*np.log(probs)
    # print(sw_dict, temp)
    fe = fe/1000
    return fe, probs

def FE_s_var(temp, seqs):
    
    fh_arr = []
    for s in range(0, 6):
        z_tot = 0
        sw_dict = {} 
        for seq in seqs:
            num_nuc, nhelix = get_nu_nhelix(seq)
            sw = math.pow(SIGMA, num_nuc) * math.pow(s, nhelix)
            # print(nhelix, num_nuc, sw, seq, math.pow(SIGMA, num_nuc), math.pow(s, nhelix), sep='\t')
            z_tot += sw
            if nhelix not in sw_dict.keys():
                sw_dict[nhelix] = sw
            else:
                sw_dict[nhelix] += sw

        probs = [sw_dict[i]/z_tot for i in range(NRES+1)]
        # print('z_total', z_tot)
        fe = -R*temp*np.log(probs)
        # print(sw_dict, temp)
        fe = fe/1000
        fh = fraction_helicity(probs)
        fh_arr.append(fh)

    plt.plot(range(6), fh_arr)
    plt.title('Fraction Helicity vs s')
    plt.xlabel('s')
    plt.grid()
    plt.ylabel(r'Fraction Helicity $\theta_H$')
    plt.tight_layout()
    plt.savefig('s_var.png')
    plt.close()
    # plt.show()

def fraction_helicity(probs):
    fh = [i*probs[i] for i in range(len(probs))]
    return sum(fh)/NRES

def plot_util(temp_range, seqs):
    fh_arr = []
    for t in temp_range:
        fe, probs = FE(t, seqs)
        fh_arr.append(fraction_helicity(probs))
        plt.plot(fe, label=f"{t} K")
    
    plt.xlabel('Residue')
    plt.ylabel('Free Energy (kJ/mol)')
    plt.title('Free Energy vs Residue')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('folded_struct_var.png')
    plt.close()

    # plt.show()

    plt.plot(temp, fh_arr)
    plt.xticks(temp)
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'Fraction Helicity $\theta_H$')
    plt.title('Fraction Helicity vs Temperature')
    plt.grid()
    plt.tight_layout()
    plt.savefig('temp_var.png')
    plt.close()
    # plt.show()

def plot_util_two_seq(temp_range, seqs):
    fh_arr = []
    for t in temp_range:
        fe, probs = FE_two_seq(t, seqs)
        fh_arr.append(fraction_helicity(probs))
        plt.plot(fe, label=f"{t} K")
    
    plt.xlabel('Residue')
    plt.ylabel('Free Energy (kJ/mol)')
    plt.title('Free Energy vs Residue (with 2 helical regions)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('folded_struct_var_two_seq.png')
    plt.close()

    # plt.show()

    plt.plot(temp, fh_arr)
    plt.xticks(temp)
    plt.xlabel('Temperature (K)')
    plt.ylabel(r'Fraction Helicity $\theta_H$')
    plt.title('Fraction Helicity vs Temperature (with 2 helical regions)')
    plt.grid()
    plt.tight_layout()
    plt.savefig('temp_var_two_seq.png')
    plt.close()

if __name__ == '__main__':

   # Use the existing variables

    temp = range(TSTART, TEND+TSTEP, TSTEP)
    sequences = get_all_sequences()

    plot_util(temp, sequences)
    plot_util_two_seq(temp, sequences)

    FE_s_var(298, sequences)

