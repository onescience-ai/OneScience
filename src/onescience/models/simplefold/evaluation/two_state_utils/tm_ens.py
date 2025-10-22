#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import subprocess
import logging
import numpy as np


def exec_subproc(cmd, timeout: int = 100) -> str:
    """Execute the external docking-related command.
    """
    if not isinstance(cmd, str):
        cmd = ' '.join([str(entry) for entry in cmd])
    try:
        rtn = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, timeout=timeout)
        out, err, return_code = str(rtn.stdout, 'utf-8'), str(rtn.stderr, 'utf-8'), rtn.returncode
        if return_code != 0:
            logging.error('[ERROR] Execuete failed with command "' + cmd + '", stderr: ' + err)
            raise ValueError('Return code is not 0, check input files')
        return out
    except subprocess.TimeoutExpired:
        logging.error('[ERROR] Execuete failed with command ' + cmd)
        raise ValueError(f'Timeout({timeout})')


def parse_tmscore_output(out: str) -> dict:
    """Parse TM-score output.
    Args:
        out: str in utf-8 encoding from stdout of TM-score.
    """
    start = out.find('RMSD')
    end = out.find('rotation')
    out = out[start:end]
    try:
        rmsd, _, tm, _, gdt_ts, gdt_ha, _, _ = out.split('\n')
    except:
        raise ValueError(f'TM-score output format is not correct:\n{out}')
    
    rmsd = float(rmsd.split('=')[-1])
    tm = float(tm.split('=')[1].split()[0])
    gdt_ts = float(gdt_ts.split('=')[1].split()[0])
    gdt_ha = float(gdt_ha.split('=')[1].split()[0])
    return {'rmsd': rmsd, 'tm': tm, 'gdt_ts': gdt_ts, 'gdt_ha': gdt_ha}


def _tmscore(model_pdb_path, native_pdb_path, TMscore_bin, return_dict=False):
    """Calculate TM-score between two PDB files, each containing 
        only one chain and only CA atoms.
    Args:
        model_pdb_path (str): path to PDB file 1 (model)
        native_pdb_path (str): path to PDB file 2 (native)
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float)
    """
    out = exec_subproc([TMscore_bin, '-seq', model_pdb_path, native_pdb_path])
    res = parse_tmscore_output(out)
    if return_dict:
        return res
    return res['tm']


def tm_ensemble_at_N(samples, gt1, gt2, N=5):
    # Note: an increase of N will always show better "coverage" metric but we limit to N 
    if len(samples) != N:
        print(f"Metric is based on {N} only, but got {len(samples)} samples")

    tm_wrt_gt1 = []
    tm_wrt_gt2 = []
    for sample_i in samples:
        tm_wrt_gt1.append(_tmscore(sample_i, gt1))
        tm_wrt_gt2.append(_tmscore(sample_i, gt2))
    tm_ens_val = 0.5 * (np.max(tm_wrt_gt1) + np.max(tm_wrt_gt2))
    tm_native_val = 0.5 * (_tmscore(gt1, gt2) + _tmscore(gt2, gt1))  # Note that TM score is not symmetric
    return {
        "tm_native": tm_native_val,
        "tm_ens": tm_ens_val,
        "tm_sample_gt1": tm_wrt_gt1,
        "tm_sample_gt2": tm_wrt_gt2,
    }


def tm_diversity(samples):
    """Calculate diversity among pairs of modeled protein structures for each score.
    
    Returns:
        dict of scores('rmsd': float, 'tm': float, 'gdt_ts': float, 'gdt_ha': float, 'lddt': float)
    """
    tm_list = []
    for i in range(len(samples)-1):
        for j in range(i+1, len(samples)):
            tm_list.append(_tmscore(samples[i], samples[j]))

    return {
        "tm_div": np.mean(tm_list),
    }
