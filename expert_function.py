import math

def judge_fn(func_type, job_old,job_new):
    old_submit_time = job_old.submit_time
    old_runtime = job_old.run_time
    old_procs = job_old.request_number_of_processors
    new_submit_time = job_new.submit_time
    new_runtime = job_new.run_time
    new_procs = job_new.request_number_of_processors
    if func_type == 0: #HPC2N
        old_score = math.log10(old_runtime) * old_procs + 40 * math.log10(old_submit_time)
        new_score = math.log10(new_runtime) * new_procs + 40 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 1: #SDSC-BLUE-2000
        old_score = 0.0053 * math.sqrt(old_runtime) * 3.4 * 1e-5 * old_procs + 0.0079 * math.log10(old_submit_time)
        new_score = 0.0053 * math.sqrt(new_runtime) * 3.4 * 1e-5 * new_procs + 0.0079 * math.log10(new_submit_time)
        if old_score < new_score:
            return False
    if func_type == 2: #PIK-IPLEX-2009
        old_score = 4.9 * 1e-5 * math.sqrt(old_runtime) + 0.0021 * math.sqrt(old_procs) - 2 * 1e-5 * math.sqrt(old_submit_time)
        new_score = 4.9 * 1e-5 * math.sqrt(new_runtime) + 0.0021 * math.sqrt(new_procs) - 2 * 1e-5 * math.sqrt(new_submit_time)
        if old_score < new_score:
            return False
    return True
