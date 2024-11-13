from algorithms import FedAvg_SGD_ZO

import meta_data as md

def judge_whether_print(current_round):
    if current_round <= 10:
        return True
    elif current_round <= 200:
        return current_round % 10 == 0
    elif current_round <= 500:
        return current_round % 50 == 0
    elif current_round <= 2000:
        return current_round % 100 == 0
    elif current_round <= 4000:
        return current_round % 200 == 0
    else:
        return current_round % 500 == 0

