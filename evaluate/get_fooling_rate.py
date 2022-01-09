import re
import argparse
from pathlib import Path

import numpy as np

def get_ap_steps(txt):
    """Get the total attack plan steps from wb text file
    Args:
        txt (list of strs): the txt file
            for example:
            im000006_idx2_f8_t11_ap0_wb[1]_bb[0]
            im000006_idx2_f8_t11_ap1_wb[1]_bb[0]
            im000006_idx2_f8_t11_ap2_wb[1]_bb[1]
            im000006_idx2_f8_t11_ap3_wb[1]_bb[1]
            ...
    Returns:
        n_steps (int): the number of attack plan steps
    """
    n_steps = -1
    for line in txt:
        ap = int(re.findall(r"ap(.+?)\_", line)[0])
        if ap > n_steps:
            n_steps = ap
        else:
            break
    n_steps += 1
    return n_steps


def get_fooling_rate(txt, accumulate=True):
    """Return the success rate of sequential attacks (accumulative or not).
    Args:
        txt (list of strs): the txt file
        accumulate (bool): accumulate results or not
    Returns:
        wb_success (np.mdarray): the counts of successes of wb attacks at each step
        bb_success (np.mdarray): ...
    """
    # get the number of wb and bb
    n_wb = len(eval(re.findall(r"wb(\[.*?\])+?", txt[0])[0]))
    n_bb = len(eval(re.findall(r"bb(\[.*?\])+?", txt[0])[0]))

    # sequential (1 + 5) + oneshot
    n_steps = get_ap_steps(txt)
    wb_success = np.zeros([n_steps,n_wb]).astype(int)
    bb_success = np.zeros([n_steps,n_bb]).astype(int)

    im_visited = set()
    for line_idx, line in enumerate(txt):
        if line.startswith('im'):
            im_id = re.findall(r"im(.+?)\_", line)[0]
            idx = re.findall(r"idx(.+?)\_", line)[0]
            from_class = re.findall(r"f(.+?)\_", line)[0]
            to_class = re.findall(r"t(.+?)\_", line)[0]
            ap = int(re.findall(r"ap(.+?)\_", line)[0])
            wb = eval(re.findall(r"wb(\[.*?\])+?", line)[0])
            bb = eval(re.findall(r"bb(\[.*?\])+?", line)[0])

            if im_id not in im_visited:
                im_visited.add(im_id)
                wb_table = np.zeros([n_steps,n_wb]).astype(int)
                bb_table = np.zeros([n_steps,n_bb]).astype(int)
            wb_table[ap] = wb
            bb_table[ap] = bb
            
            # when all aps of an image is read
            if (line_idx+1) % n_steps == 0:
                if accumulate:
                    for i in range(1,6):
                        wb_table[i] = wb_table[i] | wb_table[i-1]
                        bb_table[i] = bb_table[i] | bb_table[i-1]
                    
                    if n_steps > 6:
                        # if use combine
                        i = 6
                        wb_table[i] = wb_table[i] | wb_table[0]
                        bb_table[i] = bb_table[i] | bb_table[0]
                        
                        for i in range(7,11):
                            wb_table[i] = wb_table[i] | wb_table[i-1]
                            bb_table[i] = bb_table[i] | bb_table[i-1]

                wb_success += wb_table
                bb_success += bb_table
    return wb_success, bb_success


def main():
    parser = argparse.ArgumentParser(description="Calculate the fooling rate.")
    
    parser.add_argument("--eps", nargs="?", default=30, help="perturbation level: 10,20,30,40,50")
    parser.add_argument("--root", nargs="?", default='result', help="the folder name of result")
    parser.add_argument('-bb', action='store_true', help="use bb txt file")
    args = parser.parse_args()
    eps = int(args.eps)
    result_folder = args.root
    use_bb_file = args.bb

    # parse wb train txt and bb test txt
    exp = f"run_sequential_attack_eps{eps}"
    result_root = Path(f"../attacks/{result_folder}/") / exp
    if use_bb_file:
        file = open(result_root / f"{exp}_bb.txt", "r")
    else:
        file = open(result_root / f"{exp}.txt", "r")
    txt = file.readlines()

    # read files and count ap steps for each image
    n_steps = get_ap_steps(txt)
    print(f"n_steps: {n_steps}")
    valid_len = len(txt) // n_steps * n_steps
    txt = txt[:valid_len]
    n_attack = len(txt) // n_steps
    print(f"n_attack: {n_attack}\n")

    print(f"\nfooling rate:")
    wb_success_accum, bb_success_accum = get_fooling_rate(txt, accumulate=True)
    # wb_success_split, bb_success_split = get_fooling_rate(txt, accumulate=False)
    print([f"wb{i}" for i in range(wb_success_accum.shape[1])] + [f"bb{i}" for i in range(bb_success_accum.shape[1])])
    print()
    # fooling_rate = np.hstack((wb_success_accum, bb_success_accum, wb_success_split, bb_success_split)) / n_attack * 100
    fooling_rate = np.hstack((wb_success_accum, bb_success_accum)) / n_attack * 100
    for row in fooling_rate:
        # print(*[f"{item:.2f}" for item in row])
        print('\t'.join([f"{item:.2f}" for item in row]))

if __name__ == "__main__":
    main()