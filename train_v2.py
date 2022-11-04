import os
from datetime import datetime

# from Runners.Ludo_DQN import run_game


# from Runners.Ludo_AC import run_game
# from Runners.Ludo_Reinforce import run_game
from Runners.Ludo_DQN_4p_v2 import run_game
# from Runners.Ludo_AC_Multi import run_game
# from Runners.Run_Small import run_game


def main():
    episodes = 100_001
    run_no = 3
    run_id = s1 = f'{run_no:04d}'

    returns = run_game(episodes, run_id)

    string = print_full(returns, run_id)
    # string = print_small(returns, run_id)

    print("RUN %s complete" % run_id)
    print(string)


def print_full(returns, run_id):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    string = '%s_%s.txt' % (date_time, run_id)
    file_path = "results/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f = open("%s%s" % (file_path, string), "a")

    f.write(','.join(str(e) for e in returns[0]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[1]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[2]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[3]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[4]))
    f.write("\n")
    # f.write(','.join(str(e) for e in returns[5]))
    # f.write("\n")
    f.flush()
    f.close()
    return string


def print_small(returns, run_id):
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    string = '%s_%s.txt' % (date_time, run_id)
    file_path = "results/small/"
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    f = open("results/small/%s.txt" % string, "a")

    f.write(','.join(str(e) for e in returns[0]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[1]))
    f.write("\n")
    f.write(','.join(str(e) for e in returns[2]))
    f.write("\n")
    f.flush()
    f.close()

    return string


# hare run --rm -v "$(pwd)":/code --workdir /code bath:2020 python3 /code/unit_tests.py

if __name__ == '__main__':
    main()
