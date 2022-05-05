import os
from datetime import datetime
from Runners.Ludo_Reinforce import run_game


def main():
    episodes = 40_000
    run_no = 53
    run_id = s1 = f'{run_no:04d}'

    returns = run_game(episodes, run_id)

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

    print("RUN %s complete" % run_id)
    print(string)

# hare run --rm -v "$(pwd)":/code --workdir /code bath:2020 python3 /code/unit_tests.py

# plt.xticks(range(min(min0, min1) - 0.1, max(max0, max1) + 0.1))


if __name__ == '__main__':
    main()
