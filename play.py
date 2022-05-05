from Runners.Human_play import run_game


def main():
    # True to auto-move when only one move option
    # False to play all moves
    skip_single = True

    # Path from content route to weights for agent to play against
    weights = "model_output/DQN_4SP.hdf5"
    run_game(skip_single, weights)


if __name__ == '__main__':
    main()
