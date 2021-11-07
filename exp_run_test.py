import os
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="testing atari")
    parser.add_argument("--igame", default="A", type=str,
                        help="inital of game")
    parser.add_argument("--use_mem", default=1, type=int,
                        help="mem script or not")
    parser.add_argument("--num_run", default=50, type=int,
                        help="no of eval")
    parser.add_argument("--seed", default=0, type=int,
                        help="random seed")

    args  = parser.parse_args()

    num_run = args.num_run
    seed = args.seed


    agames = [["Alien","Amidar","Asteroids","Atlantis"],
            ["BankHeist","BattleZone","BeamRider","Berzerk","Bowling","Boxing", "Breakout"],
              ["Centipede", "ChopperCommand", "CrazyClimber"],
              ["DemonAttack", "DoubleDunk"],
              ["Enduro"],
              ["FishingDerby", "Freeway", "Frostbite"],
              ["Gopher", "Gravitar"],
              ["Hero"],
              ["IceHockey"],
              ["Jamesbond"],
              ["Kangaroo", "Krull", "KungFuMaster"],
              ["MontezumaRevenge"],["MsPacman"],
              ["NameThisGame"],
              ["Phoenix", "Pitfall", "Pong", "PrivateEye"],
              ["Qbert"],
              ["RoadRunner","Robotank"],
              ["Seaquest", "Skiing", "Solaris","SpaceInvaders"]]

    if args.use_mem==1:
        run_file = "main_mem.py"
        temp_file = "./temp.txt"
    else:
        run_file = "main.py"
        temp_file = "./temp0.txt"



    games = None
    for games in agames:
        if games[0][0] == args.igame:
            break

    with open(f"./test{num_run}{run_file}{seed}{games[0][0]}.txt", "w") as f:
        f.write("Game Score\n")

    for game in games:
        gamen = game+"NoFrameskip-v4"
        if args.use_mem==0:
            command = f"python  {run_file} --double --dueling --evaluation_interval  {num_run}  --evaluate --render --env {gamen} --episode-life 0 --seed {seed}"
            e = os.system(command)
            if e != 0:
                command = f"python  {run_file} --dueling --evaluation_interval  {num_run}  --evaluate --render --env {gamen} --episode-life 0 --seed {seed}"
                e= os.system(command)
            if e != 0:
                command = f"python  {run_file}  --evaluation_interval  {num_run}  --evaluate --render --env {gamen} --episode-life 0 --seed {seed}"
                e = os.system(command)
            if e != 0:
                command = f"python  {run_file} --double --evaluation_interval  {num_run}  --evaluate --render --env {gamen} --episode-life 0 --seed {seed}"
                os.system(command)
        else:
            command = f"python  {run_file} --evaluation_interval  {num_run}  --evaluate --render --env {gamen} --episode-life 0 --seed {seed}"
            os.system(command)
        r = -100000

        with open(temp_file) as f:
            r = f.read()
        with open(f"./test{num_run}{run_file}{seed}{game[0]}.txt", "a") as f:
            f.write(f"{game} {r}\n")


