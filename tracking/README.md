## Tracking Occulusion Handler

Most Autonomous Vehicles use Lidar to track their surroundings. This allows us to also track surrounding vehicles and form a trajectory to predict their future path. However, obstacles can block our tracked vehicle, which leads to broken chains and overall noisy tracking output. My solution was to match these broken tracks by a set of parameters. This can be viewed as a graphing problem and I used the union-find algorithm to do this efficiently. Unfortunately, you will not be able to run this code because it requires a lot of setups. But feel free to take a look at the code. 

The occlusion handler class can be found in `improved.py` and is called in line 65 of `main.py`.