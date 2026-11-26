# DronePendulumRL
This is Percy's work on MBRL.
This branch is implementation in 3D space MBRL, similar to hw4.

# Work Plan
1. Implement Mujoco Simulation (drone, pendulum, environment with no obstacles). Assume the drone has a low-level controller which takes in the following inputs:
T: collective thrust (scalar)
p: desired roll rate (scalar)
q: desired pitch rate (scalar)
r: desired yaw rate (scalar)

For simplicity, assume that the drone reacts instantaneously for now (i.e. given a setpoint, it is reached immediately at the next tick in the simulation.)

To test the simulation, have the following scenario:
-Let the drone take on a random thrust, roll, pitch, and yaw (fixed throughout the simulation) and let it run with those values; do not fix the seed, just use a random one every time

Run the obstacle-free Starling 2 Max demo from the repository root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=src python scripts/run_ideal_rate.py
```

The MuJoCo viewer opens by default and runs for 60 simulated seconds.
`--duration 5` to change the run length.

2. Define state and action variables:

State Variables: pos [roll, pitch, yaw, theta1, theta2], vel [vx, vy, vz, p, q, r, omega1, omega2], accelerations [ax, ay, az]
Note: theta and omegas refer to the double pendulum joint angles and angular velocities
Note: p,q, and r are the angular velocities of roll, pitch, and yaw

To start, let the policy know the pendulum angles and velocities; leave out the accelerations. Lets call this "Simple Model".
Later on, try removing this information and replacing it with a k-length history instead of the other variables; include the accelerations. Lets call this "Advanced Model".


Action Variables: thrust (collective), p_des, q_des, r_des

3. Define the bounds of the variables in training randomization

Drone
>Position: Start at 5m above the floor; x=0, y=0; the drone should never collide with the floor
>Angular Position: [-30,30] degrees in roll and pitch; full [0,360] in yaw, encode yaw as cos(yaw) and sin(yaw) to avoid discontinuity when the angles circle back from 359 to 0; uniformly sampled
>Velocities: [-1,1] m/s for vx, vy, and vz; uniformly sampled

Pendulum
>Position: [-50,50] degrees from vertical for theta1 relative to the drone, where 0 is straight down from the drone; [-120,120] relative to theta1 for theta2; uniformly sampled
>Angular Velocities: [-60,60] degrees/s for omega1, omega2; uniformly sampled

Actions
>Collective Thrust: [0.6,1.4] times the hover thrust; uniformly sampled
>Angular Velocities: [-20,20] degrees/s for p,q, and r; uniformly sampled

Control Interval
>Let's start with 0.1s for now (~10Hz control seems reasonable for random shooting?)

Pendulum Parameters:
Link 1 Length: 100mm
Link 2 Length: 75mm
Mass 1 (at the end of link 1): 79g
Mass 2 (at the end of link 2): 49g
Link 1 mass (not including mass 1): 38g
Link 2 mass (not including mass 1 nor mass 2): 19g

Note: gravity always 9.81 downwards; air resistance enabled

4. Train prediction model s' (s,a) on randomly generated poses

5. Define a reward function.

-> Come up with a reward function here

6. Check that we aren't going out of domain after training on random dataset, otherwise we are going to have to train based on actions chosen by model like in homework 4

7. Run the policy and see how well it works



