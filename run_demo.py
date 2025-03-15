import numpy as np
import time
import yaml
import utils
from solver.CILQR import CILQR
import animation
import matplotlib.pyplot as plt


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)

    return config


def ego_state_loader(config):
    return np.array(config['initial_condition']['state'])


def ref_waypoints_loader(config):
    return utils.get_ref_waypoints(config)


def ref_velo_loader(config):
    return np.array(config['reference_trajectory']['velo_ref'])


def obstacle_attr_loader(config):
    obstacle_params = config['obstacle']

    return np.array([obstacle_params['width'], obstacle_params['length'], obstacle_params['d_safe']])


def obstacle_pred_loader(config):
    mpc_params = config['mpc']
    N = mpc_params['N']
    dt = mpc_params['dt']

    obstacle_params = config['obstacle']
    obstacle_state = obstacle_params['init_state']
    obstacle_whba = obstacle_params['wheelbase']

    return utils.const_velo_prediction(obstacle_state, N, dt, obstacle_whba)

def main():
    config = load_config('config.yaml')

    planner = CILQR(config)

    ego_state = ego_state_loader(config)
    test_ego_state = ego_state.tolist();
    ref_waypoints = ref_waypoints_loader(config)
    test_ref_waypoints = ref_waypoints.tolist()
    ref_velo = ref_velo_loader(config)
    test_ref_velo = ref_velo.tolist()
    obstacle_attr = obstacle_attr_loader(config)
    test_obstacle_attr = obstacle_attr.tolist()
    obstacle_pred = obstacle_pred_loader(config)
    test_obstacle_pred = obstacle_pred.tolist()

    solver_start_t = time.process_time()
    opti_u, opti_x = planner.solve(ego_state,
                                   ref_waypoints,
                                   ref_velo,
                                   obstacle_attr,
                                   obstacle_pred
                                   )
    print('----CILQR Solution Time: {} seconds----'.format(time.process_time() - solver_start_t))
    test_opti_u = opti_u.tolist()
    test_opti_x = opti_x.tolist()
    animation.staticShow(test_opti_u, test_opti_x, ref_waypoints)
    animation.vis(config, ref_waypoints, obstacle_attr, obstacle_pred, opti_x)

if __name__ == "__main__":
    main()
