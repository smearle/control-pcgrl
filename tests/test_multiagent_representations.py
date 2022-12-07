import pytest
from random import randint
from copy import deepcopy
from pathlib import Path
from itertools import permutations, product
import numpy as np
from control_pcgrl import wrappers
from control_pcgrl.rl.envs import make_env

@pytest.fixture
def basic_env_config():
    return {
        'problem': {
            'name': 'binary',
            'weights': {'path_length': 100},
            'controls': '???',
            'alp_gmm': '???',
        },
        'hardware': {'n_cpu': 1, 'n_gpu': 1, 'num_envs_per_worker': 10}, 
        'multiagent': {'n_agents': 2},
        'representation': 'turtle',
        'map_shape': [16, 16],
        'crop_shape': [32, 32],
        'max_board_scans': 3,
        'n_aux_tiles': 0,
        'evaluation_env': False,
        'observation_size': None,
        'controls': None,
        'change_percentage': None,
        'static_prob': None,
        'action_size': None,
        'log_dir': Path('./')
    }


def validate_turtle_actions(actions, old_positions, new_positions, old_map, new_map):
    map_shape = old_map.shape
    def validate_move(action, old_position, new_position):
        if action == 0:
            if old_position[0] == 0:
                assert new_position[0] == old_position[0]
            else:
                assert old_position[0] - new_position[0] == 1 
            assert old_position[1] - new_position[1] == 0
        elif action == 1:
            if old_position[0] == map_shape[0] - 1:
                assert new_position[0] == old_position[0]
            else:
                assert old_position[0] - new_position[0] == -1
            assert old_position[1] - new_position[1] == 0
        elif action == 2:
            if old_position[1] == 0:
                assert new_position[1] == old_position[1]
            else:
                assert old_position[1] - new_position[1] == 1
            assert old_position[0] - new_position[0] == 0
        elif action == 3:
            if old_position[1] == map_shape[1] - 1:
                assert new_position[1] == old_position[1]
            else:
                assert old_position[1] - new_position[1] == -1
            assert old_position[0] - new_position[0] == 0
    
    for agent, old_pos, new_pos in zip(actions, old_positions, new_positions):
        action = actions[agent]
        if action < 4:
            validate_move(action, old_pos, new_pos)
        else:
            # position shouldn't change when we place an item
            assert tuple(old_pos) == tuple(new_pos)
            assert new_map[tuple(new_pos)] == action - 4

@pytest.mark.parametrize(
    'action_0,action_1',
    permutations(list(range(6)), 2)
)
def test_multiagent_turtle(basic_env_config, action_0, action_1):
    # GIVEN
    env_config = basic_env_config
    env_name = 'binary-turtle-v0'
    env = wrappers.CroppedImagePCGRLWrapper(env_name, **env_config)
    env = wrappers.MultiAgentWrapper(env, **env_config)
    actions = {'agent_0': action_0, 'agent_1': action_1}
    env.reset()
    rep = env.unwrapped._rep
    init_positions = deepcopy(rep.get_positions())
    #init_positions = deepcopy(rep._positions)
    init_map = deepcopy(rep.rep._map)
    
    # WHEN
    rep.update(actions)

    # THEN
    validate_turtle_actions(
            actions,
            init_positions,
            rep._positions,
            init_map,
            rep.rep._map
            )

@pytest.mark.parametrize(
    'action_0,action_1',
    permutations(list(range(2)), 2)
)
def test_multiagent_narrow(basic_env_config, action_0, action_1):
    # GIVEN
    env_config = basic_env_config
    env_config['representation'] = 'narrow'
    env_name = 'binary-narrow-v0'
    env = wrappers.CroppedImagePCGRLWrapper(env_name, **env_config)
    env = wrappers.MultiAgentWrapper(env, **env_config)
    env.reset()
    rep = env.unwrapped._rep
    init_map = deepcopy(rep.rep._map)
    init_positions = deepcopy(rep._positions)
    actions = {'agent_0': action_0, 'agent_1': action_1}
    # WHEN
    rep.update(actions)
    new_map = rep.rep._map

    # THEN
    new_positions = rep.get_positions()
    # check that position is updated correctly
    # Note: Test does not account for changes in vertical position
    assert new_positions[0][1] - 1 == init_positions[0][1]
    assert new_positions[1][1] - 1 == init_positions[1][1]
    # check that map is updated correctly
    assert new_map[tuple(init_positions[0])] == actions['agent_0']
    assert new_map[tuple(init_positions[1])] == actions['agent_1']

# INCOMPLETE TEST
@pytest.mark.parametrize(
    'position_x_0,position_y_0,action_0,position_x_1,position_y_1,action_1',
     [[randint(0, 15), randint(0, 15), randint(0, 1), randint(0, 15), randint(0, 15), randint(0, 1)]]
)
def test_multiagent_wide(
    basic_env_config,
    position_x_0,
    position_y_0,
    action_0,
    position_x_1,
    position_y_1,
    action_1
    ):
    # GIVEN
    env_config = basic_env_config
    env_config['representation'] = 'wide'
    env_name = 'binary-wide-v0'
    env = wrappers.ActionMapImagePCGRLWrapper(env_name, **env_config)
    env = wrappers.MultiAgentWrapper(env, **env_config)
    env.reset()
    rep = env.unwrapped._rep
    init_map = deepcopy(rep.rep._map)
    init_positions = deepcopy(rep._positions)
    actions = {
        'agent_0': [position_y_0, position_x_0, action_0],
        'agent_1': [position_y_1, position_x_1, action_1]
        }

    # WHEN
    rep.update(actions)

    # THEN check that map is changed correctly
    new_map = rep.rep._map
    assert new_map[position_y_0][position_x_0] == action_0
    assert new_map[position_y_1][position_x_1] == action_1
    # make sure that the map being modified is the same one that the pcgrl env uses
    np.testing.assert_array_equal(new_map, rep.unwrapped._map)


def test_rep_seeding():
    """
    When creating a new environment, how can we seed it so that we get the same level map
    """
    pass
