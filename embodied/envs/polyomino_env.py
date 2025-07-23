import gymnasium as gym
import zmq, json, time
from enum import Enum
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='polyomino_env.log', filemode='a')
logging.info("============== Polyomino Environment initialized ================")

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    ROTATE_COUNTERCLOCKWISE = 4
    ROTATE_CLOCKWISE = 5
    ZOOM_IN = 6
    ZOOM_OUT = 7
    NEXT_SHAPE = 8
    SELECT_SAME = 9
    SELECT_DIFFERENT = 10

class PolyominoEnvironment(gym.Env):
    def __init__(self, PORT = 10002, LISTENER_PORT = 10001, HOST = 'localhost', TIMEOUT = 15000, MSG_TIMEOUT_FILTER = '', MAX_TIMESTEPS = 1000):
        self.ACTION_MAP = {
              'W': 'up',
              'S': 'down',
              'A': 'left',
              'D': 'right',
              'Q': 'rotate_counterclockwise',
              'E': 'rotate_clockwise',
              '+': 'zoom_in',
              '_': 'zoom_out',
              'N': 'next_shape',
              '1': 'select_same_shape',
              '0': 'select_different_shape'
        }
        self.index = 0

        self.ACTION_KEYS = list(self.ACTION_MAP.keys())
        self.ACTION_DESC = list(self.ACTION_MAP.values())
        self.action_space = gym.spaces.Discrete(len(self.ACTION_KEYS))

        self.PORT = PORT
        self.LISTENER_PORT = LISTENER_PORT
        self.HOST = HOST
        self.TIMEOUT = TIMEOUT
        self.MSG_TOPIC_FILTER = MSG_TIMEOUT_FILTER

        self.MAX_TIMESTEPS = MAX_TIMESTEPS
        self.current_timestep = 0

        self.MAX_PROBLEMS = 50
        self.current_problem = 0

        self.SELECTION_ACTIONS = [Actions.SELECT_SAME.value, Actions.SELECT_DIFFERENT.value]

        self.seqno = 0

        self.latest_env_state = None

        self.answered = False
        
        # 128 x 128 pixel images with 1 channel (grayscale)
        self.observation_space = gym.spaces.Dict({
            "left": gym.spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
            "right": gym.spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8),
        })


        self.context = zmq.Context()
        self._connect()
        self._listener_connect()


    def _connect(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.HOST}:{self.PORT}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.TIMEOUT)

    def _listener_connect(self):
        self.listener = self.context.socket(zmq.SUB)
        self.listener.setsockopt_string(zmq.SUBSCRIBE, self.MSG_TOPIC_FILTER)
        self.listener.setsockopt(zmq.RCVTIMEO, self.TIMEOUT)
        self.listener.connect(f"tcp://{self.HOST}:{str(self.LISTENER_PORT)}")

    def _create_request(self, data):
        header = {
            'seqno': self.seqno,
            'time': round(time.time() * 1000) # milliseconds
        }

        return {'header': header, 'data': data}
    
    def _send(self, data):
        self.seqno += 1
        request = self._create_request(data)
        encoded_req = json.dumps(request)
        self.socket.send_string(encoded_req)
        try:
            reply = self.socket.recv_json(flags=0)
        except zmq.Again:
            raise RuntimeError("Timeout waiting for reply on REQ socket")

        return self._wait_for_update(self.seqno) 

    def _recv(self):
        poller = zmq.Poller()
        poller.register(self.listener, zmq.POLLIN)
        socks = dict(poller.poll(self.TIMEOUT))
        if self.listener in socks and socks[self.listener] == zmq.POLLIN:
            msg = self.listener.recv_string()
            ndx = msg.find('{')
            topic, encoded_payload = msg[0:ndx - 1], msg[ndx:]
            payload = json.loads(encoded_payload)
            return topic, payload
        raise zmq.Again("No message received within timeout")

    def _wait_for_update(self, seqNo, timeout_ms=15000):
        end_time = time.time() + (timeout_ms / 1000)
        lastActionSeqNo = -1
        while time.time() < end_time:
            try:
                topic, payload = self._recv()
                print(f"Received topic: {topic}")
            except zmq.Again:
                continue
            if "/state" in topic:
                lastActionSeqNo = payload["data"]["last_action_seqno"]
                if lastActionSeqNo >= seqNo:
                    self.latest_env_state = {
                        'state': [payload['data']['left_viewport']['screenshot'], payload['data']['right_viewport']['screenshot']],
                        'isSame': payload['data']['same']
                    }
                    return payload
        raise TimeoutError(f"Timeout waiting for environment state update with seqNo {seqNo}, Last action seqNo: {lastActionSeqNo}")

    def _check_selection(self, selected_same):
        return self.latest_env_state["isSame"] == selected_same
    
    def calculate_reward(self, action):
        #simplified for test
        if action in self.SELECTION_ACTIONS:
            if self.answered:
                reward = -0.05
            else:
                isCorrect = self._check_selection(action == Actions.SELECT_SAME.value)
                reward = 1 if isCorrect else -1
            self.answered = True
        else:
            if action == Actions.NEXT_SHAPE.value and not self.answered:
                reward = -1
            else:
                reward = -0.05

        return reward

    def reset(self, seed=42):
        data = {
            'event': {
                'type': 'action',
                'value': self.ACTION_DESC[Actions.NEXT_SHAPE.value]
            }
        }
        self._send(data)
        self.current_timestep = 0
        self.current_problem = 0

        left, right = self.latest_env_state["state"]

        self.answered= False

        observation = {
            "left": np.array(left, dtype=np.uint8).reshape(128, 128, 1),
            "right": np.array(right, dtype=np.uint8).reshape(128, 128, 1)
        }


        info = {}
        return (observation, info)

    def step(self, action):
        self.current_timestep += 1
        data = {
            'event': {
                'type': 'action',
                'value': self.ACTION_DESC[action]
            }
        }
        self._send(data)

        reward = self.calculate_reward(action)

        if action == Actions.NEXT_SHAPE.value:
            self.current_problem += 1
            self.answered= False # reset after choosing the next shape

        left, right = self.latest_env_state["state"]

        observation = {
            "left": np.array(left, dtype=np.uint8).reshape(128, 128, 1),
            "right": np.array(right, dtype=np.uint8).reshape(128, 128, 1)
        }


        info = {}
        # terminated = self.MAX_TIMESTEPS <= self.current_timestep;
        terminated = self.MAX_PROBLEMS <= self.current_problem
        truncated = False
        logging.info(f"Observation: {observation}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        return observation, reward, terminated, truncated, info

    def close(self):
        self.socket.close()
        self.listener.close()
        self.context.term()


"""
environment_state: pixel data (screenshot)

are the actions hidden to the agent based on the playMode?; currently agent can send the actions but if the env is not in playMode nothing will change


different approach to listener msgs.: use sqn no to sync the data
"""


