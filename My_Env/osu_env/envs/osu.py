import subprocess
import time
import cv2
import gymnasium as gym
from gymnasium import spaces
import mss
import numpy as np
import requests


class OSU_Env(gym.Env):

    def __init__(self, **kwargs):
        self.human_play_test = kwargs.get("human_play_test", False)
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 72, 128), dtype=np.float32)  # 1440 * 2560 / 20
        self.action_space = spaces.Dict({"move_action": spaces.Box(low=-1, high=1, shape=(2,)), 
                                         "click_action": spaces.Discrete(2)})

        self.z_action_down = 0
        self.x_action_down = 0
        self.step_dates = []
        self.step_count = 0

    def _get_obs(self):
        with mss.mss() as sct:
            # monitor = {"top": 160, "left": 0, "width": 2560, "height": 1440}
            monitor = {"top": 0, "left": 0, "width": 2560, "height": 1440}
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)
            image = cv2.resize(image, (0, 0), fx=0.05, fy=0.05)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # show to human
            # _, binary_image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY)
            # show_image = cv2.resize(image, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("Grayscale Screen", show_image)

            image_np = image.astype('float32') / 255.0 # normalize to [0, 1]
            image_np = np.expand_dims(image_np, axis=0)
            return image_np

    def _get_data(self):
        response = requests.get("http://localhost:24050/json")  # 4ms
        response.raise_for_status()
        data = response.json()
        gameplay = data["gameplay"]
        hit = gameplay["hits"]
        data = {
            "300": hit["300"],
            "geki": hit["geki"],
            "100": hit["100"],
            "katu": hit["katu"],
            "50": hit["50"],
            "0": hit["0"],
            "combo": gameplay["combo"]["current"],
            "hp": gameplay["hp"]["normal"],
            "current_time": data["menu"]["bm"]["time"]["current"],
            "full_time": data["menu"]["bm"]["time"]["full"],
            "first_time": data["menu"]["bm"]["time"]["firstObj"],
            "songover_time": data["menu"]["bm"]["time"]["mp3"],
            "score": gameplay["score"],
        }

        return data

    def _get_different_data(self, data, data_previous):
        data_diffient = {}
        for key, value in data.items():
            data_diffient[key] = value - data_previous[key]
        return data_diffient

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Parameters:
        - seed: Random seed for the environment.
        - options: Dictionary of options to control the reset behavior.
            - win (bool): Whether the previous game was won. Default is False.
            - retry (bool): Whether to retry the game. Default is False.
            - auto_wait_first (bool): Whether to automatically wait for the first object. Default is True.
        """
        super().reset(seed=seed, options=options)
        
        self.step_count = 0
        self.z_action_down = 0
        self.x_action_down = 0
        
        options = options or {}
        start = options.get("start", False)
        iswin = options.get("win", False)
        retry = options.get("retry", False)
        auto_wait_first = options.get("auto_wait_first", True)
        
        subprocess.run(['xdotool','keyup','z'])
        subprocess.run(['xdotool','keyup','x'])
        
        if start:
            # subprocess.run(['xdotool', 'key', 'F2'])
            subprocess.run(['xdotool', 'key', 'Enter'])
        else:
            # wait cutscene
            if iswin:
                data = self._get_data()
                time.sleep((data["songover_time"] - data["full_time"])/1000)
            time.sleep(3) # wait to show end menu 
            # start movement
            if retry:
                if iswin:
                    subprocess.run(['xdotool', 'key', 'Escape'])
                    time.sleep(1)
                    subprocess.run(['xdotool', 'key', 'Enter'])
                else:
                    subprocess.run(['xdotool', 'key', 'Down'])
                    subprocess.run(['xdotool', 'key', 'Enter'])
            else:  # random 
                subprocess.run(['xdotool', 'key', 'Escape'])
                time.sleep(1)
                subprocess.run(['xdotool', 'key', 'F2'])
                subprocess.run(['xdotool', 'key', 'Enter'])

        time.sleep(1) # wait for game start cutscene
        subprocess.run(['xdotool', 'mousemove', f'{2560/2}',f'{1440/2}'])
        data = self._get_data()
        if auto_wait_first:
            time.sleep(max(0 , ((data["first_time"]/1000)-1) ))
        else:  # human skip
            time.sleep(2)
        self.step_dates.clear()
        self.step_dates.append(data)
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # Action  ( 1 move : ~4ms )
        MOVE_SPEED = 0.2 #  [0,1] x (2560,1440) x MOVE_SPEED
        if self.human_play_test == False:
            move_action = action["move_action"]
            z_action = action["click_action"]
            # x_action = action["x_action"]
            # Key
            if self.z_action_down != z_action:
                if z_action:
                    subprocess.run(['xdotool', 'keydown', 'z'])
                else:
                    subprocess.run(['xdotool', 'keyup', 'z'])
                self.z_action_down = z_action
            # if self.x_action_down != x_action:
            #     if x_action:
            #         subprocess.run(['xdotool', 'keydown', 'x'])
            #     else:
            #         subprocess.run(['xdotool', 'keyup', 'x'])
            #     self.x_action_down = x_action
            # Mouse
            subprocess.run(['xdotool', 'mousemove_relative', '--',
                           f'{move_action[0] * 2560 * MOVE_SPEED}', f'{move_action[1] * 1440 * MOVE_SPEED}'])

        # Get_Observation & Get_Data
        observation = self._get_obs()  # 40ms
        data = self._get_data()  # 11ms
        self.step_dates.append(data)
        if self.step_count != 0:
            data_previous = self.step_dates[self.step_count-1]
        else:
            data_previous = data
        diff = self._get_different_data(data, data_previous)
        
        # Check_Over & reward
        SCORCE = 0.01 # mutiply score
        GAMEOVER = -250
        WIN = 1000
        reward = 0
        terminated = False
        truncated = False
        reward += diff["score"] * SCORCE
        if data["hp"] == 0:  # game over
            truncated = True
            reward += GAMEOVER
        elif data["current_time"] >= data["full_time"]:  # win
            terminated = True
            reward += WIN
        # Other
        self.step_count += 1
        return observation, reward, terminated, truncated, {"step_count":self.step_count}

    def render(self):
        pass

    def close(self):
        # self.gosuemory_process.terminate()
        # print("close")
        cv2.destroyAllWindows()
        pass
