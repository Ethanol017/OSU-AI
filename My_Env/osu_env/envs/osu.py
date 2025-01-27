import subprocess
import time
import cv2
import gymnasium as gym
from gymnasium import spaces
from evdev import UInput, ecodes as e
import mss
import numpy as np
import requests



class OSU_Env(gym.Env):


    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(72, 128), dtype=np.float32) # 1440 * 2560 / 20
        self.action_space = spaces.Dict({"move_action":spaces.Box(low=-1,high=1,shape=(2,)),"z_action":spaces.Discrete(2),"x_action":spaces.Discrete(2)})

        self.z_action_down = False
        self.x_action_down = False
        self.data_previous = {}

    def _get_obs():
        with mss.mss() as sct:
            monitor = {"top": 160, "left": 0, "width": 2560, "height": 1440}
            screenshot = sct.grab(monitor)
            image = np.array(screenshot)
            image = cv2.resize(image, (0, 0), fx=0.05, fy=0.05)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # show to human
            # _, binary_image = cv2.threshold(resized_img, 80, 255, cv2.THRESH_BINARY)
            # show_image = cv2.resize(resized_img, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
            # cv2.imshow("Grayscale Screen", show_image)
            
            image_np = image.astype('float32') / 255.0  # 將像素值縮放到 [0, 1]
            return image_np
        
    def _get_data():
        response = requests.get("http://localhost:24050/json")  # 4ms
        response.raise_for_status()  
        data = response.json()
        gameplay = data["gameplay"]
        hit = gameplay["hits"]
        data = {
            "300":hit["300"],
            "geki":hit["geki"],
            "100":hit["100"],
            "katu":hit["katu"],
            "50":hit["50"],
            "0":hit["0"],
            "combo":gameplay["combo"]["current"],
            "hp":gameplay["hp"]["normal"],
            "current_time":data["menu"]["bm"]["time"]["current"],
            "full_time":data["menu"]["bm"]["time"]["full"],
            "first_time":data["menu"]["bm"]["time"]["firstObj"]
            }
        
        return data
        

    def reset(self, seed=None,win=False,retry=True,auto_wait_first=True):
        super().reset(seed=seed)
        if retry:
            if win:
                subprocess.run(['xdotool', 'mousemove', '1547', '838'])
                subprocess.run(['xdotool', 'click', '1']) # 左鍵
            else:
                subprocess.run(['xdotool', 'mousemove', '1014', '621'])
                subprocess.run(['xdotool', 'click', '1']) # 左鍵
        else:
            subprocess.run(['xdotool', 'key', 'Escape'])
            subprocess.run(['xdotool', 'key', 'F2'])
            subprocess.run(['xdotool', 'key', 'Enter'])
                  
        if auto_wait_first:
            data = self._get_data()
            time.sleep( data["first_time"]/1000 )
        else: # human skip 
            time.sleep(2)
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        # Action  ( 1 move : ~4ms )
        move_action = action["move_action"]
        z_action = action["z_action"]    
        x_action = action["x_action"]
        if self.z_action_down != z_action:
            if z_action:
                subprocess.run(['xdotool', 'keydown', 'z'])
            else:
                subprocess.run(['xdotool', 'keyup', 'z']) 
        if self.x_action_down != x_action:
            if x_action:
                subprocess.run(['xdotool', 'keydown', 'x'])
            else:
                subprocess.run(['xdotool', 'keyup', 'x']) 
        subprocess.run(['xdotool', 'mousemove_relative', f'{move_action[0]}', f'{move_action[1]}'])
        
        # Get_Observation & Get_Data
        observation = self._get_obs() # 40ms
        data = self._get_data() # 11ms
        
        # Check_Over & reward
        reward = 0
        
        if data["hp"] == '0': # game over
            truncated = True
            pass  
        elif data["current_time"] >= data["full_time"]: # win
            terminated = True
            pass 
        else:
            terminated = False 
            truncated = False
            pass #continum
        
        
        # Other
        self.data_previous = data
        self.z_action_down = z_action
        self.x_action_down = x_action
        
        return observation, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        # self.gosuemory_process.terminate()
        # print("close")
        cv2.destroyAllWindows()
        pass
