import asyncio
import websockets
import json

import gymnasium as gym

X_POS_ADDRESS, Y_POS_ADDRESS = 0xD362, 0xD361
MAP_N_ADDRESS = 0xD35E

class StreamWrapper(gym.Wrapper):
    def __init__(self, env, stream_metadata={}):
        super().__init__(env)
        self.ws_address = "wss://transdimensional.xyz/broadcast"
        self.stream_metadata = stream_metadata
        self.ws_timeout_s = float(stream_metadata.get("ws_timeout_s", 1.0))
        self.include_extra = bool(stream_metadata.get("include_extra", True))
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.websocket = None
        self.loop.run_until_complete(
            self.establish_wc_connection()
        )
        self.upload_interval = int(stream_metadata.get("upload_interval", 120))
        self.steam_step_counter = 0
        self.env = env
        self.coord_list = []
        if hasattr(env, "pyboy"):
            self.emulator = env.pyboy
        elif hasattr(env, "game"):
            self.emulator = env.game
        else:
            raise Exception("Could not find emulator!")

    def step(self, action):

        x_pos = self.emulator.memory[X_POS_ADDRESS]
        y_pos = self.emulator.memory[Y_POS_ADDRESS]
        map_n = self.emulator.memory[MAP_N_ADDRESS]
        self.coord_list.append([x_pos, y_pos, map_n])

        if self.steam_step_counter >= self.upload_interval:
            if self.include_extra:
                self.stream_metadata["extra"] = self.env.get_lead_pokemon_info()
            self.loop.run_until_complete(
                self.broadcast_ws_message(
                    json.dumps(
                        {
                          "metadata": self.stream_metadata,
                          "coords": self.coord_list
                        }
                    )
                )
            )
            self.steam_step_counter = 0
            self.coord_list = []

        self.steam_step_counter += 1

        return self.env.step(action)

    async def broadcast_ws_message(self, message):
        if self.websocket is None:
            await self.establish_wc_connection()
        if self.websocket is not None:
            try:
                await asyncio.wait_for(self.websocket.send(message), timeout=self.ws_timeout_s)
            except websockets.exceptions.WebSocketException as e:
                self.websocket = None
            except asyncio.TimeoutError:
                self.websocket = None

    async def establish_wc_connection(self):
        try:
            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.ws_address,
                    open_timeout=self.ws_timeout_s,
                    close_timeout=self.ws_timeout_s,
                    ping_timeout=self.ws_timeout_s,
                ),
                timeout=self.ws_timeout_s,
            )
        except:
            self.websocket = None
