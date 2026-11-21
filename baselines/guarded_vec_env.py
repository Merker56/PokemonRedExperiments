"""SubprocVecEnv that notices when an emulator worker dies or goes silent.

With start_method="fork" (used to save memory), every worker inherits copies of
the other workers' pipe ends. If one worker exits (for example the stall guard in
red_gym_env_v2.py shutting down a stuck emulator), its pipe stays open in its
siblings, so the main process never gets an end-of-file and waits forever.

Here every receive checks, every few seconds, that the worker is still alive and
gives up after `timeout_s`. Either way training stops with a clear error, and
run_overnight.py restarts it from the last checkpoint straight away.
"""
from stable_baselines3.common.vec_env import SubprocVecEnv


class _GuardedConnection:
    def __init__(self, conn, process, index, timeout_s):
        self._conn = conn
        self._process = process
        self._index = index
        self._timeout_s = timeout_s

    def send(self, obj):
        return self._conn.send(obj)

    def recv(self):
        waited = 0.0
        while not self._conn.poll(5.0):
            waited += 5.0
            if not self._process.is_alive():
                raise RuntimeError(
                    f"env worker {self._index} (pid {self._process.pid}) died with exit code "
                    f"{self._process.exitcode}; see runs/stalls/agent_{self._index:02d}.txt")
            if waited >= self._timeout_s:
                raise TimeoutError(f"env worker {self._index} (pid {self._process.pid}) "
                                   f"sent nothing for {waited:.0f} s")
        return self._conn.recv()

    def poll(self, *args, **kwargs):
        return self._conn.poll(*args, **kwargs)

    def close(self):
        return self._conn.close()


class GuardedSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, start_method=None, timeout_s=300):
        super().__init__(env_fns, start_method=start_method)
        self.remotes = [_GuardedConnection(r, p, i, timeout_s)
                        for i, (r, p) in enumerate(zip(self.remotes, self.processes))]
