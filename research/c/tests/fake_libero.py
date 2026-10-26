"""A stand-in for RPent's LIBERO env server, Pi0.5 server and SAM3 server: enough of
their RPC surface to build a real ``LiberoToolkit`` and drive its primitives without a
simulator or a GPU.

The world is one end effector. Its success predicate is ``eef_z > SUCCESS_Z``, evaluated
after every env step like LIBERO's ``done = self._check_success()``, so success can turn
false again when the arm moves back down. Each env step moves the end effector by
``action[:3] * STEP_GAIN``.
"""
from __future__ import annotations

import numpy as np

SUCCESS_Z = 1.20
STEP_GAIN = 0.05
CHUNK = 5
META = {"suite": "libero_object_swap", "task": 3, "seed": 45, "max_episode_steps": 10000}


class FakeLiberoRpc:
    """The ``env.*`` RPC methods of ``robots/libero/env_server.py`` (single env)."""

    def __init__(self, meta: dict | None = None, start_z: float = 1.00):
        self.meta = dict(meta or META)
        self.start_z = start_z
        self.calls: list[str] = []
        self._reset_world()

    def _reset_world(self):
        self.eef = np.array([0.0, 0.0, self.start_z], dtype=np.float64)
        self.steps = 0

    def _success(self) -> bool:
        return bool(self.eef[2] > SUCCESS_Z)

    def _obs(self) -> dict:
        states = np.zeros(9, dtype=np.float32)
        states[:3] = self.eef
        states[6], states[7] = 0.04, -0.04
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        return {"main_images": img, "wrist_images": img, "states": states,
                "task_descriptions": "put the bowl on the plate"}

    def _raw(self) -> dict:
        return {"robot0_eef_pos": self.eef.copy(), "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
                "robot0_gripper_qpos": np.array([0.04, -0.04]), "bowl_1_pos": np.zeros(3),
                "agentview_image": np.zeros((32, 32, 3), dtype=np.uint8),
                "agentview_depth": np.ones((32, 32), dtype=np.float32),
                "robot0_eye_in_hand_image": np.zeros((32, 32, 3), dtype=np.uint8),
                "robot0_eye_in_hand_depth": np.ones((32, 32), dtype=np.float32)}

    def _one_step(self, action):
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        self.eef[:3] += np.clip(a[:3], -1, 1) * STEP_GAIN
        self.steps += 1
        return self._success(), self.steps >= self.meta["max_episode_steps"]

    def call(self, method, args=(), kwargs=None, timeout_s=None):
        kwargs = kwargs or {}
        self.calls.append(method)
        if method == "env.get_env_meta":
            return dict(self.meta)
        if method == "env.reset":
            self._reset_world()
            return self._obs(), {}
        if method == "env.step":
            term, trunc = self._one_step(args[0])
            return self._obs(), np.float32(term), np.bool_(term), np.bool_(trunc), {}
        if method == "env.chunk_step":
            terms, truncs, frames = [], [], []
            for a in np.asarray(args[0]):
                t, tr = self._one_step(a)
                terms.append(t); truncs.append(tr); frames.append(self._obs())
            obs = frames if kwargs.get("return_all_frames") else frames[-1]
            return obs, np.asarray(terms, dtype=np.float32), np.asarray(terms), np.asarray(truncs), {}
        if method == "env.raw_obs":
            return self._raw()
        if method == "env.get_camera_meta":
            return {}
        if method == "env.render_camera":
            h, w = min(kwargs.get("height", 64), 64), min(kwargs.get("width", 64), 64)
            rgb = np.zeros((h, w, 3), dtype=np.uint8)
            return (rgb, np.ones((h, w), dtype=np.float32)) if kwargs.get("depth") else rgb
        if method == "env.get_task_language":
            return "put the bowl on the plate"
        if method == "env.check_success":
            return self._success()
        raise KeyError(f"fake env server has no method {method}")


class FakeVla:
    """Pi0.5 stand-in: every chunk lifts the end effector by ``lift`` per step."""

    def __init__(self, lift: float = 1.0):
        self.lift = lift

    def predict_action_batch(self, obs, mode="eval"):
        actions = np.zeros((CHUNK, 7), dtype=np.float32)
        actions[:, 2] = self.lift
        return actions, {}
