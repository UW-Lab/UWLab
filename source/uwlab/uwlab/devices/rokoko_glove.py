# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import socket
import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.devices import DeviceBase
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

if TYPE_CHECKING:
    from uwlab.devices import RokokoGlovesCfg


class RokokoGlove(DeviceBase):
    """A Rokoko_Glove controller for sending SE(3) commands as absolute poses of hands individual part
    This class is designed to track hands and fingers's pose from rokoko gloves.
    It uses the udp network protocol to listen to Rokoko Live Studio data gathered from Rokoko smart gloves,
    and process the data in form of torch Tensor.
    Addressing the efficiency and ease to understand, the tracking will only be performed with user's parts
    input, and all Literal of available parts is exhaustively listed in the comment under method __init__.

    available tracking literals:
        LEFT_HAND:
            leftHand, leftThumbProximal, leftThumbMedial, leftThumbDistal, leftThumbTip,
            leftIndexProximal, leftIndexMedial, leftIndexDistal, leftIndexTip,
            leftMiddleProximal, leftMiddleMedial, leftMiddleDistal, leftMiddleTip,
            leftRingProximal, leftRingMedial, leftRingDistal, leftRingTip,
            leftLittleProximal, leftLittleMedial, leftLittleDistal, leftLittleTip

        RIGHT_HAND:
            rightHand, rightThumbProximal, rightThumbMedial, rightThumbDistal, rightThumbTip
            rightIndexProximal, rightIndexMedial, rightIndexDistal, rightIndexTip,
            rightMiddleProximal, rightMiddleMedial, rightMiddleDistal, rightMiddleTip
            rightRingProximal, rightRingMedial, rightRingDistal, rightRingTip,
            rightLittleProximal, rightLittleMedial, rightLittleDistal, rightLittleTip
    """

    def __init__(
        self,
        cfg: RokokoGlovesCfg,  # Make sure this matches the port used in Rokoko Studio Live
        device="cuda:0",
    ):
        """Initialize the Rokoko_Glove Controller.
        Be aware that current implementation outputs pose of each hand part in the same order as input list,
        but parts come from left hand always come before parts from right hand.

        Args:
            UDP_IP: The IP Address of network to listen to, 0.0.0.0 refers to all available networks
            UDP_PORT: The port Rokoko Studio Live sends to
            left_hand_track: the tracking point of left hand this class will be tracking.
            right_hand_track: the tracking point of right hand this class will be tracking.
            scale: the overall scale for the hand.
            proximal_offset: the inter proximal offset that shorten or widen the spread of hand.
        """
        import lz4.frame

        self.lz4frame = lz4.frame
        self.device = device
        self._additional_callbacks = dict()
        # Define the IP address and port to listen on
        self.UDP_IP = cfg.UDP_IP
        self.UDP_PORT = cfg.UDP_PORT
        self.scale = cfg.scale
        self.proximal_offset = cfg.proximal_offset
        self.thumb_scale = cfg.thumb_scale
        self.left_fingertip_names = cfg.left_hand_track
        self.right_fingertip_names = cfg.right_hand_track
        self.command_type = cfg.command_type

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

        self.normalize_position = True
        self.normalize_rotation = True

        print(f"Listening for UDP packets on {self.UDP_IP}:{self.UDP_PORT}")

        # fmt: off
        self.left_hand_joint_names = [
            'leftHand',
            'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
            'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
            'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
            'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
            'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip']

        self.right_hand_joint_names = [
            'rightHand',
            'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
            'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
            'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
            'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
            'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip']
        # fmt: on

        self.left_joint_dict = {self.left_hand_joint_names[i]: i for i in range(len(self.left_hand_joint_names))}
        self.right_joint_dict = {self.right_hand_joint_names[i]: i for i in range(len(self.right_hand_joint_names))}

        self.left_finger_dict = {i: self.left_joint_dict[i] for i in self.left_fingertip_names}
        self.right_finger_dict = {i: self.right_joint_dict[i] for i in self.right_fingertip_names}

        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)
        self.fingertip_poses = torch.zeros(
            (len(self.left_hand_joint_names) + len(self.right_hand_joint_names), 7), device=self.device
        )
        output_indices_list = [
            *[self.right_joint_dict[i] for i in self.left_fingertip_names],
            *[self.right_joint_dict[i] + len(self.left_hand_joint_names) for i in self.right_fingertip_names],
        ]
        self.output_indices = torch.tensor(output_indices_list, device=self.device)

        # necessary joints for compute normalization
        self.necessary_joints = []
        if len(self.right_fingertip_names):
            self.necessary_joints.extend(["rightIndexProximal", "rightMiddleProximal", "rightHand"])
        if len(self.left_fingertip_names):
            self.necessary_joints.extend(["leftIndexProximal", "leftMiddleProximal", "leftHand"])

    def reset(self):
        "Reset Internal Buffer"
        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)

    def advance(self):
        """Provides the properly scaled, ordered, selected tracking results received from Rokoko Studio.

        Returns:
            A tuple containing the 2D (n,7) pose array ordered by user inputted joint track list, and a dummy truth value.
        """
        self.left_fingertip_poses = torch.zeros((len(self.left_hand_joint_names), 7), device=self.device)
        self.right_fingertip_poses = torch.zeros((len(self.right_hand_joint_names), 7), device=self.device)
        body_data = self._get_gloves_data()

        for joint_name in self.left_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.left_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.left_fingertip_poses[self.left_joint_dict[joint_name]][3:] = joint_rotation

        for joint_name in self.right_fingertip_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][3:] = joint_rotation
            # for normalization purpose

        for joint_name in self.necessary_joints:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][3:] = joint_rotation

        left_wrist_position = self.left_fingertip_poses[0][:3]
        if len(self.left_fingertip_names) > 0:
            # scale
            self.left_fingertip_poses[:, :3] = (
                self.left_fingertip_poses[:, :3] - left_wrist_position
            ) * self.scale + left_wrist_position
            # reposition
            leftIndexProximalIdx = self.left_joint_dict["leftIndexProximal"]
            leftMiddleProximalIdx = self.left_joint_dict["leftMiddleProximal"]
            leftRingProximalIdx = self.left_joint_dict["leftRingProximal"]
            leftLittleProximalIdx = self.left_joint_dict["leftLittleProximal"]

            reposition_vector = (
                self.left_fingertip_poses[leftMiddleProximalIdx][:3]
                - self.left_fingertip_poses[leftIndexProximalIdx][:3]
            )
            self.left_fingertip_poses[leftIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.left_fingertip_poses[leftLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.fingertip_poses[: len(self.left_fingertip_poses)] = self.left_fingertip_poses

        right_wrist_position = self.right_fingertip_poses[0][:3]
        if len(self.right_fingertip_names) > 0:
            rightThumbProximalIdx = self.right_joint_dict["rightThumbProximal"]
            rightIndexProximalIdx = self.right_joint_dict["rightIndexProximal"]
            rightMiddleProximalIdx = self.right_joint_dict["rightMiddleProximal"]
            rightRingProximalIdx = self.right_joint_dict["rightRingProximal"]
            rightLittleProximalIdx = self.right_joint_dict["rightLittleProximal"]
            # normalize
            rot_matrix = self.normalize_hand_positions(self.right_joint_dict)
            position_normalized_poses = self.right_fingertip_poses[:, :3] - right_wrist_position
            if self.normalize_rotation:
                self.right_fingertip_poses[:, :3] = position_normalized_poses @ rot_matrix
            # scale
            self.right_fingertip_poses[:, :3] = self.right_fingertip_poses[:, :3] * self.scale
            t_idx = rightThumbProximalIdx
            self.right_fingertip_poses[t_idx : t_idx + 4, :3] = (
                self.right_fingertip_poses[t_idx : t_idx + 4, :3] * self.thumb_scale
            )
            if not self.normalize_position:
                self.right_fingertip_poses[:, :3] += right_wrist_position
            # reposition
            reposition_vector = (
                self.right_fingertip_poses[rightMiddleProximalIdx][:3]
                - self.right_fingertip_poses[rightIndexProximalIdx][:3]
            )
            self.right_fingertip_poses[rightIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.right_fingertip_poses[rightLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
            self.fingertip_poses[len(self.left_fingertip_poses) :] = self.right_fingertip_poses
        self.fingertip_poses[self.output_indices] = self.to_world_convention(self.fingertip_poses[self.output_indices])
        if self.command_type == "pos":
            return self.fingertip_poses[self.output_indices][:, :3]
        return self.fingertip_poses[self.output_indices]

    def to_world_convention(self, poses: torch.Tensor):
        # # to world convention
        poses[:, 1] = -poses[:, 1]

        x = torch.tensor([0], device=self.device)
        y = torch.tensor([0], device=self.device)
        z = torch.tensor([0.4], device=self.device)
        quat = quat_from_euler_xyz(x, y, z)
        poses[:, :3] = quat_apply(quat, poses[:, :3])
        return poses

    def normalize_hand_positions(self, hand_keypoints):
        wrist = self.right_fingertip_poses[hand_keypoints["rightHand"]][:3]
        index_proximal = self.right_fingertip_poses[hand_keypoints["rightIndexProximal"]][:3]
        middle_proximal = self.right_fingertip_poses[hand_keypoints["rightMiddleProximal"]][:3]

        # Define the plane with two vectors
        vec1 = index_proximal - wrist
        vec2 = middle_proximal - index_proximal

        # Compute orthonormal basis for the plane
        vec1_normalized = vec1 / vec1.norm()
        vec2_proj = vec2 - torch.dot(vec2, vec1_normalized) * vec1_normalized
        vec2_normalized = vec2_proj / vec2_proj.norm()
        vec3_normalized = torch.linalg.cross(vec1_normalized, vec2_normalized)

        # Construct rotation matrix
        rotation_matrix = torch.stack([vec1_normalized, vec2_normalized, vec3_normalized], dim=-1)

        return rotation_matrix

    def _get_gloves_data(self):
        while True:
            try:
                data, addr = self.sock.recvfrom(8192)  # Buffer size is 1024 bytes
                break
            except OSError as e:
                print(f"Error: {e}")
                continue
        decompressed_data = self.lz4frame.decompress(data)
        received_json = json.loads(decompressed_data)
        # received_json = json.loads(data)
        body_data = received_json["scene"]["actors"][0]["body"]
        return body_data

    def add_callback(self, key: str, func: Callable):
        # check keys supported by callback
        if key not in ["L", "R"]:
            raise ValueError(f"Only left (L) and right (R) buttons supported. Provided: {key}.")
        # TODO: Improve this to allow multiple buttons on same key.
        self._additional_callbacks[key] = func

    def debug_advance_all_joint_data(self):
        """Provides the properly scaled, all tracking results received from Rokoko Studio.
        It is intended to use a debug and visualization function inspecting all data from Rokoko Glove.

        Returns:
            A tuple containing the 2D (42,7) pose array(left:0-21, right:21-42), and a dummy truth value.
        """
        body_data = self._get_gloves_data()

        # for joint_name in self.left_hand_joint_names:
        #     joint_data = body_data[joint_name]
        #     joint_position = torch.tensor(list(joint_data["position"].values()))
        #     joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
        #     self.left_fingertip_poses[self.left_joint_dict[joint_name]][:3] = joint_position
        #     self.left_fingertip_poses[self.left_joint_dict[joint_name]][3:] = joint_rotation

        for joint_name in self.right_hand_joint_names:
            joint_data = body_data[joint_name]
            joint_position = torch.tensor(list(joint_data["position"].values()))
            joint_rotation = torch.tensor(list(joint_data["rotation"].values()))
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][:3] = joint_position
            self.right_fingertip_poses[self.right_joint_dict[joint_name]][3:] = joint_rotation

        # left_wrist_position = self.left_fingertip_poses[0][:3]

        # self.left_fingertip_poses[:, :3] = (self.left_fingertip_poses[:, :3] - left_wrist_position) * self.scale + left_wrist_position
        # self.fingertip_poses[:len(self.left_fingertip_poses)] = self.left_fingertip_poses

        right_wrist_position = self.right_fingertip_poses[0][:3]
        # scale
        rightThumbProximalIdx = self.right_joint_dict["rightThumbProximal"]
        rightIndexProximalIdx = self.right_joint_dict["rightIndexProximal"]
        rightMiddleProximalIdx = self.right_joint_dict["rightMiddleProximal"]
        rightRingProximalIdx = self.right_joint_dict["rightRingProximal"]
        rightLittleProximalIdx = self.right_joint_dict["rightLittleProximal"]
        # scale
        self.right_fingertip_poses[:, :3] = (
            self.right_fingertip_poses[:, :3] - right_wrist_position
        ) * self.scale + right_wrist_position
        t_idx = rightThumbProximalIdx
        self.right_fingertip_poses[t_idx : t_idx + 4, :3] = (
            self.right_fingertip_poses[t_idx : t_idx + 4, :3] - right_wrist_position
        ) * self.thumb_scale + right_wrist_position
        # reposition
        reposition_vector = (
            self.right_fingertip_poses[rightMiddleProximalIdx][:3]
            - self.right_fingertip_poses[rightIndexProximalIdx][:3]
        )
        self.right_fingertip_poses[rightIndexProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightMiddleProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightRingProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.right_fingertip_poses[rightLittleProximalIdx:, :3] += self.proximal_offset * reposition_vector
        self.fingertip_poses[len(self.left_fingertip_poses) :] = self.right_fingertip_poses

        return self.fingertip_poses, True
