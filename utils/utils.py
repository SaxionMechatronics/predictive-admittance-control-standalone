"""
Utility functions for dynamics and control
"""
import numpy as np
from numpy import matrix as mat
import casadi as ca
from casadi import vertcat, horzcat, norm_2
from casadi import sqrt, cos, sin, atan2, fabs
from typing import Tuple


def rad(angle: float) -> float:
    """Convert angle from degrees to radians"""
    return angle * np.pi / 180


def quat2rot_casadi(q: ca.SX) -> ca.SX:
    """
    Compute rotation matrix from quaternion (CasADi version)
    Args:
        q: Quaternion [qw, qx, qy, qz]
    Returns:
        R: 3x3 rotation matrix (CasADi expression)
    """
    r11 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    r21 = 2 * (q[1]*q[2] + q[0]*q[3])
    r31 = 2 * (q[1]*q[3] - q[0]*q[2])
    r12 = 2 * (q[1]*q[2] - q[0]*q[3])
    r22 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    r32 = 2 * (q[2]*q[3] + q[0]*q[1])
    r13 = 2 * (q[1]*q[3] + q[0]*q[2])
    r23 = 2 * (q[2]*q[3] - q[0]*q[1])
    r33 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return ca.horzcat(
        ca.vertcat(r11, r21, r31),
        ca.vertcat(r12, r22, r32),
        ca.vertcat(r13, r23, r33)
    )


def quat2rot_numpy(q: np.ndarray) -> np.ndarray:
    """
    Compute rotation matrix from quaternion (NumPy version)
    Args:
        q: Quaternion [qw, qx, qy, qz]
    Returns:
        R: 3x3 rotation matrix (numpy array)
    """
    r11 = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    r21 = 2 * (q[1]*q[2] + q[0]*q[3])
    r31 = 2 * (q[1]*q[3] - q[0]*q[2])
    r12 = 2 * (q[1]*q[2] - q[0]*q[3])
    r22 = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    r32 = 2 * (q[2]*q[3] + q[0]*q[1])
    r13 = 2 * (q[1]*q[3] + q[0]*q[2])
    r23 = 2 * (q[2]*q[3] - q[0]*q[1])
    r33 = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2

    return np.array([
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33]
    ])


def hamilton_prod(q1: ca.SX, q2: ca.SX) -> ca.SX:
    """
    Hamilton product (quaternion multiplication) q1 * q2
    Args:
        q1, q2: Quaternions [qw, qx, qy, qz]
    Returns:
        Product quaternion
    """
    return ca.vertcat(
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1],
        q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
    )


def dist_quat(q1: ca.SX, q2: ca.SX) -> ca.SX:
    """
    Compute angular distance between two quaternions
    Args:
        q1, q2: Quaternions [qw, qx, qy, qz]
    Returns:
        Angular distance metric
    """
    q1n = q1 / ca.norm_2(q1)
    q2n = q2 / ca.norm_2(q2)
    return 1 - ca.fabs(ca.dot(q1n, q2n))


def skew_mat(v):
    """
    Compute skew-symmetric matrix from vector
    Args:
        v: 3D vector
    Returns:
        3x3 skew-symmetric matrix
    """
    return vertcat(
        horzcat(0, -v[2], v[1]),
        horzcat(v[2], 0, -v[0]),
        horzcat(-v[1], v[0], 0)
    )


def axis_rot(axis: str, angle: float) -> np.ndarray:
    """
    Compute rotation matrix around a given axis
    Args:
        axis: 'x', 'y', or 'z'
        angle: Rotation angle in radians
    Returns:
        3x3 rotation matrix (numpy array)
    """
    if axis == 'x':
        return np.array([[1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)]])
    elif axis == 'y':
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)]])
    elif axis == 'z':
        return np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
    else:
        raise ValueError(f"Invalid axis: {axis}")

def calc_quat_error(q, q_ref):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    qw_ref, qx_ref, qy_ref, qz_ref = q_ref[0], q_ref[1], q_ref[2], q_ref[3]

    q_ref_conj = vertcat(qw_ref, -qx_ref, -qy_ref, -qz_ref)
    norm_sq_q_ref = casadi.dot(q_ref, q_ref)
    q_ref_inv = q_ref_conj / norm_sq_q_ref

    qrw, qrx, qry, qrz = q_ref_inv[0], q_ref_inv[1], q_ref_inv[2], q_ref_inv[3]

    # Compute the vector part of the quaternion product (q_ref_inv * q)
    # This represents the error quaternion's vector component
    ex = qrw * qx + qrx * qw + qry * qz - qrz * qy
    ey = qrw * qy - qrx * qz + qry * qw + qrz * qx
    ez = qrw * qz + qrx * qy - qry * qx + qrz * qw

    # return qw * qrx + qx * qrw + qy * qrz - qz * qry; 
    # qw * qry - qx * qrz + qy * qrw + qz * qrx; 
    # qw * qrz + qx * qry - qy * qrx + qz * qrw

    # The attitude error is typically defined as 2 * [ex, ey, ez]
    return vertcat(2 * ex, 2 * ey, 2 * ez)


def GTMRP_props(n: int, l: float, alpha: float, beta: float,
                com: list = [0, 0, 0], alpha0: int = -1) -> Tuple[list, list]:
    """
    Compute position and orientation of propellers in a Generically Tilted Multi-Rotor
    Args:
        n: Number of propellers
        l: Distance from propellers to CoM
        alpha: Alpha tilting angle (rad)
        beta: Beta tilting angle (rad)
        com: Position of geometric center wrt CoM
        alpha0: Sign of alpha tilting angle for first propeller
    Returns:
        p: List of position vectors
        R: List of rotation matrices
    """
    R = [axis_rot('z', i * (np.pi / (n / 2))) @
         axis_rot('x', alpha0 * (-1)**i * alpha) @
         axis_rot('y', beta) for i in range(n)]

    p = [l * axis_rot('z', i * (np.pi / (n / 2))) @ np.array([1, 0, 0]) +
         np.array(com) for i in range(n)]

    return p, R


def GTMRP_matrix(R: list, p: list, c_f: list, c_t: list, sign: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute allocation matrix for Generically Tilted Multi-Rotor
    Args:
        R: List of (3x3) orientation matrices
        p: List of (3x1) position vectors
        c_f: Propeller force coefficient(s)
        c_t: Propeller torque coefficient(s)
        sign: Rotation direction for first prop (-1:CCW, 1:CW)
    Returns:
        G_f: Force allocation matrix (3 x n)
        G_t: Torque allocation matrix (3 x n)
    """
    r = range(len(R))
    Riz = [R[i] @ np.array([0, 0, 1]) for i in r]
    G_f = np.column_stack([Riz[i] for i in r])
    G_t = np.column_stack([
        np.cross(p[i], Riz[i]) +
        c_t[i] / c_f[i] * sign * (-1)**i * Riz[i]
        for i in r
    ])

    return np.array(G_f), np.array(G_t)


def alloc(n: int, l: float, alpha: float, beta: float, c_f, c_t,
          com: list = [0, 0, 0], sign: int = 1, alpha0: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute allocation matrices for multi-rotor
    Args:
        n: Number of rotors
        l: Arm length
        alpha: Alpha tilt angle (degrees)
        beta: Beta tilt angle (degrees)
        c_f: Force coefficient (scalar or list)
        c_t: Torque coefficient (scalar or list)
        com: Center of mass offset
        sign: Rotation sign
        alpha0: First rotor tilt sign
    Returns:
        GF: Force allocation matrix
        GT: Torque allocation matrix
    """
    if isinstance(c_f, (float, int)):
        c_f = [c_f for _ in range(n)]
    if isinstance(c_t, (float, int)):
        c_t = [c_t for _ in range(n)]
    
    p, R = GTMRP_props(n, l, rad(alpha), rad(beta), com, alpha0)
    GF, GT = GTMRP_matrix(R, p, c_f, c_t, sign)
    
    return GF, GT