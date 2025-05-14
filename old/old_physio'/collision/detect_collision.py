# collisiondetector.py
import numpy as np
# from common import utils

def detect_collision(p, A, B, radius):
    """
    Check if point p collides with a segment AB given an effective collision radius.
    Returns (collision_flag, penetration_depth, collision_normal, closest_point).
    """
    closest, dist = point_to_segment(p, A, B)
    if dist < radius:
        penetration = radius - dist
        normal = (p - closest) / dist if dist > 0 else np.array([0, 1])
        return True, penetration, normal, closest
    return False, 0, None, None



def point_to_segment(p, A, B):
    """
    Compute the closest point on segment AB to point p and the distance.
    """
    AB = B - A
    if np.dot(AB, AB) == 0:
        return A, np.linalg.norm(p - A)
    t = np.dot(p - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0, 1)
    closest = A + t * AB
    dist = np.linalg.norm(p - closest)
    return closest, dist
