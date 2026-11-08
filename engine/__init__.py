from __future__ import annotations
from .vector3 import Vector3

class Matrix3:
    def __init__(self, 
                 m00: float = 1.0, m01: float = 0.0, m02: float = 0.0,
                 m10: float = 0.0, m11: float = 1.0, m12: float = 0.0,
                 m20: float = 0.0, m21: float = 0.0, m22: float = 1.0):
        self.data = [
            float(m00), float(m01), float(m02),
            float(m10), float(m11), float(m12),
            float(m20), float(m21), float(m22)
        ]

    def __repr__(self) -> str:
        d = self.data
        return f"Matrix3([{d[0]}, {d[1]}, {d[2]}], [{d[3]}, {d[4]}, {d[5]}], [{d[6]}, {d[7]}, {d[8]}])"

    def __str__(self) -> str:
        d = self.data
        return f"[{d[0]}, {d[1]}, {d[2]}]\n[{d[3]}, {d[4]}, {d[5]}]\n[{d[6]}, {d[7]}, {d[8]}]"

    def get(self, row: int, col: int) -> float:
        return self.data[row * 3 + col]

    def set(self, row: int, col: int, value: float) -> None:
        self.data[row * 3 + col] = float(value)

    def multiply_vector(self, v: Vector3) -> Vector3:
        d = self.data
        return Vector3(
            d[0] * v.x + d[1] * v.y + d[2] * v.z,
            d[3] * v.x + d[4] * v.y + d[5] * v.z,
            d[6] * v.x + d[7] * v.y + d[8] * v.z
        )

    def multiply_matrix(self, other: Matrix3) -> Matrix3:
        a = self.data
        b = other.data
        return Matrix3(
            a[0]*b[0] + a[1]*b[3] + a[2]*b[6],
            a[0]*b[1] + a[1]*b[4] + a[2]*b[7],
            a[0]*b[2] + a[1]*b[5] + a[2]*b[8],
            a[3]*b[0] + a[4]*b[3] + a[5]*b[6],
            a[3]*b[1] + a[4]*b[4] + a[5]*b[7],
            a[3]*b[2] + a[4]*b[5] + a[5]*b[8],
            a[6]*b[0] + a[7]*b[3] + a[8]*b[6],
            a[6]*b[1] + a[7]*b[4] + a[8]*b[7],
            a[6]*b[2] + a[7]*b[5] + a[8]*b[8]
        )

    def scale(self, scalar: float | int) -> Matrix3:
        s = float(scalar)
        d = self.data
        return Matrix3(
            d[0]*s, d[1]*s, d[2]*s,
            d[3]*s, d[4]*s, d[5]*s,
            d[6]*s, d[7]*s, d[8]*s
        )

    def transpose(self) -> Matrix3:
        d = self.data
        return Matrix3(
            d[0], d[3], d[6],
            d[1], d[4], d[7],
            d[2], d[5], d[8]
        )

    def determinant(self) -> float:
        d = self.data
        return (
            d[0] * (d[4] * d[8] - d[5] * d[7]) -
            d[1] * (d[3] * d[8] - d[5] * d[6]) +
            d[2] * (d[3] * d[7] - d[4] * d[6])
        )

    def inverse(self) -> Matrix3:
        det = self.determinant()
        if abs(det) < 1e-9:
            return Matrix3(0, 0, 0, 0, 0, 0, 0, 0, 0)
        inv_det = 1.0 / det
        d = self.data
        return Matrix3(
            (d[4] * d[8] - d[5] * d[7]) * inv_det,
            (d[2] * d[7] - d[1] * d[8]) * inv_det,
            (d[1] * d[5] - d[2] * d[4]) * inv_det,
            (d[5] * d[6] - d[3] * d[8]) * inv_det,
            (d[0] * d[8] - d[2] * d[6]) * inv_det,
            (d[2] * d[3] - d[0] * d[5]) * inv_det,
            (d[3] * d[7] - d[4] * d[6]) * inv_det,
            (d[1] * d[6] - d[0] * d[7]) * inv_det,
            (d[0] * d[4] - d[1] * d[3]) * inv_det
        )

    def __add__(self, other: Matrix3) -> Matrix3:
        a = self.data
        b = other.data
        return Matrix3(
            a[0]+b[0], a[1]+b[1], a[2]+b[2],
            a[3]+b[3], a[4]+b[4], a[5]+b[5],
            a[6]+b[6], a[7]+b[7], a[8]+b[8]
        )

    def __sub__(self, other: Matrix3) -> Matrix3:
        a = self.data
        b = other.data
        return Matrix3(
            a[0]-b[0], a[1]-b[1], a[2]-b[2],
            a[3]-b[3], a[4]-b[4], a[5]-b[5],
            a[6]-b[6], a[7]-b[7], a[8]-b[8]
        )

    def __mul__(self, other: Matrix3 | Vector3 | float | int) -> Matrix3 | Vector3:
        if isinstance(other, Matrix3):
            return self.multiply_matrix(other)
        elif isinstance(other, Vector3):
            return self.multiply_vector(other)
        elif isinstance(other, (int, float)):
            return self.scale(other)
        raise TypeError("Unsupported multiplication type")

    def __rmul__(self, other: float | int) -> Matrix3:
        return self.scale(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix3):
            return False
        for i in range(9):
            if abs(self.data[i] - other.data[i]) >= 1e-9:
                return False
        return True

    @staticmethod
    def identity() -> Matrix3:
        return Matrix3(1, 0, 0, 0, 1, 0, 0, 0, 1)

    @staticmethod
    def zero() -> Matrix3:
        return Matrix3(0, 0, 0, 0, 0, 0, 0, 0, 0)

    @staticmethod
    def from_outer_product(u: Vector3, v: Vector3) -> Matrix3:
        return Matrix3(
            u.x * v.x, u.x * v.y, u.x * v.z,
            u.y * v.x, u.y * v.y, u.y * v.z,
            u.z * v.x, u.z * v.y, u.z * v.z
        )
