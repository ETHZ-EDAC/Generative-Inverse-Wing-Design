"""A simple 1D function defined by discrete (x, y) points."""

from __future__ import annotations

from typing import Literal, Optional, overload

from aerosandbox.numpy import np
import matplotlib.pyplot as plt


class Function:
    """A simple 1D function defined by discrete (x, y) points.
    Args:
        x: 1D array of x values (must be sorted in ascending order).
        y: array of y values (must be the same length as x)
           y can be:
             - 1D array of length n_points
             - 2D array shaped (n_points, n_dims)
             - 2D array shaped (n_dims, n_points) (will be transposed)

    Returns:
        A Function object that can evaluate at given x,
        plot, and integrate the function.
        For multi-dimensional y, operations are performed
        independently on each dimension. In (n_points, n_dims) format.
    """


    def __init__(self, x: np.ndarray, y: np.ndarray):

        assert x.ndim == 1, "x must be a 1D array."

        n = x.shape[0]

        if y.ndim == 1:
            assert y.shape[0] == n, "y must be the same length as x."
        elif y.ndim == 2:
            if y.shape[0] == n:
                # already (n_points, n_dims)
                pass
            elif y.shape[1] == n:
                # shape was (n_dims, n_points) -> transpose
                raise ValueError("y must have the same number of columns as the length of x.")
                y = y.T
            assert y.shape[0] == n,\
                  "y must have the same number of rows as the length of x."
        else:
            raise ValueError("y must be 1D or 2D array.")

        self._x: np.ndarray = x
        self._y: np.ndarray = y
        self._y_one_dim: bool = self._y.ndim == 1
        self._y_n_dims: int = 1 if self._y_one_dim else self._y.shape[1]

    def __len__(self) -> int:
        return len(self._x)

    # help the typechecker in evaluation
    @overload
    def evaluate(self, x_eval: float) -> float: ...

    @overload
    def evaluate(self, x_eval: np.ndarray) -> np.ndarray: ...

    def evaluate(self, x_eval: np.ndarray | float) -> np.ndarray | float:
        if self._y_one_dim:
            return np.interp(x_eval, self._x, self._y)
        else:
            if isinstance(x_eval, float):
                # For a single float, return 1D array of shape (n_dims,)
                out = np.zeros((self._y_n_dims,))
                for i in range(self._y_n_dims):
                    out[i] = np.interp(x_eval, self._x, self._y[:, i])
                return out
            else:
                # For array input, return 2D array of shape (n_points, n_dims)
                out = np.zeros((len(x_eval), self._y_n_dims))
                for i in range(self._y_n_dims):
                    out[:, i] = np.interp(x_eval, self._x, self._y[:, i])
                return out

    def rotate(self, angle_deg: float | np.ndarray,
               rot_center: np.ndarray = np.zeros(2)) -> "Function":
        """Rotates the function values by a given angle in degrees.
        Only applicable for 2D functions (y with 2 dimensions).
        Angle is defined positive counter-clockwise.
        Args:
            angle_deg: Angle in degrees to rotate the function values.
            rot_center: 2D point about which to rotate the function values.
        Returns:
            A new Function representing the rotated values.
        """
        assert self._y_n_dims == 2,\
              "rotate is only applicable for 2D functions (y with 2 dimensions)."

        n = len(self._x)

        if isinstance(angle_deg, float):
            angle_deg = np.ones_like(self._x) * angle_deg
        else:
            assert angle_deg.shape == (n,),\
                "angle_deg must be a scalar or an array of the same length as x."

        if rot_center.ndim == 1:
            assert rot_center.shape == (2,),\
                "rot_center must be length-2 (x,y) or shape (n_points,2)."
            rot_center = np.tile(rot_center.reshape(1, 2), (n, 1))
        else:
            assert rot_center.shape == (n, 2),\
                "rot_center must be length-2 or an array of shape (n_points,2)."

        # Vectorized rotation about center:
        # translate -> rotate -> translate back
        angle_rad = np.radians(angle_deg)
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)

        y = self._y  # shape (n,2)
        # translate
        y_rel_x = y[:, 0] - rot_center[:, 0]
        y_rel_y = y[:, 1] - rot_center[:, 1]

        # rotate: x' = c*x - s*y ; y' = s*x + c*y
        x_rot = c * y_rel_x - s * y_rel_y + rot_center[:, 0]
        y_rot = s * y_rel_x + c * y_rel_y + rot_center[:, 1]
        y_rotated = np.vstack([x_rot, y_rot]).T

        return Function(self._x, y_rotated)

    def to_dict(self) -> dict:
        """Returns the parameters of the function."""
        return {
            "x": self._x.tolist(),
            "y": self._y.tolist()
        }

    @staticmethod
    def from_dict(d: dict) -> Optional["Function"]:
        """Creates a Function from a dictionary of parameters."""
        x = np.array(d["x"])
        y = np.array(d["y"])

        if x is None or y is None:
            return None

        return Function(x, y)

    def plot(self, scale: Literal["linear", "log"] = "linear") -> None:
        if self._y_one_dim:
            if scale == "linear":
                plt.plot(self._x, self._y, label="Function")
            else:
                plt.semilogy(self._x, self._y, label="Function")
        else:
            fig, axes = plt.subplots(self._y_n_dims, 1, figsize=(6, 4*self._y_n_dims))
            for i in range(self._y_n_dims):
                if scale == "linear":
                    axes[i].plot(self._x, self._y[:, i], label=f"y_component_{i}")
                else:
                    axes[i].semilogy(self._x, self._y[:, i], label=f"y_component_{i}")
                axes[i].set_xlabel("x")
                axes[i].set_ylabel("f(x)")
                axes[i].set_title("Function Plot")
                axes[i].grid(True)
                axes[i].legend()
        plt.tight_layout()
        plt.grid(True)
        plt.show()


    def integrate(self) -> "Function":
        if self._y_one_dim:
            F = np.zeros_like(self._y)
            for i in range(1, len(self._x)):
                F[i] = F[i-1] + np.trapezoid(self._y[i-1:i+1], self._x[i-1:i+1])
        else:
            F = np.zeros((len(self._x), self._y_n_dims))
            for dim in range(self._y_n_dims):
                for i in range(1, len(self._x)):
                    F[i, dim] = F[i-1, dim] + np.trapezoid(self._y[i-1:i+1, dim], self._x[i-1:i+1])
        return Function(self._x, F)

    def integrate_between(self, x_start: float, x_end: float) -> float | np.ndarray:
        """Integrates the function between two x-values.
        Args:
            x_start: Start x-value.
            x_end: End x-value > x_start.
        Returns:
            The definite integral of the function between x_start and x_end.
            If the function is multi-dimensional, returns an array of integrals
            for each dimension.
        """

        assert x_end > x_start,\
              "x_end must be greater than x_start."

        k0 = np.searchsorted(self._x, x_start, side="right")
        k1 = np.searchsorted(self._x, x_end, side="left")
        xs = np.concatenate(([x_start], self._x[k0:k1], [x_end]))

        if self._y_one_dim:
            ys = np.interp(xs, self._x, self._y)
            return np.trapezoid(ys, xs)
        else:
            out = np.zeros((self._y_n_dims,))
            for i in range(self._y_n_dims):
                ys = np.interp(xs, self._x, self._y[:, i])
                out[i] = np.trapezoid(ys, xs)
            return out

    def area_preserving_boxcar(self, box_centers: np.ndarray) -> "Function":
        """Computes area-preserving boxcar averages of the function.
        Args:
            box_centers: 1D array of box center x-positions.
        Returns:
            A new Function representing the area-preserving boxcar average.
        """
        if self._y_one_dim:
            out = np.zeros(box_centers.shape)
            y_spacing = np.concatenate([[0], np.diff(box_centers), [0]])
            for i, yi in enumerate(box_centers):
                a, b = yi-y_spacing[i]/2, yi + y_spacing[i+1]/2

                if b <= a:
                    raise ValueError("a must be greater than b")

                out[i] = self.integrate_between(a, b)/(b-a)
        else:
            out = np.zeros((box_centers.shape[0], self._y_n_dims))
            y_spacing = np.concatenate([[0], np.diff(box_centers), [0]])
            for i, yi in enumerate(box_centers):
                a, b = yi-y_spacing[i]/2, yi + y_spacing[i+1]/2

                if a >= b:
                    raise ValueError("a must be greater than b")

                out[i, :] = self.integrate_between(a, b)/(b-a)
        return Function(box_centers, out)

class ChordFunction(Function):
    """
    A Rational Bezier Curve of second order defining chord distribution along the span.
    Where the control point is at (x_ctrl, c_root).
    Args:
        x_ctrl: Relative x-coordinate of control point along half-span (0 to 1).
        weight_ctrl: Weight of control point in rational bezier curve for chord lengths (0 to 100).
        c_root: Chord length at wing root.
        c_tip: Chord length at wing tip.
    """

    def __init__(self, chord_ctrl_pos: float, chord_weight_ctrl: float, chord_root: float, chord_tip: float):
        self._ctrl_pos: float = chord_ctrl_pos
        self._weight_ctrl: float = chord_weight_ctrl
        self._c_root: float = chord_root
        self._c_tip: float = chord_tip
        self.validate()

        x, y, p = self._build_curve()
        self._p = p

        super().__init__(x, y)

    def validate(self):
        assert 0.0 <= self._ctrl_pos <= 1.0,\
              "ctrl_pos must be between 0.0 and 1.0."
        assert self._c_root > 0.0,\
              "c_root must be positive."
        assert self._c_tip <= self._c_root,\
              "c_tip must be less than or equal to c_root."
        assert self._c_tip >= 0.05,\
              "c_tip must be greater than or equal to 0.05."

    def _build_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        t = np.linspace(0, 1, 100)
        w = np.clip(self._weight_ctrl, 1e-6, 100) # prevent division by zero or extreme weights
        # control points
        p0 = np.array([0.0, self._c_root])
        p1 = np.array([self._ctrl_pos, self._c_root])
        p2 = np.array([1.0, self._c_tip])
        # Bernstein polynomials
        b0 = (1 - t) ** 2
        b1 = 2 * (1 - t) * t
        b2 = t ** 2

        # Denominator: Sum of weighted Bernstein polynomials
        d = b0 + b1 * w + b2
        # Numerator: Weighted sum of control points in x and y
        x = (b0 * p0[0] + b1 * w * p1[0] + b2 * p2[0]) / d
        y = (b0 * p0[1] + b1 * w * p1[1] + b2 * p2[1]) / d

        p = np.vstack([p0, p1, p2])
        return x, y, p

    def to_dict(self) -> dict:
        """Returns the control points of the rational bezier curve."""
        return {
            "chord_ctrl_pos": float(self._ctrl_pos),
            "chord_weight_ctrl": float(self._weight_ctrl),
            "chord_root": float(self._c_root),
            "chord_tip": float(self._c_tip),
        }

    @staticmethod
    def from_dict(d: dict) -> "ChordFunction":
        """Creates a ChordFunction from a dictionary of parameters."""
        return ChordFunction(
            chord_ctrl_pos=d["chord_ctrl_pos"],
            chord_weight_ctrl=d["chord_weight_ctrl"],
            chord_root=d["chord_root"],
            chord_tip=d["chord_tip"]
        )

    def plot(self):
        plt.plot(self._x, self._y, label="Rational Bezier Curve")
        #add control points to plot
        plt.scatter(self._p[:, 0], self._p[:, 1], color='red', label="Control Points")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Function Plot")
        plt.grid(True)
        plt.show()

class TwistFunction(Function):
    """Function defining twist distribution along the span.
        Interpolates twist distribution based on root and tip twist angles
        based on an exponent function.
    Args:
        twist_root: Twist angle at wing root (degrees).
        twist_tip: Twist angle at wing tip (degrees).
        mean_pos: Relative spanwise position
        where twist is halfway between root and tip (0 to 1).
        for most wings 0.5 to 0.75 is plausible.
    """

    def __init__(self, twist_root: float, twist_tip: float, twist_mean_pos: float):
        self._twist_root: float = twist_root
        self._twist_tip: float = twist_tip
        self._twist_mean_pos: float = twist_mean_pos
        self.validate()

        x = np.linspace(0, 1, 100)
        # mean position is the point at which the twist is halfway between root and tip
        exponent = np.log(0.5) / np.log(self._twist_mean_pos)
        y = self._twist_root + (self._twist_tip - self._twist_root) * (x**exponent)
        super().__init__(x, y)

    def validate(self):
        assert -90.0 <= self._twist_root and self._twist_tip <= 90.0,\
              "twist_root and twist_tip must be between -90.0 and 90.0."
        assert 0.0 <= self._twist_mean_pos <= 1.0,\
              "twist_mean_pos must be between 0.0 and 1.0."

    def to_dict(self) -> dict:
        """Returns the parameters of the twist function."""
        return {
            "twist_root": float(self._twist_root),
            "twist_tip": float(self._twist_tip),
            "twist_mean_pos": float(self._twist_mean_pos)
        }

    @staticmethod
    def from_dict(d: dict) -> "TwistFunction":
        """Creates a TwistFunction from a dictionary of parameters."""
        return TwistFunction(
            twist_root=d["twist_root"],
            twist_tip=d["twist_tip"],
            twist_mean_pos=d["twist_mean_pos"]
        )

class DummyFunction():
    """A dummy function that does nothing.
    Used to mathch outputs for non compted functions.
    """

    def to_dict(self) -> dict:
        """Returns an empty dictionary."""
        return {"x": None, "y": None}


def main():
    sin = np.sin(np.linspace(0, 2 * np.pi, 300))
    pow2 = np.linspace(0, 2 * np.pi, 300) ** 2/10
    f = Function(np.linspace(0, 2 * np.pi, 300), np.vstack([sin, pow2]).T)
    print(f.integrate_between(0, np.pi))

    TwistFunction(10, -5, 0.4).plot()

if __name__ == "__main__":
    main()
