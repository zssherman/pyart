"""
Retrieval of VADs from a radar object.

"""

import logging

import numpy as np

from ..config import get_field_name
from ..core import HorizontalWindProfile

logger = logging.getLogger(__name__)


def vad_michelson(radar, vel_field=None, z_want=None, gatefilter=None):
    """
    Velocity azimuth display.

    Creates a VAD object containing U Wind, V Wind and height that
    can then be used to plot and produce the velocity azimuth display.

    Parameters
    ----------
    radar : Radar
        Radar object used.
    vel_field : string, optional
        Velocity field to use for VAD calculation.
    z_want : array, optional
        Heights for where to sample vads from.
        None will result in np.linspace(0, 10000, 100).
    gatefilter : GateFilter, optional
        A GateFilter indicating radar gates that should be excluded
        from the VAD calculation.

    Returns
    -------
    vad : HorizontalWindProfile
        A velocity azimuth display object containing height, speed,
        direction, u_wind, v_wind from a radar object.

    References
    ----------
    Michelson, D. B., Andersson, T., Koistinen, J., Collier, C. G.,
    Riedl, J., Szturc, J., Gjertsen, U., Nielsen, A. and Overgaard, S.
    (2000) BALTEX Radar Data Centre Products and their Methodologies.
    In SMHI Reports. Meteorology and Climatology. Swedish Meteorological
    and Hydrological Institute, Norrkoping.

    """
    speeds = []
    angles = []
    heights = []

    # Pulling z data from radar
    z_gate_data = radar.gate_z["data"]

    # Setting parameters
    if z_want is None:
        z_want = np.linspace(0, 10000, 100)

    # Parse field parameters
    if vel_field is None:
        radar.check_field_exists("velocity")
        vel_field = get_field_name("velocity")

    # Selecting what velocity data to use based on gatefilter
    if gatefilter is not None:
        velocities = np.ma.masked_where(
            gatefilter.gate_excluded, radar.fields[vel_field]["data"]
        )
    else:
        velocities = radar.fields[vel_field]["data"]

    # Process each sweep
    for i in range(radar.nsweeps):
        index_start = radar.sweep_start_ray_index["data"][i]
        index_end = radar.sweep_end_ray_index["data"][i] + 1

        used_velocities = velocities[index_start:index_end]
        azimuth = radar.azimuth["data"][index_start:index_end]
        elevation = radar.fixed_angle["data"][i]

        # Calculating speed and angle
        speed, angle = _vad_calculation(used_velocities, azimuth, elevation)

        logger.debug(
            "Sweep %d, max height: %f meters",
            i,
            z_gate_data[index_start, :].max(),
        )

        # Filling arrays with data
        speeds.append(speed)
        angles.append(angle)
        heights.append(z_gate_data[index_start, :])

    # Combining arrays and sorting by height
    speed_array = np.concatenate(speeds)
    angle_array = np.concatenate(angles)
    height_array = np.concatenate(heights)
    arg_order = height_array.argsort()
    speed_ordered = speed_array[arg_order]
    height_ordered = height_array[arg_order]
    angle_ordered = angle_array[arg_order]

    # Calculating U and V wind
    u_ordered, v_ordered = _sd_to_uv(speed_ordered, angle_ordered)
    u_mean = _interval_mean(u_ordered, height_ordered, z_want)
    v_mean = _interval_mean(v_ordered, height_ordered, z_want)
    vad = HorizontalWindProfile.from_u_and_v(z_want, u_mean, v_mean)
    return vad


def _vad_calculation(velocity_field, azimuth, elevation):
    """
    Calculates VAD for a single sweep using least-squares fitting.

    Fits the radial velocity model:

        V_r = u_m + a * sin(az) + b * cos(az)

    using np.linalg.lstsq at each range gate, then recovers the
    horizontal wind speed and direction:

        speed = sqrt(a^2 + b^2) / cos(elevation)
        angle = arctan2(a, b)

    Uses a vectorized solve when no data are missing, and falls back
    to a per-bin solve when masked/NaN values are present.

    Parameters
    ----------
    velocity_field : 2D masked array, shape (nrays, nbins)
        Radial velocity data for one sweep.
    azimuth : 1D array, shape (nrays,)
        Azimuth angles in degrees for each ray.
    elevation : float
        Elevation angle of the sweep in degrees.

    Returns
    -------
    speed : 1D array, shape (nbins,)
        Horizontal wind speed at each range gate.
    angle : 1D array, shape (nbins,)
        Wind direction (radians, mathematical convention) at each
        range gate.
    """
    nrays, nbins = velocity_field.shape
    cos_el = np.cos(np.deg2rad(elevation))

    # Convert masked array to NaN-filled regular array
    vel = np.ma.filled(np.ma.asarray(velocity_field), fill_value=np.nan)

    # Design matrix: [1, sin(az), cos(az)]
    az_rad = np.deg2rad(azimuth)
    A = np.column_stack(
        [
            np.ones(nrays),
            np.sin(az_rad),
            np.cos(az_rad),
        ]
    )

    has_missing = np.any(np.isnan(vel))

    if not has_missing:
        # Fast path: solve all bins at once
        # lstsq with a matrix RHS solves all columns simultaneously
        coeffs, _, _, _ = np.linalg.lstsq(A, vel, rcond=None)
        # coeffs shape: (3, nbins)
        a_values = coeffs[1, :]
        b_values = coeffs[2, :]
        speed = np.sqrt(a_values**2 + b_values**2) / cos_el
        angle = np.arctan2(a_values, b_values)
    else:
        # Per-bin fallback for missing data
        speed = np.full(nbins, np.nan)
        angle = np.full(nbins, np.nan)

        for j in range(nbins):
            vr = vel[:, j]
            valid = ~np.isnan(vr)

            # Need at least 3 valid rays to solve for 3 unknowns
            if np.sum(valid) < 3:
                continue

            coeffs, _, _, _ = np.linalg.lstsq(A[valid, :], vr[valid], rcond=None)
            a_val = coeffs[1]
            b_val = coeffs[2]

            speed[j] = np.sqrt(a_val**2 + b_val**2) / cos_el
            angle[j] = np.arctan2(a_val, b_val)

    return speed, angle


def _interval_mean(data, current_z, wanted_z):
    """
    Find the mean of *data* (indexed by *current_z*) inside height
    bins centred on each element of *wanted_z* with width equal to
    the spacing of *wanted_z*.

    Parameters
    ----------
    data : 1D array
        Data values sorted by height.
    current_z : 1D array
        Heights corresponding to *data* (must be sorted).
    wanted_z : 1D array
        Target height levels (assumed uniformly spaced).

    Returns
    -------
    mean_values : 1D array
        Mean of *data* in each height bin. NaN where no data exist.
    """
    delta = wanted_z[1] - wanted_z[0]
    mean_values = np.full(len(wanted_z), np.nan)

    for i, z in enumerate(wanted_z):
        lower = z - delta / 2.0
        upper = z + delta / 2.0
        mask = (current_z >= lower) & (current_z < upper)
        if np.any(mask):
            vals = data[mask]
            valid = ~np.isnan(vals)
            if np.any(valid):
                mean_values[i] = np.nanmean(vals)

    return mean_values


def _sd_to_uv(speed, direction):
    """
    Convert speed and direction (radians, mathematical convention)
    to u and v wind components.

    Parameters
    ----------
    speed : array
        Wind speed.
    direction : array
        Wind direction in radians (mathematical convention).

    Returns
    -------
    u : array
        U-component of wind.
    v : array
        V-component of wind.
    """
    return (np.sin(direction) * speed), (np.cos(direction) * speed)


def vad_browning(
    radar,
    velocity,
    z_want=None,
    valid_ray_min=16,
    gatefilter=None,
    window=2,
    weight="equal",
):
    """
    Velocity azimuth display.
    Note: This code uses only one sweep. Before using the
    velocity_azimuth_display function, use, for example:
    one_sweep_radar = radar.extract_sweeps([0])

    Parameters
    ----------
    radar : Radar
        Radar object used.
    velocity : string
        Velocity field to use for VAD calculation.

    Other Parameters
    ----------------
    z_want : array
        Array of desired heights to be sampled for the vad
        calculation.
    valid_ray_min : int
        Amount of rays required to include that level in
        the VAD calculation.
    gatefilter : GateFilter
        A GateFilter indicating radar gates that should be excluded when
        from the import vad calculation.
    window : int
        Value to use for window when determining new values in the
        _Averag1D function.
    weight : string
        A string to indicate weighting method to use. 'equal' for
        equal weighting when interpolating or 'idw' for inverse
        distribution squared weighting for interpolating.
        Default is 'equal'.

    Returns
    -------
    height : array
        Heights in meters above sea level at which horizontal winds were
        sampled.
    speed : array
        Horizontal wind speed in meters per second at each height.
    direction : array
        Horizontal wind direction in degrees at each height.
    u_wind : array
        U-wind mean in meters per second.
    v_wind : array
        V-wind mean in meters per second.

    Reference
    ----------
    K. A. Browning and R. Wexler, 1968: The Determination
    of Kinematic Properties of a Wind Field Using Doppler
    Radar. J. Appl. Meteor., 7, 105–113

    """
    velocities = radar.fields[velocity]["data"]
    if gatefilter is not None:
        velocities = np.ma.masked_where(gatefilter.gate_excluded, velocities)
    azimuths = radar.azimuth["data"][:]
    elevation = radar.fixed_angle["data"][0]

    u_wind, v_wind = _vad_calculation_b(velocities, azimuths, elevation, valid_ray_min)
    bad = np.logical_or(np.isnan(u_wind), np.isnan(v_wind))
    good_u_wind = u_wind[~bad]
    good_v_wind = v_wind[~bad]
    radar_height = radar.gate_z["data"][0]
    good_height = radar_height[~bad]
    if z_want is None:
        z_want = np.linspace(0, 1000, 100)[:50]
    try:
        print("max height", np.max(good_height), " meters")
        print("min height", np.min(good_height), " meters")
    except ValueError:
        raise ValueError(
            "Not enough data in this radar sweep " "for a vad calculation."
        )

    u_interp = _Average1D(
        good_height, good_u_wind, z_want[1] - z_want[0] / window, weight
    )
    v_interp = _Average1D(
        good_height, good_v_wind, z_want[1] - z_want[0] / window, weight
    )

    u_wanted = u_interp(z_want)
    v_wanted = v_interp(z_want)
    u_wanted = np.ma.masked_equal(u_wanted, 99999.0)
    v_wanted = np.ma.masked_equal(v_wanted, 99999.0)

    vad = HorizontalWindProfile.from_u_and_v(z_want, u_wanted, v_wanted)
    return vad


def _vad_calculation_b(velocities, azimuths, elevation, valid_ray_min):
    """Calculates VAD for a scan and returns u_mean and
    v_mean. velocities is a 2D array, azimuths is a 1D
    array, elevation is a number.
    Note:
    We need to solve: Ax = b
    where:
    A = [sum_sin_squared_az, sum_sin_cos_az    ] = [a, b]
        [sum_sin_cos_az,     sum_cos_squared_az]   [c, d]
    b = [sum_sin_vel_dev] = [b_1]
        [sum_cos_vel_dev]   [b_2]
    The solution to this is:
    x = A-1 * b
    A-1 is:
     1    [ d,  -b ]
    --- * [ -c,  a ]
    |A|
    and the determinate, det is: det = a*d - b*c
    Therefore the elements of x are:
    x_1 = (d* b_1  + -b * b_2) / det = (d*b_1 - b*b_2) / det
    x_2 = (-c * b_1 +  a * b_2) / det = (a*b_2 - c*b_1) / det
    """
    velocities = velocities.filled(np.nan)
    shape = velocities.shape
    _, nbins = velocities.shape

    invalid = np.isnan(velocities)
    valid_rays_per_gate = np.sum(~np.isnan(velocities), axis=0)
    too_few_valid_rays = valid_rays_per_gate < valid_ray_min
    invalid[:, too_few_valid_rays] = True

    sin_az = np.sin(np.deg2rad(azimuths))
    cos_az = np.cos(np.deg2rad(azimuths))
    sin_az = np.repeat(sin_az, nbins).reshape(shape)
    cos_az = np.repeat(cos_az, nbins).reshape(shape)
    sin_az[invalid] = np.nan
    cos_az[invalid] = np.nan

    mean_velocity_per_gate = np.nanmean(velocities, axis=0).reshape(1, -1)
    velocity_deviation = velocities - mean_velocity_per_gate

    sum_cos_vel_dev = np.nansum(cos_az * velocity_deviation, axis=0)
    sum_sin_vel_dev = np.nansum(sin_az * velocity_deviation, axis=0)

    sum_sin_cos_az = np.nansum(sin_az * cos_az, axis=0)
    sum_sin_squared_az = np.nansum(sin_az**2, axis=0)
    sum_cos_squared_az = np.nansum(cos_az**2, axis=0)

    # The A matrix
    a = sum_sin_squared_az
    b = sum_sin_cos_az
    c = sum_sin_cos_az
    d = sum_cos_squared_az

    # The b vector
    b_1 = sum_sin_vel_dev
    b_2 = sum_cos_vel_dev

    # solve for the x vector
    determinant = a * d - b * c
    x_1 = (d * b_1 - b * b_2) / determinant
    x_2 = (a * b_2 - c * b_1) / determinant

    # calculate horizontal components of winds
    elevation_scale = 1 / np.cos(np.deg2rad(elevation))
    u_mean = x_1 * elevation_scale
    v_mean = x_2 * elevation_scale
    return u_mean, v_mean


def _inverse_dist_squared(dist):
    """Obtaining distance weights by using distance weighting
    interpolation, using the inverse distance-squared relationship.
    """
    weights = 1 / (dist * dist)
    weights[np.isnan(weights)] = 99999.0
    return weights


class _Average1D:
    """Used to find the nearest gate height and horizontal wind
    value with respect to the user's desired height."""

    def __init__(self, x, y, window, weight, fill_value=99999.0):
        sort_idx = np.argsort(x)
        self.x_sorted = x[sort_idx]
        self.y_sorted = y[sort_idx]
        self.window = window
        self.fill_value = fill_value

        if weight == "equal":
            self.weight_func = lambda x: None
        elif weight == "idw":
            self.weight_func = _inverse_dist_squared
        elif callable(weight):
            self.weight_func = weight
        else:
            raise ValueError("Invalid weight argument:", weight)

    def __call__(self, x_new, window=None):
        if window is None:
            window = self.window

        y_new = np.zeros_like(x_new, dtype=self.y_sorted.dtype)
        for i, center in enumerate(x_new):
            bottom = center - window
            top = center + window
            start = np.searchsorted(self.x_sorted, bottom)
            stop = np.searchsorted(self.x_sorted, top)

            x_in_window = self.x_sorted[start:stop]
            y_in_window = self.y_sorted[start:stop]
            if len(x_in_window) == 0:
                y_new[i] = self.fill_value
            else:
                distances = x_in_window - center
                weights = self.weight_func(distances)
                y_new[i] = np.average(y_in_window, weights=weights)
        return y_new
