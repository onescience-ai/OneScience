import datetime

import numpy as np
import pytz
import torch

RAD_PER_DEG = torch.tensor(np.pi / 180.0)
DATETIME_2000 = datetime.datetime(2000, 1, 1, 12, 0, 0, tzinfo=pytz.utc).timestamp()


def _dali_mod(a, b):
    return a - b * torch.floor(a / b)


def cos_zenith_angle(
    time: torch.tensor,
    latlon: torch.tensor,
):
    """
    Dali datapipe for computing Cosine of sun-zenith angle for lon, lat at time (UTC).

    Parameters
    ----------
    time : dali.types.DALIDataType
        Time in seconds since 2000-01-01 12:00:00 UTC. Shape `(seq_length,)`.
    latlon : dali.types.DALIDataType
        Latitude and longitude in degrees. Shape `(2, nr_lat, nr_lon)`.

    Returns
    -------
    dali.types.DALIDataType
        Cosine of sun-zenith angle. Shape `(seq_length, 1, nr_lat, nr_lon)`.
    """
    # print(type(latlon))
    lat = latlon[0:1, :, :].unsqueeze(0) * RAD_PER_DEG  # 形状：(1, 1, nr_lat, nr_lon)
    lon = latlon[1:2, :, :].unsqueeze(0) * RAD_PER_DEG  # 形状：(1, 1, nr_lat, nr_lon)
    time = time.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    return _star_cos_zenith(time, lat, lon)


def _days_from_2000(model_time):  # pragma: no cover
    """Get the days since year 2000."""
    return (model_time - DATETIME_2000) / (24.0 * 3600.0)


def _greenwich_mean_sidereal_time(model_time):
    """
    Greenwich mean sidereal time, in radians.
    Reference:
        The AIAA 2006 implementation:
            http://www.celestrak.com/publications/AIAA/2006-6753/
    """
    jul_centuries = _days_from_2000(model_time) / 36525.0
    theta = 67310.54841 + jul_centuries * (
        876600 * 3600
        + 8640184.812866
        + jul_centuries * (0.093104 - jul_centuries * 6.2 * 10e-6)
    )

    theta_radians = _dali_mod((theta / 240.0) * RAD_PER_DEG, 2 * np.pi)
    return theta_radians


def _local_mean_sidereal_time(model_time, longitude):
    """
    Local mean sidereal time. requires longitude in radians.
    Ref:
        http://www.setileague.org/askdr/lmst.htm
    """
    return _greenwich_mean_sidereal_time(model_time) + longitude


def _sun_ecliptic_longitude(model_time):
    """
    Ecliptic longitude of the sun.
    Reference:
        http://www.geoastro.de/elevaz/basics/meeus.htm
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0

    # mean anomaly calculation
    mean_anomaly = (
        357.52910
        + 35999.05030 * julian_centuries
        - 0.0001559 * julian_centuries * julian_centuries
        - 0.00000048 * julian_centuries * julian_centuries * julian_centuries
    ) * RAD_PER_DEG

    # mean longitude
    mean_longitude = (
        280.46645 + 36000.76983 * julian_centuries + 0.0003032 * (julian_centuries**2)
    ) * RAD_PER_DEG

    d_l = (
        (1.914600 - 0.004817 * julian_centuries - 0.000014 * (julian_centuries**2))
        * torch.sin(mean_anomaly)
        + (0.019993 - 0.000101 * julian_centuries) * torch.sin(2 * mean_anomaly)
        + 0.000290 * torch.sin(3 * mean_anomaly)
    ) * RAD_PER_DEG

    # true longitude
    return mean_longitude + d_l


def _obliquity_star(julian_centuries):
    """
    return obliquity of the sun
    Use 5th order equation from
    https://en.wikipedia.org/wiki/Ecliptic#Obliquity_of_the_ecliptic
    """
    return (
        23.0
        + 26.0 / 60
        + 21.406 / 3600.0
        - (
            46.836769 * julian_centuries
            - 0.0001831 * (julian_centuries**2)
            + 0.00200340 * (julian_centuries**3)
            - 0.576e-6 * (julian_centuries**4)
            - 4.34e-8 * (julian_centuries**5)
        )
        / 3600.0
    ) * RAD_PER_DEG


def _right_ascension_declination(model_time):
    """
    Right ascension and declination of the sun.
    """
    julian_centuries = _days_from_2000(model_time) / 36525.0
    eps = _obliquity_star(julian_centuries)

    eclon = _sun_ecliptic_longitude(model_time)
    x = torch.cos(eclon)
    y = torch.cos(eps) * torch.sin(eclon)
    z = torch.sin(eps) * torch.sin(eclon)
    r = torch.sqrt(1.0 - z * z)
    # sun declination
    declination = torch.atan2(z, r)
    # right ascension
    right_ascension = 2 * torch.atan2(y, (x + r))
    return right_ascension, declination


def _local_hour_angle(model_time, longitude, right_ascension):
    """
    Hour angle at model_time for the given longitude and right_ascension
    longitude in radians
    Ref:
        https://en.wikipedia.org/wiki/Hour_angle#Relation_with_the_right_ascension
    """
    return _local_mean_sidereal_time(model_time, longitude) - right_ascension


def _star_cos_zenith(model_time, lat, lon):
    """
    Return cosine of star zenith angle
    lon,lat in radians
    Ref:
        Azimuth:
            https://en.wikipedia.org/wiki/Solar_azimuth_angle#Formulas
        Zenith:
            https://en.wikipedia.org/wiki/Solar_zenith_angle
    """

    ra, dec = _right_ascension_declination(model_time)
    h_angle = _local_hour_angle(model_time, lon, ra)

    cosine_zenith = torch.sin(lat) * torch.sin(dec) + torch.cos(lat) * torch.cos(
        dec
    ) * torch.cos(h_angle)
    return cosine_zenith
