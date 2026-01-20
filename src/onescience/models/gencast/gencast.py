from typing import Any, Optional, Tuple
import chex
from onescience.models.gencast import casting
from onescience.models.gencast import denoiser
from onescience.models.gencast import dpm_solver_plus_plus_2s
from onescience.metrics.climate import losses
from onescience.models.gencast import predictor_base
from onescience.models.gencast import samplers_utils
from onescience.models.gencast import xarray_jax
import haiku as hk
import jax
import xarray

# ============================================================================
# Extract Configuration from Channels
# ============================================================================
def parse_channels(channels):
    """Parse channels to extract surface vars, atmospheric vars, and pressure levels."""
    surface_vars = []
    atmospheric_base_vars = set()
    pressure_levels = set()
    
    for ch in channels:
        parts = ch.split('_')
        # Check if last part is a number (pressure level)
        try:
            pressure = int(parts[-1])
            # It's an atmospheric variable
            base_name = '_'.join(parts[:-1])
            atmospheric_base_vars.add(base_name)
            pressure_levels.add(pressure)
        except (ValueError, IndexError):
            # It's a surface variable
            surface_vars.append(ch)
    
    return surface_vars, sorted(list(atmospheric_base_vars)), sorted(list(pressure_levels))

channels = [
    # Surface variables (precipitation moved to end)
    '2m_temperature',                    # 0
    'mean_sea_level_pressure',           # 1
    '10m_v_component_of_wind',           # 2
    '10m_u_component_of_wind',           # 3
    'sea_surface_temperature',           # 4
    # Atmospheric variables (temperature)
    'temperature_50', 'temperature_100', 'temperature_150', 'temperature_200',
    'temperature_250', 'temperature_300', 'temperature_400', 'temperature_500',
    'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925',
    'temperature_1000',
    # Atmospheric variables (geopotential)
    'geopotential_50', 'geopotential_100', 'geopotential_150', 'geopotential_200',
    'geopotential_250', 'geopotential_300', 'geopotential_400', 'geopotential_500',
    'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925',
    'geopotential_1000',
    # Atmospheric variables (u_component_of_wind)
    'u_component_of_wind_50', 'u_component_of_wind_100', 'u_component_of_wind_150',
    'u_component_of_wind_200', 'u_component_of_wind_250', 'u_component_of_wind_300',
    'u_component_of_wind_400', 'u_component_of_wind_500', 'u_component_of_wind_600',
    'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925',
    'u_component_of_wind_1000',
    # Atmospheric variables (v_component_of_wind)
    'v_component_of_wind_50', 'v_component_of_wind_100', 'v_component_of_wind_150',
    'v_component_of_wind_200', 'v_component_of_wind_250', 'v_component_of_wind_300',
    'v_component_of_wind_400', 'v_component_of_wind_500', 'v_component_of_wind_600',
    'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925',
    'v_component_of_wind_1000',
    # Atmospheric variables (vertical_velocity)
    'vertical_velocity_50', 'vertical_velocity_100', 'vertical_velocity_150',
    'vertical_velocity_200', 'vertical_velocity_250', 'vertical_velocity_300',
    'vertical_velocity_400', 'vertical_velocity_500', 'vertical_velocity_600',
    'vertical_velocity_700', 'vertical_velocity_850', 'vertical_velocity_925',
    'vertical_velocity_1000',
    # Atmospheric variables (specific_humidity)
    'specific_humidity_50', 'specific_humidity_100', 'specific_humidity_150',
    'specific_humidity_200', 'specific_humidity_250', 'specific_humidity_300',
    'specific_humidity_400', 'specific_humidity_500', 'specific_humidity_600',
    'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925',
    'specific_humidity_1000',
    # Precipitation (last channel)
    'total_precipitation_12hr',          # 84 (last)
]
# Parse channels to get configuration
TARGET_SURFACE_VARS, TARGET_ATMOSPHERIC_VARS, PRESSURE_LEVELS = parse_channels(channels)

# Surface variables without precipitation (for input)
TARGET_SURFACE_NO_PRECIP_VARS = tuple([v for v in TARGET_SURFACE_VARS if 'precipitation' not in v])

# Convert to tuples for consistency with original code
TARGET_SURFACE_VARS = tuple(TARGET_SURFACE_VARS)
TARGET_ATMOSPHERIC_VARS = tuple(TARGET_ATMOSPHERIC_VARS)
PRESSURE_LEVELS_WEATHERBENCH_13 = tuple(PRESSURE_LEVELS)

# All possible atmospheric variables (for reference)
ALL_ATMOSPHERIC_VARS = (
    "potential_vorticity",
    "specific_rain_water_content",
    "specific_snow_water_content",
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
    "vorticity",
    "divergence",
    "relative_humidity",
    "ozone_mass_mixing_ratio",
    "specific_cloud_liquid_water_content",
    "specific_cloud_ice_water_content",
    "fraction_of_cloud_cover",
)

# Forcing and static variables (not in channels, needed for model input)
GENERATED_FORCING_VARS = (
    'year_progress_sin',
    'year_progress_cos',
    'day_progress_sin',
    'day_progress_cos',
)

STATIC_VARS = (
    'geopotential_at_surface',
    'land_sea_mask',
)


# ============================================================================
# Task Configuration
# ============================================================================
@chex.dataclass(frozen=True, eq=True)
class TaskConfig:
    """Defines inputs and targets on which a model is trained and/or evaluated."""
    input_variables: tuple[str, ...]
    target_variables: tuple[str, ...]
    forcing_variables: tuple[str, ...]
    pressure_levels: tuple[int, ...]
    input_duration: str


# Build task configuration from channels
TASK = TaskConfig(
    input_variables=TARGET_SURFACE_NO_PRECIP_VARS + TARGET_ATMOSPHERIC_VARS + GENERATED_FORCING_VARS + STATIC_VARS,
    target_variables=TARGET_SURFACE_VARS + TARGET_ATMOSPHERIC_VARS,
    forcing_variables=GENERATED_FORCING_VARS,
    pressure_levels=PRESSURE_LEVELS_WEATHERBENCH_13,
    input_duration='24h',  # GenCast uses current frame and 12hr prior
)


# ============================================================================
# Sampler and Noise Configuration
# ============================================================================
@chex.dataclass(frozen=True, eq=True)
class SamplerConfig:
    """Configures the sampler used to draw samples from GenCast."""
    max_noise_level: float = 80.
    min_noise_level: float = 0.03
    num_noise_levels: int = 20
    rho: float = 7.
    stochastic_churn_rate: float = 2.5
    churn_min_noise_level: float = 0.75
    churn_max_noise_level: float = float('inf')
    noise_level_inflation_factor: float = 1.05


@chex.dataclass(frozen=True, eq=True)
class NoiseConfig:
    training_noise_level_rho: float = 7.0
    training_max_noise_level: float = 88.0
    training_min_noise_level: float = 0.02


@chex.dataclass(frozen=True, eq=True)
class CheckPoint:
    description: str
    license: str
    params: dict[str, Any]
    task_config: TaskConfig
    denoiser_architecture_config: denoiser.DenoiserArchitectureConfig
    sampler_config: SamplerConfig
    noise_config: NoiseConfig
    noise_encoder_config: denoiser.NoiseEncoderConfig


# ============================================================================
# GenCast Model
# ============================================================================
class GenCast(predictor_base.Predictor):
    """Predictor for a denoising diffusion model following the framework of [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models
    Karras, Aittala, Aila and Laine, 2022
    https://arxiv.org/abs/2206.00364
    """

    def __init__(
        self,
        task_config: TaskConfig,
        denoiser_architecture_config: denoiser.DenoiserArchitectureConfig,
        sampler_config: Optional[SamplerConfig] = None,
        noise_config: Optional[NoiseConfig] = None,
        noise_encoder_config: Optional[denoiser.NoiseEncoderConfig] = None,
    ):
        """Constructs GenCast."""
        # Count output variables (only ERA5 variables, exclude forcing and static)
        num_surface_vars = len(
            set(task_config.target_variables)
            - set(ALL_ATMOSPHERIC_VARS)
        )
        num_atmospheric_vars = len(
            set(task_config.target_variables)
            & set(ALL_ATMOSPHERIC_VARS)
        )
        num_outputs = (
            num_surface_vars
            + len(task_config.pressure_levels) * num_atmospheric_vars
        )
        
        denoiser_architecture_config.node_output_size = num_outputs
        self._denoiser = denoiser.Denoiser(
            noise_encoder_config,
            denoiser_architecture_config,
        )
        self._sampler_config = sampler_config
        self._sampler = None
        self._noise_config = noise_config

    def _c_in(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the noisy targets input to the underlying network."""
        return (noise_scale**2 + 1)**-0.5

    def _c_out(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the underlying network's raw outputs."""
        return noise_scale * (noise_scale**2 + 1)**-0.5

    def _c_skip(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        """Scaling applied to the skip connection."""
        return 1 / (noise_scale**2 + 1)

    def _loss_weighting(self, noise_scale: xarray.DataArray) -> xarray.DataArray:
        r"""The loss weighting \lambda(\sigma) from the paper."""
        return self._c_out(noise_scale) ** -2

    def _preconditioned_denoiser(
        self,
        inputs: xarray.Dataset,
        noisy_targets: xarray.Dataset,
        noise_levels: xarray.DataArray,
        forcings: Optional[xarray.Dataset] = None,
        **kwargs) -> xarray.Dataset:
        """The preconditioned denoising function D from the paper (Eqn 7)."""
        raw_predictions = self._denoiser(
            inputs=inputs,
            noisy_targets=noisy_targets * self._c_in(noise_levels),
            noise_levels=noise_levels,
            forcings=forcings,
            **kwargs)
        return (raw_predictions * self._c_out(noise_levels) +
                noisy_targets * self._c_skip(noise_levels))

    def loss_and_predictions(
        self,
        inputs: xarray.Dataset,
        targets: xarray.Dataset,
        forcings: Optional[xarray.Dataset] = None,
    ) -> Tuple[predictor_base.LossAndDiagnostics, xarray.Dataset]:
        return self.loss(inputs, targets, forcings), self(inputs, targets, forcings)

    def loss(self,
             inputs: xarray.Dataset,
             targets: xarray.Dataset,
             forcings: Optional[xarray.Dataset] = None,
             ) -> predictor_base.LossAndDiagnostics:

        if self._noise_config is None:
            raise ValueError('Noise config must be specified to train GenCast.')

        dtype = casting.infer_floating_dtype(targets)
        key = hk.next_rng_key()
        
        # Handle both Dataset and dict formats
        if isinstance(inputs, dict):
            batch_size = list(inputs.values())[0].sizes['batch']
        else:
            batch_size = inputs.sizes['batch']
            
        noise_levels = xarray_jax.DataArray(
            data=samplers_utils.rho_inverse_cdf(
                min_value=self._noise_config.training_min_noise_level,
                max_value=self._noise_config.training_max_noise_level,
                rho=self._noise_config.training_noise_level_rho,
                cdf=jax.random.uniform(key, shape=(batch_size,), dtype=dtype)),
            dims=('batch',))

        noise = (
            samplers_utils.spherical_white_noise_like(targets) * noise_levels
        )
        noisy_targets = targets + noise

        denoised_predictions = self._preconditioned_denoiser(
            inputs, noisy_targets, noise_levels, forcings)

        loss, diagnostics = losses.weighted_mse_per_level(
            denoised_predictions,
            targets,
            per_variable_weights={
                '2m_temperature': 1.0,
                '10m_u_component_of_wind': 0.1,
                '10m_v_component_of_wind': 0.1,
                'mean_sea_level_pressure': 0.1,
                'sea_surface_temperature': 0.1,
                'total_precipitation': 0.1
            },
        )
        loss *= self._loss_weighting(noise_levels)
        return loss, diagnostics

    def __call__(self,
                 inputs: xarray.Dataset,
                 targets_template: xarray.Dataset,
                 forcings: Optional[xarray.Dataset] = None,
                 **kwargs) -> xarray.Dataset:
        if self._sampler_config is None:
            raise ValueError(
                'Sampler config must be specified to run inference on GenCast.'
            )
        if self._sampler is None:
            self._sampler = dpm_solver_plus_plus_2s.Sampler(
                self._preconditioned_denoiser, **self._sampler_config
            )
        return self._sampler(inputs, targets_template, forcings, **kwargs)


# ============================================================================
# Utility Functions
# ============================================================================
def print_config_summary():
    """Print configuration summary."""
    print("=" * 80)
    print("GENCAST CONFIGURATION (from channels)")
    print("=" * 80)
    print(f"\nTotal channels: {len(channels)}")
    print(f"  Input channels (in_channels):  {len(in_channels)}")
    print(f"  Output channels (out_channels): {len(out_channels)}")
    
    print(f"\nExtracted from channels:")
    print(f"  Surface variables: {len(TARGET_SURFACE_VARS)}")
    for var in TARGET_SURFACE_VARS:
        print(f"    - {var}")
    
    print(f"\n  Atmospheric variables: {len(TARGET_ATMOSPHERIC_VARS)}")
    for var in TARGET_ATMOSPHERIC_VARS:
        print(f"    - {var}")
    
    print(f"\n  Pressure levels: {len(PRESSURE_LEVELS_WEATHERBENCH_13)}")
    print(f"    {PRESSURE_LEVELS_WEATHERBENCH_13}")
    
    print(f"\nAdditional variables (not in channels):")
    print(f"  Forcing: {GENERATED_FORCING_VARS}")
    print(f"  Static:  {STATIC_VARS}")
    
    print(f"\nTask Configuration:")
    print(f"  Input variables:  {len(TASK.input_variables)}")
    print(f"    = {len(in_channels)} ERA5 + {len(GENERATED_FORCING_VARS)} forcing + {len(STATIC_VARS)} static")
    print(f"  Target variables: {len(TASK.target_variables)}")
    print(f"  Input duration:   {TASK.input_duration}")
    print("=" * 80)


if __name__ == "__main__":
    print_config_summary()
    
    print("\n" + "=" * 80)
    print("CHANNELS DETAIL")
    print("=" * 80)
    print(f"{'Index':<6} {'In':<4} {'Out':<5} {'Channel Name'}")
    print("-" * 80)
    for i, ch in enumerate(channels):
        in_flag = "✓" if i in in_channels else ""
        out_flag = "✓" if i in out_channels else ""
        print(f"{i:<6} {in_flag:<4} {out_flag:<5} {ch}")