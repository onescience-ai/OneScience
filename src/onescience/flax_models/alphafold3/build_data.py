

"""Script for building intermediate data."""

import os
from importlib import resources
import pathlib
import site

import onescience.flax_models.alphafold3.constants.converters
from onescience.flax_models.alphafold3.constants.converters import ccd_pickle_gen
from onescience.flax_models.alphafold3.constants.converters import chemical_component_sets_gen


def build_data():
  """Builds intermediate data."""
  explicit_cif_path = os.environ.get('ALPHAFOLD3_CIFPP_COMPONENTS')
  if explicit_cif_path:
    cif_path = pathlib.Path(explicit_cif_path)
    if not cif_path.exists():
      raise ValueError(f'Configured components.cif does not exist: {cif_path}')
  else:
    for site_path in site.getsitepackages():
      path = pathlib.Path(site_path) / 'share/libcifpp/components.cif'
      if path.exists():
        cif_path = path
        break
    else:
      raise ValueError('Could not find components.cif')

  output_root = os.environ.get('ALPHAFOLD3_DATA_OUTPUT_DIR')
  if output_root:
    out_root = pathlib.Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
  else:
    out_root = pathlib.Path(resources.files(onescience.flax_models.alphafold3.constants.converters))
  ccd_pickle_path = out_root / 'ccd.pickle'
  chemical_component_sets_pickle_path = out_root / 'chemical_component_sets.pickle'
  ccd_pickle_gen.main(['', str(cif_path), str(ccd_pickle_path)])
  
  chemical_component_sets_gen.main(
      ['', str(chemical_component_sets_pickle_path)]
  )


def main() -> None:
    build_data()


if __name__ == "__main__":
    main()
