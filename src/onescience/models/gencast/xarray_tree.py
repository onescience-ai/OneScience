from typing import Any, Callable

import xarray


def map_structure(func: Callable[..., Any], *structures: Any) -> Any:
  """Maps func through given structures with xarrays. See tree.map_structure."""
  if not callable(func):
    raise TypeError(f'func must be callable, got: {func}')
  if not structures:
    raise ValueError('Must provide at least one structure')

  first = structures[0]
  if isinstance(first, xarray.Dataset):
    data = {k: func(*[s[k] for s in structures]) for k in first.keys()}
    if all(isinstance(a, (type(None), xarray.DataArray))
           for a in data.values()):
      data_arrays = [v.rename(k) for k, v in data.items() if v is not None]
      try:
        return xarray.merge(data_arrays, join='exact')
      except ValueError:  # Exact join not possible.
        pass
    return data
  if isinstance(first, dict):
    return {k: map_structure(func, *[s[k] for s in structures])
            for k in first.keys()}
  if isinstance(first, (list, tuple, set)):
    return type(first)(map_structure(func, *s) for s in zip(*structures))
  return func(*structures)
