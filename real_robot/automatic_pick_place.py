"""CLI for the shared DynamicVLA-style demonstrator and VX300 adapter.

The old stationary_pick_place entry point remains a compatible alias.
"""
from .stationary_pick_place import main


if __name__ == '__main__':
    raise SystemExit(main())
