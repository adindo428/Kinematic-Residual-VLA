"""Load the pinned VLASH policy without its unrelated robot-CLI side effects."""
import importlib.machinery
import importlib.util
import sys


def activate():
    if 'vlash' not in sys.modules:
        spec=importlib.machinery.PathFinder.find_spec('vlash',sys.path)
        if spec is None:raise ImportError('Pinned VLASH source path is missing')
        # All policy/layer modules load from their original files. The root
        # initializer only registers hardware/CLI plugins and imports SDKs for
        # real robots, which are not used by this offline DOM trainer.
        module=importlib.util.module_from_spec(spec)
        module.__version__='0.1.0'
        sys.modules['vlash']=module
