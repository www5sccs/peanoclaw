__all__ = ['peanoclaw']

import os, site
stdlib_dir = os.path.dirname(os.__file__)
locallib_dir = site.USER_SITE
real_clawpack_path = os.path.join(stdlib_dir, 'clawpack')
real_local_clawpack_path = os.path.join(locallib_dir, 'clawpack')
__path__.append(real_clawpack_path)
__path__.append(real_local_clawpack_path)
try:
    execfile(os.path.join(real_clawpack_path, '__init__.py'))
except IOError:
    pass
try:
    execfile(os.path.join(real_local_clawpack_path, '__init__.py'))
except IOError:
    pass

#import clawpack.peanoclaw