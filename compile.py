from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
ext_modules = [
	Extension("train",  ["train.py"]),
	Extension("test",  ["test.py"]),
	Extension("utils",  ["utils.py"]),
]
setup(
    name = 'Auto-Synthesis',
    cmdclass = {'build_ext': build_ext},
	ext_modules=cythonize(ext_modules, compiler_directives={'always_allow_keywords': True})
)