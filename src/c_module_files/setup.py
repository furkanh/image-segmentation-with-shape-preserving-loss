from distutils.core import setup, Extension

distance_module = Extension("c_module",
                            sources = ["matrix.c",
                                       "conversion.c",
                                       "hausdorff.c",
                                       "disjointSets.c",
                                       "cv.c",
                                       "cModule.c",
                                       "IoU.c"])

setup(name="C Package",
      version="1.0",
      description="This module combines all functions implemented in C.",
      ext_modules = [distance_module])