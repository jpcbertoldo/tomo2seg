from collections import defaultdict
from enum import Enum
from functools import partial
from itertools import product
from typing import *
import numpy as np
from numpy import ndarray

def _compose(f: Callable, g: Callable) -> Callable:
    """(f, g) -> fog"""
    def composed_function(x):
        return f(g(x))
    return composed_function


class GT3D(Enum):
    """"""
    identity = 0

    rotx90 = 10
    rotx180 = 11
    flipx = 12
    transpose_yz = 13

    roty90 = 20
    roty180 = 21
    flipy = 22
    transpose_xz = 23

    rotz90 = 30
    rotz180 = 31
    flipz = 32
    transpose_xy = 33

    transpose_yzx = 100
    transpose_zxy = 101

    # flipx_transposexz = 110
    # flipx_transposexy = 111
    #
    # flipy_transposexy = 210
    # flipy_transposeyz = 211
    #
    # flipz_transposexz = 310
    # flipz_transposeyz = 311
    #
    # rotation90__flip_vertical = 7


# a dictionary of transformations so that they can be referenced with a string key
_GT3D_FUNCTIONS: Dict[GT3D, Callable[[ndarray], ndarray]] = {
    GT3D.identity: lambda x: x,  # identity

    GT3D.rotx90: partial(np.rot90, axes=(1, 2), k=1),
    GT3D.flipx: partial(np.flip, axis=0),
    GT3D.transpose_yz: partial(np.transpose, axes=(0, 2, 1)),
    GT3D.rotx180: partial(np.rot90, axes=(1, 2), k=2),

    GT3D.roty90: partial(np.rot90, axes=(2, 0), k=1),
    GT3D.flipy: partial(np.flip, axis=1),
    GT3D.transpose_xz: partial(np.transpose, axes=(2, 1, 0)),
    GT3D.roty180: partial(np.rot90, axes=(1, 2), k=2),

    GT3D.rotz90: partial(np.rot90, axes=(0, 1), k=1),
    GT3D.flipz: partial(np.flip, axis=2),
    GT3D.transpose_xy: partial(np.transpose, axes=(1, 0, 2)),
    GT3D.rotz180: partial(np.rot90, axes=(1, 2), k=2),

    GT3D.transpose_yzx: partial(np.transpose, axes=(1, 2, 0)),
    GT3D.transpose_zxy: partial(np.transpose, axes=(2, 0, 1)),
}


# composed transformations - any other combination will result in something equivalent to these here
# todo verify compositions
_GT3D_FUNCTIONS.update({
    # GT3D.flip_horizontal__transpose: _compose(
    #     f=_GT2D_FUNCTIONS[GT2D.flip_horizontal],
    #     g=_GT2D_FUNCTIONS[GT2D.transpose]
    # ),
    # GT3D.flip_vertical__flip_horizontal: _compose(
    #     f=_GT2D_FUNCTIONS[GT2D.flip_vertical],
    #     g=_GT2D_FUNCTIONS[GT2D.flip_horizontal]
    # ),
    # GT3D.rotation90__flip_vertical: _compose(
    #     f=_GT2D_FUNCTIONS[GT2D.rotation90],
    #     g=_GT2D_FUNCTIONS[GT2D.flip_vertical]
    # ),
})

_GT3D_VAL2FUNC = {
    gt.value: func for gt, func in _GT3D_FUNCTIONS.items()
}


vol = np.random.rand(3, 3, 3)
all_simple = list(GT3D)
groups = defaultdict(list)


class Equivalence(Exception):
    pass

# for t1, t2 in list(product(all_simple, all_simple)):
#     vol_t1t2 = _compose(
#         _GT3D_FUNCTIONS[t1],
#         _GT3D_FUNCTIONS[t2],
#     )(vol.copy())
#
#     try:
#         for gkey in groups.keys():
#             group = groups[gkey]
#             gtrans = group[0]
#             vol_trans = _compose(
#                 _GT3D_FUNCTIONS[gtrans[0]],
#                 _GT3D_FUNCTIONS[gtrans[1]],
#             )(vol.copy())
#             if np.array_equal(vol_trans, vol_t1t2):
#                 groups[gkey].append((t1, t2))
#                 raise Equivalence()
#
#         groups[(t1, t2)].append((t1, t2))
#
#     except Equivalence:
#         continue


for t1, t2, t3 in list(product(all_simple, all_simple, all_simple)):
    vol_t1t2t3 = _compose(
        _GT3D_FUNCTIONS[t1],
        _compose(
            _GT3D_FUNCTIONS[t2],
            _GT3D_FUNCTIONS[t3],
        ),
    )(vol.copy())

    try:
        for gkey in groups.keys():
            gtrans = groups[gkey][0]
            vol_trans = _compose(
                _GT3D_FUNCTIONS[gtrans[0]],
                _compose(
                    _GT3D_FUNCTIONS[gtrans[1]],
                    _GT3D_FUNCTIONS[gtrans[2]],
                ),
            )(vol.copy())
            if np.array_equal(vol_trans, vol_t1t2t3):
                groups[gkey].append((t1, t2, t3))
                raise Equivalence()

        groups[(t1, t2, t3)].append((t1, t2, t3))

    except Equivalence:
        continue

#
# for t1, t2, t3, t4 in list(product(all_simple, all_simple, all_simple, all_simple)):
#     vol_t1t2t3t4 = _compose(
#         _GT3D_FUNCTIONS[t1],
#         _compose(
#             _GT3D_FUNCTIONS[t2],
#             _compose(
#                 _GT3D_FUNCTIONS[t3],
#                 _GT3D_FUNCTIONS[t4],
#             ),
#         ),
#     )(vol.copy())
#
#     try:
#         for gkey in groups.keys():
#             gtrans = groups[gkey][0]
#             vol_trans = _compose(
#                 _GT3D_FUNCTIONS[gtrans[0]],
#                 _compose(
#                     _GT3D_FUNCTIONS[gtrans[1]],
#                     _compose(
#                         _GT3D_FUNCTIONS[gtrans[2]],
#                         _GT3D_FUNCTIONS[gtrans[3]],
#                     ),
#                 ),
#             )(vol.copy())
#             if np.array_equal(vol_trans, vol_t1t2t3t4):
#                 groups[gkey].append((t1, t2, t3, t4))
#                 raise Equivalence()
#
#         groups[(t1, t2, t3, t4)].append((t1, t2, t3, t4))
#
#     except Equivalence:
#         continue
#

# nb. transformations combined / nb. of groups
# 2 / 43
# 3 / 48
# 4 / 48

# len(groups)
# list(groups.keys())


    """"""
    identity = 0

    rotx90 = 10
    rotx180 = 11
    flipx = 12
    transpose_yz = 13

    roty90 = 20
    roty180 = 21
    flipy = 22
    transpose_xz = 23

    rotz90 = 30
    rotz180 = 31
    flipz = 32
    transpose_xy = 33

    transpose_yzx = 100
    transpose_zxy = 101

identity = []
simples = []
doubles = []
noidentity = []

for gkey, group in groups.items():
    max_nidentity = 0
    for t1, t2, t3 in group:
        nidentity = 0
        if t1 == GT3D.identity:
            nidentity += 1
        if t2 == GT3D.identity:
            nidentity += 1
        if t3 == GT3D.identity:
            nidentity += 1
        max_nidentity = max(max_nidentity, nidentity)
    if max_nidentity == 0:
        noidentity.append(gkey)
    if max_nidentity == 1:
        simples.append(gkey)
    if max_nidentity == 2:
        doubles.append(gkey)
    if max_nidentity == 3:
        identity.append(gkey)

# identity (1)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.identity: 0>)

# doubles (12)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.rotx90: 10>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.rotx180: 11>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.flipx: 12>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.transpose_yz: 13>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.roty90: 20>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.flipy: 22>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.transpose_xz: 23>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.rotz90: 30>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.flipz: 32>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.transpose_xy: 33>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.transpose_yzx: 100>)
    # (<GT3D.identity: 0>, <GT3D.identity: 0>, <GT3D.transpose_zxy: 101>)

# simples (30)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.rotx180: 11>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.flipx: 12>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.flipy: 22>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.transpose_xz: 23>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.rotz90: 30>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.transpose_xy: 33>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.transpose_yzx: 100>)
    # (<GT3D.identity: 0>, <GT3D.rotx90: 10>, <GT3D.transpose_zxy: 101>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.flipx: 12>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.transpose_xz: 23>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.rotz90: 30>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.transpose_xy: 33>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.transpose_yzx: 100>)
    # (<GT3D.identity: 0>, <GT3D.rotx180: 11>, <GT3D.transpose_zxy: 101>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.transpose_yz: 13>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.roty90: 20>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.flipy: 22>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.transpose_xz: 23>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.flipz: 32>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.transpose_yzx: 100>)
    # (<GT3D.identity: 0>, <GT3D.flipx: 12>, <GT3D.transpose_zxy: 101>)
    # (<GT3D.identity: 0>, <GT3D.roty90: 20>, <GT3D.rotx180: 11>)
    # (<GT3D.identity: 0>, <GT3D.roty90: 20>, <GT3D.transpose_yz: 13>)
    # (<GT3D.identity: 0>, <GT3D.roty90: 20>, <GT3D.transpose_xy: 33>)
    # (<GT3D.identity: 0>, <GT3D.roty90: 20>, <GT3D.transpose_zxy: 101>)
    # (<GT3D.identity: 0>, <GT3D.flipy: 22>, <GT3D.rotz90: 30>)
    # (<GT3D.identity: 0>, <GT3D.transpose_xz: 23>, <GT3D.rotx180: 11>)
    # (<GT3D.identity: 0>, <GT3D.rotz90: 30>, <GT3D.roty90: 20>)
    # (<GT3D.identity: 0>, <GT3D.rotz90: 30>, <GT3D.flipz: 32>)
    # (<GT3D.identity: 0>, <GT3D.transpose_zxy: 101>, <GT3D.rotx180: 11>)

# noidentity (5)
    # (<GT3D.rotx90: 10>, <GT3D.rotx180: 11>, <GT3D.flipx: 12>)
    # (<GT3D.rotx90: 10>, <GT3D.flipx: 12>, <GT3D.flipy: 22>)
    # (<GT3D.rotx90: 10>, <GT3D.flipx: 12>, <GT3D.transpose_xz: 23>)
    # (<GT3D.rotx90: 10>, <GT3D.flipy: 22>, <GT3D.rotz90: 30>)
    # (<GT3D.rotx90: 10>, <GT3D.transpose_xz: 23>, <GT3D.rotx180: 11>)


# ===========================================  normalized names  ===========================================

# simples (30)
    # rotx90_rotx180
    # rotx90_flipx
    # rotx90_flipy
    # rotx90_transpose_xz
    # rotx90_rotz90
    # rotx90_transpose_xy
    # rotx90_transpose_yzx
    # rotx90_transpose_zxy
    # rotx180_flipx
    # rotx180_transpose_xz
    # rotx180_rotz90
    # rotx180_transpose_xy
    # rotx180_transpose_yzx
    # rotx180_transpose_zxy
    # flipx_transpose_yz
    # flipx_roty90
    # flipx_flipy
    # flipx_transpose_xz
    # flipx_flipz
    # flipx_transpose_yzx
    # flipx_transpose_zxy
    # roty90_rotx180
    # roty90_transpose_yz
    # roty90_transpose_xy
    # roty90_transpose_zxy
    # flipy_rotz90
    # transpose_xz_rotx180
    # rotz90_roty90
    # rotz90_flipz
    # transpose_zxy:_rotx180

# noidentity (5)
    # rotx90_rotx180_flipx
    # rotx90_flipx_flipy
    # rotx90_flipx_transpose_xz
    # rotx90_flipy_rotz90
    # rotx90_transpose_xz_rotx180

