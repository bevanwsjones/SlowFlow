# ----------------------------------------------------------------------------------------------------------------------
#  This file is part of the SlowFlow distribution  (https://github.com/bevanwsjones/SlowFlow).
#  Copyright (c) 2020 Bevan Walter Stewart Jones.
#
#  This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation, version 3.
#
#  This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
#  warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along with this program. If not, see
#  <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------------------------------------------------
# filename: limiter.py
# description: todo
# ----------------------------------------------------------------------------------------------------------------------

import enum as enum


# ----------------------------------------------------------------------------------------------------------------------
# Face flux limiters
# ----------------------------------------------------------------------------------------------------------------------

class Limiter(enum.Enum):
    superbee = 1
    minmod = 2
    van_albada = 3


def limiter(ratio, enumerator_limiter):
    """
    Flux limiter control function, selects the appriate limiter based on the passed enumerator
    :param ratio: gradient ratio
    :param enumerator_limiter: an enumerator from the Limiter enumerator class
    :return: limited ratio
    """
    if enumerator_limiter is Limiter.superbee:
        return superbee(ratio)
    if enumerator_limiter is Limiter.minmod:
        return minmod(ratio)
    if enumerator_limiter is Limiter.van_albada:
        return van_albada(ratio)


def superbee(ratio):
    """
    Superbee limiter of Roe 1986
    :param ratio: gradient ratio
    :return: limited ratio
    """
    return max(0.0, min(2.0*ratio, 1.0), min(ratio, 2.0))


def minmod(ratio):
    """
    minmod limiter of Roe 1986
    :param ratio: gradient ratio
    :return: limited ratio
    """
    return max(0, min(1, ratio))


def van_albada(ratio):
    """
    van Albada limiter of van Albada, et al. 1982
    :param ratio: gradient ratio
    :return: limited ratio
    """
    return ratio * (ratio + 1.0) / (ratio ** 2 + 1)
