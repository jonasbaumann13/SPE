"""
MIT License

Copyright (c) 2024 Jonas Baumann, Steffen Staeck, Christopher Schlesiger

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED BY Jonas Baumann, Steffen Staeck, Christopher Schlesiger "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


try:
    import xraydb as db
    XRAYDB_IS_AVAILABLE = True
except:
    XRAYDB_IS_AVAILABLE = False
from enum import Enum
from typing import Union

SPE_LABELS = {'4px-Area': 'four_px_area',
             'Clustering': 'clustering',
             '4px-Area-Clustering': 'four_px_area_clustering',
             'Gaussian-Model-Fit (no jit)': 'gaussian_model_fit',
             'Quick Gaussian-Model-Fit': 'qgmf',
             'ASCA': 'gendreau',
             'EPIC': 'epic',
             'Small UNet': 'small_unet',
             'Big UNet': 'big_unet',
             }
SPE_MODES = {v:k for k,v in SPE_LABELS.items()}

class EasyEnum(Enum):
    """
    maps strings to integers and vice versa.

    Examples
    --------
    >>> elements = EasyEnum("Element", "H He Li")
    >>> print(int(elements['He']))
    2
    >>> print(str(elements(2)))
    He
    """
    def __str__(self) -> str:
        return self.name

    def __int__(self) -> int:
        return int(self.value)


Z = EasyEnum("Element",
             "H  He  " +
             "Li Be B  C  N  O  F  Ne " +
             "Na Mg Al Si P  S  Cl Ar " +
             "K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr " +
             "Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe " +
             "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn " +
             "Fr Ra Ac Th Pa U  Np Pu Am Cm Bk Cf Es")


IUPAC = [
             "K-L1", "K-L2", "K-L3", "K-M2", "K-M3", "K-M4,5", "K-N2,3", "K-N4,5",      # 1 - 8
             "L1-M2", "L1-M3", "L1-N2", "L1-N3",                                        # 9 - 12
             "L2-M1", "L2-M4", "L2-N4", "L2-O4",                                        # 13 - 16
             "L3-M1", "L3-M4", "L3-M5", "L3-N1", "L3-N4,5", "L3-O4,5"]                  # 17 - 22

SIEGB = [
             "Ka3", "Ka2", "Ka1", "Kb3", "Kb1", "Kb5", "Kb2", "Kb4",   # 1 - 8
             "Lb4", "Lb3", "Lg2", "Lg3",                               # 9 - 12
             "Ln", "Lb1", "Lg1", "Lg6",                                # 13 - 16
             "Ll", "La2", "La1", "Lb6", "Lb2,15", "Lb5"]               # 17 - 22

def line_E(z:Union[int, str], l:Union[int,str]):
    """
    return xraydb energy of specified line in keV.
    Parameters
    ----------
    :param z: element atomic number or string
    :param l: line index or string in Iupac or Sigbahn notation

    Returns
    -------
    energy of defined line in keV
    """
    if XRAYDB_IS_AVAILABLE:
        if isinstance(z, int):  z = str(Z(z))
        if isinstance(l, int):
            l = SIEGB[l]
        elif isinstance(l, str):
            if '-' in l:
                l = SIEGB[IUPAC.index(l)]
        edge = IUPAC[SIEGB.index(l)].split('-')[0]
        return db.xray_lines(z,edge)[l].energy/1000
    else:
        return 0