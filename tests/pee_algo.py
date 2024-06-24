#!/usr/bin/env python2
# -*- coding: utf-8 -*-
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


import sys
sys.path.append('../')
import pytest
import numpy as np
from base.constants import SPE_MODES
from base import spe


class TestPeeModule(object):
    def setup_method(self):
        from base.spe_module import spe_class
        data_dir = "../example_data/data/*.tsv"
        eval_dir = "data"
        eval_name = "spe_module_test"
        spe_mode = "four_px_area"
        npz_dic = np.load("../example_data/dark.npz")
        MD_median_frame = npz_dic['a']
        MD_std_frame = npz_dic['c']
        self.evaluation = spe_class(data_dir, eval_dir, eval_name, spe_mode, '', md_frame=MD_median_frame, std_frame=MD_std_frame)


    def test_spe_no_torch(self):
        eval_name = self.evaluation.eval_name
        for spe_mode in SPE_MODES:
            if not 'unet' in spe_mode:
                new_eval_name = eval_name + '_' + spe_mode
                self.evaluation.change_settings(spe_mode=spe_mode, eval_name=new_eval_name)
                self.evaluation.make_spe(10)

class TestPeeAlgorithm(object):
    def setup_method(self):
        self.image = np.random.normal(0, 0.1, (7,7))
        self.image[2,3] = 3
        self.image[3,2] = 3
        self.image[3,3] = 4

    def test_epic(self):
        out = spe.epic(self.image, 3, 6, 0.1)
        print(out)

if __name__ == '__main__':
   tpm = TestPeeModule()
   tpm.setup_method()
   tpm.make_spe_no_torch()

   tpa = TestPeeAlgorithm()
   tpa.setup_method()
   tpa.test_epic()



