
# -*- coding: utf-8 -*-
#

import sys
import os
import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath('../deep4cast/'))

extensions = ['sphinx.ext.autodoc']
source_suffix = '.rst'
master_doc = 'index'
project = u'Deep4Cast'
copyright = u''
exclude_patterns = ['_build']
pygments_style = 'sphinx'
# html_theme = "pytorch_sphinx_theme"
html_theme = "sphinx_rtd_theme"
autoclass_content = "both"
