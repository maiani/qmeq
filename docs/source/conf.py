# -- Paths ------------------------------------------------
import sys
import os

sys.path.insert(0, os.path.abspath('../../qmeq/'))
sys.path.insert(0, os.path.abspath('../..'))

# -- General configuration ------------------------------------------------
needs_sphinx = '1.3'

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'nbsphinx',
              'nbsphinx_link',
             ]

# The example notebooks live in the ``examples/`` directory and are pulled into
# the documentation through nbsphinx-link (see ``examples/*.nblink``).  They are
# rendered from their stored outputs rather than executed on every build, as the
# second-order (2vN / RTD) calculations are far too expensive for a doc build;
# execution is exercised separately by the test suite.
nbsphinx_execute = 'never'

# The Cython extensions are optional at runtime and are not built when the
# documentation is generated directly from a source checkout.  Mock only those
# extension modules so autodoc can still import and document the Python API.
autodoc_mock_imports = [
    'qmeq.approach.c_aprclass',
    'qmeq.approach.c_kernel_handler',
    'qmeq.approach.base.c_RTD',
    'qmeq.approach.base.c_lindblad',
    'qmeq.approach.base.c_neumann1',
    'qmeq.approach.base.c_neumann2',
    'qmeq.approach.base.c_pauli',
    'qmeq.approach.base.c_redfield',
    'qmeq.approach.elph.c_lindblad',
    'qmeq.approach.elph.c_neumann1',
    'qmeq.approach.elph.c_pauli',
    'qmeq.approach.elph.c_redfield',
    'qmeq.specfunc.c_specfunc',
    'qmeq.specfunc.c_specfunc_elph',
    'qmeq.wrappers.c_lapack',
    'qmeq.wrappers.c_mytypes',
]
suppress_warnings = ['autodoc.mocked_object']

# Many classes exist both as a pure-Python implementation (e.g.
# ``qmeq.approach.aprclass.Approach``) and a compiled Cython twin
# (``qmeq.approach.c_aprclass.Approach``).  When the extensions are built both
# are documented, so bare type names such as ``Approach`` in docstrings resolve
# ambiguously.  Map them to the canonical public classes.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    'Approach': ':class:`~qmeq.approach.aprclass.Approach`',
    'ApproachElPh': ':class:`~qmeq.approach.aprclass.ApproachElPh`',
    'Approach2vN': ':class:`~qmeq.approach.aprclass.ApproachBase2vN`',
}

templates_path = ['_templates']
#source_suffix = ['.rst', '.md']
source_suffix = '.rst'
#source_encoding = 'utf-8-sig'
master_doc = 'index'

project = u'qmeq'
copyright = u'2019, Gediminas Kirsanskas'
author = u'Gediminas Kirsanskas'

version = u'1.1'
release = u'1.1'

language = 'en'
#today = ''
#today_fmt = '%B %d, %Y'
exclude_patterns = ['**.ipynb_checkpoints']
#default_role = None
#add_function_parentheses = True
#add_module_names = True
#show_authors = False
pygments_style = 'sphinx'
#modindex_common_prefix = []
#keep_warnings = False
todo_include_todos = False

# -- Options for HTML output ----------------------------------------------
#html_theme = 'alabaster'
#html_theme = 'classic'
html_theme = 'sphinx_rtd_theme'
#html_theme = 'sphinxdoc'
#html_theme = 'epub'
#html_theme_options = {}
#html_theme_path = []
#html_title = u'qmeq v0'
#html_short_title = None
#html_logo = None
#html_favicon = None
#html_static_path = ['_static']
#html_extra_path = []
#html_last_updated_fmt = None
#html_use_smartypants = True
#html_sidebars = {}
#html_additional_pages = {}
#html_domain_indices = True
#html_use_index = True
#html_split_index = False
#html_show_sourcelink = True
#html_show_sphinx = True
#html_show_copyright = True
#html_use_opensearch = ''
#html_file_suffix = None
#html_search_language = 'en'
#html_search_options = {'type': 'default'}
#html_search_scorer = 'scorer.js'
htmlhelp_basename = 'qmeqdoc'


# -- Options for LaTeX output ---------------------------------------------
latex_elements = { }
latex_documents = [
    (master_doc, 'qmeq.tex', u'qmeq Documentation',
     u'Gediminas Kirsanskas', 'manual'),
]
#latex_logo = None
#latex_use_parts = False
#latex_show_pagerefs = False
#latex_show_urls = False
#latex_appendices = []
#latex_domain_indices = True


# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'qmeq', u'qmeq Documentation',
     [author], 1)
]
#man_show_urls = False


# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (master_doc, 'qmeq', u'qmeq Documentation',
     author, 'qmeq', 'One line description of project.',
     'Miscellaneous'),
]
#texinfo_appendices = []
#texinfo_domain_indices = True
#texinfo_show_urls = 'footnote'
#texinfo_no_detailmenu = False
