from setuptools import setup, find_packages

with open('README.md', mode='r') as file:
    description = file.read()

setup(
    name = 'imperial_materials_simulation',
    version = '1.0.0',
    description = 'MMolecular simulation tool made for the theory and simulation module taken by materials science and engineering undergraduates at Imperial College London',
    author = 'Ayham Al-Saffar',
    url = 'https://github.com/AyhamSaffar/imperial-materials-simulation',
    packages = find_packages(),
    python_requires = '~=3.10',
    install_requires = [
        'ipykernel',
        'ipympl',
        'ipywidgets',
        'matplotlib',
        'numba>=0.60.0',
        'numpy',
        'pandas',
        'py3dmol',
        'scipy',
    ],
    long_description = description,
    long_description_content_type = 'text/markdown',
    )