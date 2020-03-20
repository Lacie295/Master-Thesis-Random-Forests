from setuptools import setup

setup(
    name='forester',
    packages=['forester'],
    include_package_data=True,
    install_requires=[
        'numpy', 'scikit-learn', 'plotly', 'psutil', 'requests', 'cython', 'dl8.5', 'pandas', 'gurobipy'
    ],
)
