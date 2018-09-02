from setuptools import setup

setup(name='pkgtest',
      version='1.0.0',
      description='A print test for PyPI',
      author='winycg',
      author_email='win@163.com',
      url='https://www.python.org/',
      license='MIT',
      keywords='ga nn',
      packages=['pkgtest'],
      install_requires=['numpy>=1.14', 'tensorflow>=1.7'],
      python_requires='>=3'
     )