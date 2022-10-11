from setuptools import setup, find_packages

setup(
    name='wprime_plus_b',
    version='0.0.1',
    author_email='daniel.ocampoh@udea.edu.co',
    description='WPrime + b analysis',
    url='https://github.com/deoache/b_lepton_met',
    license='MIT',
    packages=find_packages(),
    install_requires=[
          "coffea>=0.7.2",
          "correctionlib>=2.0.0rc6",
          "rhalphalib",
          "pandas",
      ],
)