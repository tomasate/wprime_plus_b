from setuptools import setup, find_packages

setup(
    name='b_lepton_met',
    version='0.0.1',
    author_email='daniel.ocampoh@udea.edu.co',
    description='W prime analysis',
    url='https://github.com/deoache/b_lepton_met',
    project_urls = {
        "Bug Tracker": "https://github.com/mike-huls/toolbox/issues"
    },
    license='MIT',
    packages=["b_lepton_met"],#find_packages(),
    install_requires=[
          "coffea>=0.7.2",
          "correctionlib>=2.0.0rc6",
          "rhalphalib",
          "pandas",
      ],
)