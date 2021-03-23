# Note create wheel via: python setup.py bdist_wheel

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='mal2_fakeshop_models',
      version='0.25',
      description='mal2 fake-shop detection model (train+verify) and dashboard builder',
      long_description=open('README.md').read(),
      long_description_content_type='text/plain',
      url='https://mal2git.x-t.at/root/mal2/tree/master/eCommerce',
      author='Olivia Dinica, Andrew Lindley, Clemens Heistracher',
      author_email='andrew.lindley@ait.ac.at',
      platforms=['linux','win64'],
      classifiers=[
                        # How mature is this project? Common values are
                        #   3 - Alpha
                        #   4 - Beta
                        #   5 - Production/Stable
                        'Development Status :: 3 - Alpha',

                        # Indicate who your project is intended for
                        'Intended Audience :: Developers',
                        #TODO pick proper topic classifier from list https://pypi.org/classifiers/
                        'Topic :: Software Development :: Scientific Tools',

                        # Pick your license as you wish (should match "license" above)
                         'License :: GPL3',

                        # Specify the Python versions you support here. In particular, ensure
                        # that you indicate whether you support Python 2, Python 3 or both.
                        'Programming Language :: Python :: 3.7',
                    ],
      license='GPL3.0',
      packages=['.','scrapy_spider','helper_classes'],
      include_package_data=True,    # include everything in source control as package data
      #packages=find_packages(include=['.','scrapy_spider','helper_classes'],exclude=['dashboard', 'data', 'docker','docs','docker-compose.yml']),
      keywords='fake-shop machine-learning research',
      python_requires='>=3.6',
      install_requires=requirements,
      zip_safe=False)