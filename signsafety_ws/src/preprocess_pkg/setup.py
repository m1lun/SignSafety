from setuptools import setup

package_name = 'preprocess_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python'
    ],
    zip_safe=True,
    maintainer='mvirtual',
    maintainer_email='milungraciastaplay@gmail.com',
    description='ROS2 package for processing and publishing pre-processed image data',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_processor = preprocess_pkg.image_processor:main',
        ],
    },
)
