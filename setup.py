from setuptools import setup, find_packages

setup(name='Blu',
	  version='0.01',
	  packages=find_packages(),
	  install_requires=["openai-whisper>=20231117 ",
						"elevenlabs>=0.2.27",
						"matplotlib>=3.8.2",
						"openai>=1.1.0",
						"flask>=2.3.3",
						"transformers>=4.37.2",
						"datasets>=2.16.1",
						"gevent>=23.9.1",
						"flask-cors>=3",
						"anthropic>=0.19.0",
						"torch>=2.2.1",
						"numpy>=1.26.4"],
	  python_requires='>=3.10')

# for dirty arch users
# sudo pacman -S python-openai python-flask python-flask-cors python-matplotlib python-numpy python-gevent python-flask-cors
# yay -S python-elevenlabs whisper-git python-transformers python-datasets
