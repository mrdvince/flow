# flow

## Getting started

## Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.10.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name flow python=3.10.6
	source activate flow
	```
	- __Windows__: 
	```bash
	conda create --name flow python=3.10.6 
	activate flow
	```

2. Clone the repository (if you haven't already!), and navigate to the `flow/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/mrdvince/flow.git
cd flow
pip install -r requirements.txt
```
3. Run the thing/generate you own video
```
python flow.py --video <video source> --to <time in seconds>
```
Do not running this on cuda will be much faster and using the M1 gpu doesn't work currently due to missing ops.
Also keeping the to 