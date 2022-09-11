# flow

<p align="center">
  <img src="images/stacked.gif" alt="animated" />
</p>

## Intro

Was just scrolling through PyTorch docs and stumbled upon optical flow, figured I should try it out.

Video used is from blender see acknowledgements below

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
3. Run the thing
Incase your run out of memory try setting `to` a smaller number, 

i.e  e.g to=30 generates up to the 30th second of the clip
```
python flow.py --video <video source> --to <time in seconds you want the video to stop at>
```
Do note, running this on cuda will be much faster and using the M1 gpu doesn't work currently due to missing ops.


## Acknowledgements

- [Coffe run blender video](https://studio.blender.org/films/coffee-run/)
- [Optical flow post on pytorch docs](https://pytorch.org/vision/stable/auto_examples/plot_optical_flow.html#sphx-glr-auto-examples-plot-optical-flow-py)