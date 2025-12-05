To use the TCML cluster, we need to create a container. You can start with the attached container.def file, which installs:
- the requirements of the virtual environment used in the exercises (PPO, DDPG, etc.); and
- the hockey environment.


To build the container, use the following command:
singularity build --fakeroot /path/to/container.sif container.def


Once the container is created, it can be used as follows:
singularity run /path/to/container.sif python3 ./my_script.py

