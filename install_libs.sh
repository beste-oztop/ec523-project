# activate the virtual environment first

pip install gym torch tyro
pip install tensorboard
pip install dataclasses
pip install wandb

#special attention for box2d
#install homebrew if not already
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install swig sdl2 sdl2_image sdl2_ttf sdl2_mixer
pip install -U box2d-py


# you can deactivate the virtual environment once you are finished