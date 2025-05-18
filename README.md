# AI-TEXT-TO-IMAGE-GENERATOR-


#OPEN TERMINAL IN VS CODE THEN TYPE THE FOLLOWING COMMANDS:-

python -m venv venv
.\venv\Scripts\activate



#IF IT THROWS ANY ERROR JUST COPY THIS COMMAND :-

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\activate



#Now for installing its dependencies:-

pip install --upgrade diffusers transformers accelerate mediapy peft
pip install mediapy




#If you'r using gpu like Nvidia 3050/3060/4050 etc.... or anyother use this :-

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121




#If using CPU Only :-

pip install torch torchvision torchaudio




#Now to run the code just type:-

python stable_diffusion.py




