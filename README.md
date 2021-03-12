# genre_project
Install pytho3 and pip

# To create virtual environment 
python3 -m venv ~/venv/audio

# To install dependencies 
pip install -r requirements.txt 

# To download datasets  
the different datasets are the music dataset which is - "GTZAN" and the books dataset which is - "LibriSpeech". GTZAN can be downloaded from: http://marsyas.info/downloads/datasets.html LibriSpeech can be downloaded from: https://www.openslr.org/12 from this link download - train-clean-100, test-clean, and dev-clean. 


# To run the project 
python main.py --dataset "GTZAN" --model "VGG11"

the different models in this project are "LeNet", "VGG11", "VGG19". These models represent the different sizes of models.  
