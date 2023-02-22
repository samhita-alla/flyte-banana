#####################
# BANANA DOCKERFILE #
#####################

# Must use cuda version 11+
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD banana/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD banana/server.py .

# Add your custom app code, init() and inference()
ADD banana/app.py .

# Add model metadata
ADD banana/model_metadata.json .

EXPOSE 8000

CMD python3 -u server.py