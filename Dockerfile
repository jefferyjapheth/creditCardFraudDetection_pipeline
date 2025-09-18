FROM astrocrpublic.azurecr.io/runtime:3.0-10 

USER root
# Install build tools BEFORE ONBUILD triggers
RUN apt-get update && apt-get install -y build-essential python3-dev