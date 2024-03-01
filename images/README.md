# Frugally Test Image

This image contains the header files needed to run inference with frugally deep. It also contains tensorflow so you can convert h5 models and convert it to frugally deep json files.

## Convert models
Run container locally: 
```bash
docker run -it 
```

- can be used to convert h5 to frugally json
- can be used to run c++ code which uses frugally
- maybe can be used to generate tensorflow lite
- I could add a script for running tests and comparing outputs
- can be used to get the compiled headers to move into our toolchain repo

- TODO: document how to built
- TODO: add ci pipeline for building and deploying
- TODO add documentation how to use it without building it yourself
- TODO add example for complex test using our balldetection maybe with drawings
- TODO add a script that can run this for all h5 models at models.naoth.de
    This script should go into another repo -> maybe in the logcrawler repo (which should also be renamed)