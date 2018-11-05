#!/usr/bin/env bash

mkdir -p data
cd data

curl https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Training.zip -o BelgiumTSC_Training.zip
curl https://btsd.ethz.ch/shareddata/BelgiumTSC/BelgiumTSC_Testing.zip -o BelgiumTSC_Testing.zip

unzip BelgiumTSC_Training.zip
unzip BelgiumTSC_Testing.zip
