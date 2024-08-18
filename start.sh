#!/bin/bash


if [ ! -d "model/oneke" ]
then
  mkdir model/oneke
fi
modelscope download --model ZJUNLP/OneKE --local_dir model/oneke
