# README.md

> menu
> 1. trival supervised learning (regression & classification)
> 2. [TODO] seq2seq models (e.g. LLM)
> 3. [TODO] reinforcement learning with human's feedback

supervised learning的三个重点：模型架构、数据处理、训练技巧，分别对应model.py, process.py和train.py

## 模型架构
pytorch架构定义的类主要重写两个函数，分别是__init__()和forward()，分别为模型初始化的architecture和datapath

## 数据处理
主要需要重写dataset, dataloader。可能包括data argument, padding和normalization(代码没写)

## 训练技巧
基本流程：
1. 引入dataloader
2. initialize model
3. training looping
  a. training
  b. validating

可能包括动态学习率调度、指标计算保存、模型ckpt的保存
