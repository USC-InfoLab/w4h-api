# W4H API

The [Wearables for Health (W4H) Toolkit](https://infolab.usc.edu/projects/W4H/) is a suite of open-source tools for managing, analyzing, and visualizing wearable data used in health applications. This repository provides the [W4H Web API](https://github.com/USC-InfoLab/w4h-api), the web services for the [W4H Web App](https://github.com/USC-InfoLab/w4h-app).

## Setup

```bash
cp .env.example .env  #do not edit for initial set up
npm install
npm start
```

To reset the installation 

## CLI

Run `bin/w4h` or see launch.json for examples.

It is convenient to add `w4h` to the path:

```sh
mkdir ~/bin
ln -s $PWD/bin/w4h ~/bin/w4h
```

and add an export in a your `.bashrc` or `.zshrc`:

```sh
export PATH=$HOME/bin:$PATH
```

---
