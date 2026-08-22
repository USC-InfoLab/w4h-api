# W4H API

The [Wearables for Health (W4H) Toolkit](https://infolab.usc.edu/projects/W4H/) is a suite of open-source tools for managing, analyzing, and visualizing wearable data used in health applications. This repository provides the [W4H Web API](https://github.com/USC-InfoLab/w4h-api), the web services for the [W4H Web App](https://github.com/USC-InfoLab/w4h-app).

## Setup

```bash
cp .env.example .env  #no need to edit for initial set up
npm install
npm start
```

To reset the installation and go through the initial set up, unset `DB_DATABASE` in .env.

### macOS: Postgres.app connection permissions

Postgres.app (2.7+) gates `trust` auth per client binary. If queries start failing with
`Postgres.app rejected "trust" authentication` (often right after restarting the API), the
`node` binary has been denied in Postgres.app's permission list. Fix it via the Postgres.app
menu bar icon → Preferences → permissions list → allow `node`, then restart the Postgres.app
server. (The denial/allow list lives in `~/Library/Preferences/com.postgresapp.Postgres2.plist`
under `ClientApplicationPermissions`.)

## CLI

The CLI will be used to access admin and recurring tasks. Run `bin/w4h` or see launch.json for examples.

In development it is convenient to add `w4h` to the path:

```sh
mkdir ~/bin
ln -s $PWD/bin/w4h ~/bin/w4h
```

and add an export in a your `.bashrc` or `.zshrc`:

```sh
export PATH=$HOME/bin:$PATH
```

When deploying you can 

---
