# W4H dashboard api

## Setup

```bash
cp .env.example .env  #edit and configure
npm install
node app.js  #run dev
npm start  #same as node app.js
```

## CLI

Run `bin/w4h` or see launch.json for examples.

It is convenient to add `spress` to the path:

```sh
mkdir ~/bin
ln -s $PWD/bin/spress ~/bin/spress
```

and add an export in a your `.bashrc` or `.zshrc`:

```sh
export PATH=$HOME/bin:$PATH
```

### Test send results with Promise.allSettled

This seems to be running slow, maybe because of image size, maybe because SES does not like too many requests at once. Seems the old foundry send was faster, we will want to check if sending was done with allSettled or another method.

```sh
spress send --to you@gmail.com public/publications/fc2d976f-2fe2-467a-ae3d-1e1bf25de042
print file: public/publications/fc2d976f-2fe2-467a-ae3d-1e1bf25de042/fc2d976f-2fe2-467a-ae3d-1e1bf25de042.json
override to address: lucnoc@gmail.com
considering all subscribers' addresses
total emails to send: 1408

spress send --to lucnoc@gmail.com public/publications/fc2d976f-2fe2-467a-ae3d-1e1bf25de042 --exec-send-all-settled
```

To count emails in gmail tag all the emails and [turn off threaded conversations](https://support.cloudhq.net/how-to-turn-off-threaded-conversations-in-gmail/). Send results:

```text
gmail count: 1408
neighborhoods: 97
First received: Sat, Jul 16, 3:14 PM
Last received: Sat, Jul 16, 6:29 PM
3:15 needed to send
```

---
