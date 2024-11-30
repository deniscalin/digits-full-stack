### This is a full stack (Flask + Vue.js) application for digit recognition.

Use the canvas to draw the image on the front end, the backend returns a label prediction from a trained ML model.

**Model specifics:** this is a 5-layer MLP model with Kaiming init and a ReLU activation. The classes are implemented from scratch using `torch.tensor` objects (not inhereting from nn.Module). The model has been trained for 300k steps, each step looking at 128 batch size (a `128 by 784 tensor`).

The model achieves a `~97.4% accuracy` score on unseen data after 40k runs on the eval set.

**Dataset:** this model has been trained on the expanded MNIST Digits [train dataset 240k rows by 785 columns](https://www.kaggle.com/datasets/deniscalin/emnist-digits?select=emnist-digits-train.csv) and evaluated on the [eval dataset 40k rows by 785 columns](https://www.kaggle.com/datasets/deniscalin/emnist-digits?select=emnist-digits-test.csv).

## Recommended IDE Setup

[VSCode](https://code.visualstudio.com/) + [Volar](https://marketplace.visualstudio.com/items?itemName=Vue.volar) (and disable Vetur).

## Customize configuration

See [Vite Configuration Reference](https://vite.dev/config/).

## Project Setup

```sh
npm install
```

### Compile and Hot-Reload for Development

```sh
npm run dev
```

### Compile and Minify for Production

```sh
npm run build
```

### Lint with [ESLint](https://eslint.org/)

```sh
npm run lint
```
