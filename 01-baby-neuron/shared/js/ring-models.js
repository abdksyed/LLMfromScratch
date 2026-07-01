import { settings } from "./config.js";
import { relu, sigmoid } from "./math.js";

// Linear classifier:
//   probability = sigmoid(w0*x + w1*y + bias)
// The 50% boundary is w0*x + w1*y + bias = 0, always a straight line.
export function makeLinearModel(random) {
  return {
    weights: [random.normal() * 0.2, random.normal() * 0.2],
    bias: 0
  };
}

export function predictLinear(model, x, y) {
  const logit = model.weights[0] * x + model.weights[1] * y + model.bias;
  return sigmoid(logit);
}

export function trainLinearEpoch(model, points) {
  const rate = 0.35;
  let dw0 = 0;
  let dw1 = 0;
  let db = 0;

  for (const point of points) {
    const error = predictLinear(model, point.x, point.y) - point.label;
    dw0 += error * point.x;
    dw1 += error * point.y;
    db += error;
  }

  model.weights[0] -= rate * dw0 / points.length;
  model.weights[1] -= rate * dw1 / points.length;
  model.bias -= rate * db / points.length;
}

// One-hidden-layer neural net:
//   hidden = ReLU(w*x + b)
//   probability = sigmoid(weighted sum of hidden values)
// Each ReLU is a hinge. Sixteen hinges can approximate a ring-shaped border.
export function makeReluModel(random) {
  const model = {
    w1: [],
    b1: [],
    w2: [],
    b2: 0
  };

  for (let h = 0; h < settings.hiddenCount; h++) {
    model.w1.push([random.normal() * 0.9, random.normal() * 0.9]);
    model.b1.push(random.normal() * 0.15);
    model.w2.push(random.normal() * 0.35);
  }

  return model;
}

function forwardRelu(model, x, y) {
  const hiddenRaw = [];
  const hidden = [];
  let logit = model.b2;

  for (let h = 0; h < settings.hiddenCount; h++) {
    const raw = model.w1[h][0] * x + model.w1[h][1] * y + model.b1[h];
    const activated = relu(raw);
    hiddenRaw.push(raw);
    hidden.push(activated);
    logit += model.w2[h] * activated;
  }

  return {
    hiddenRaw,
    hidden,
    prediction: sigmoid(logit)
  };
}

export function predictRelu(model, x, y) {
  return forwardRelu(model, x, y).prediction;
}

export function trainReluEpoch(model, points) {
  const rate = 0.045;
  const gradients = {
    w1: Array.from({ length: settings.hiddenCount }, () => [0, 0]),
    b1: Array(settings.hiddenCount).fill(0),
    w2: Array(settings.hiddenCount).fill(0),
    b2: 0
  };

  for (const point of points) {
    const state = forwardRelu(model, point.x, point.y);
    const outputError = state.prediction - point.label;

    for (let h = 0; h < settings.hiddenCount; h++) {
      gradients.w2[h] += outputError * state.hidden[h];

      // Backprop through ReLU: if raw input was below zero, that neuron was off.
      const reluSlope = state.hiddenRaw[h] > 0 ? 1 : 0;
      const hiddenError = outputError * model.w2[h] * reluSlope;

      gradients.w1[h][0] += hiddenError * point.x;
      gradients.w1[h][1] += hiddenError * point.y;
      gradients.b1[h] += hiddenError;
    }

    gradients.b2 += outputError;
  }

  for (let h = 0; h < settings.hiddenCount; h++) {
    model.w1[h][0] -= rate * gradients.w1[h][0] / points.length;
    model.w1[h][1] -= rate * gradients.w1[h][1] / points.length;
    model.b1[h] -= rate * gradients.b1[h] / points.length;
    model.w2[h] -= rate * gradients.w2[h] / points.length;
  }

  model.b2 -= rate * gradients.b2 / points.length;
}

export function accuracy(points, predict) {
  let correct = 0;

  for (const point of points) {
    const predictedLabel = predict(point.x, point.y) >= 0.5 ? 1 : 0;
    if (predictedLabel === point.label) correct++;
  }

  return correct / points.length;
}
