import {
  addVectors,
  formatFixed,
  multiplyMatrices,
  multiplyMatrixVector,
  relu,
  sigmoid
} from "./math.js";

export function makeDeepLinearModel(random, sizes = [2, 4, 4, 4, 4, 1]) {
  return makeLayerStack(random, sizes);
}

export function makeDeepReluModel(random, sizes = [2, 8, 8, 8, 8, 1]) {
  return makeLayerStack(random, sizes);
}

function makeLayerStack(random, sizes) {
  const layers = [];

  for (let i = 0; i < sizes.length - 1; i++) {
    const inputSize = sizes[i];
    const outputSize = sizes[i + 1];
    const scale = Math.sqrt(2 / inputSize);

    layers.push({
      weights: Array.from({ length: outputSize }, () =>
        Array.from({ length: inputSize }, () => random.normal() * scale)
      ),
      bias: Array(outputSize).fill(0)
    });
  }

  return { layers };
}

function affine(layer, input) {
  return layer.weights.map((row, rowIndex) => {
    let sum = layer.bias[rowIndex];
    for (let i = 0; i < row.length; i++) sum += row[i] * input[i];
    return sum;
  });
}

function forwardDeep(model, input, useRelu) {
  const activations = [input];
  const preActivations = [];
  let current = input;

  for (let i = 0; i < model.layers.length; i++) {
    const raw = affine(model.layers[i], current);
    const isHiddenLayer = i < model.layers.length - 1;
    const next = useRelu && isHiddenLayer ? raw.map(relu) : raw;

    preActivations.push(raw);
    activations.push(next);
    current = next;
  }

  return {
    activations,
    preActivations,
    prediction: sigmoid(current[0])
  };
}

export function predictDeepLinear(model, x, y) {
  return forwardDeep(model, [x, y], false).prediction;
}

export function predictDeepRelu(model, x, y) {
  return forwardDeep(model, [x, y], true).prediction;
}

export function trainDeepLinearEpoch(model, points) {
  trainDeepEpoch(model, points, false, 0.02);
}

export function trainDeepReluEpoch(model, points) {
  trainDeepEpoch(model, points, true, 0.015);
}

function trainDeepEpoch(model, points, useRelu, rate) {
  const gradients = model.layers.map((layer) => ({
    weights: layer.weights.map((row) => row.map(() => 0)),
    bias: layer.bias.map(() => 0)
  }));

  for (const point of points) {
    const state = forwardDeep(model, [point.x, point.y], useRelu);
    let delta = [state.prediction - point.label];

    for (let layerIndex = model.layers.length - 1; layerIndex >= 0; layerIndex--) {
      const input = state.activations[layerIndex];
      const layer = model.layers[layerIndex];

      for (let out = 0; out < layer.weights.length; out++) {
        gradients[layerIndex].bias[out] += delta[out];
        for (let inputIndex = 0; inputIndex < input.length; inputIndex++) {
          gradients[layerIndex].weights[out][inputIndex] += delta[out] * input[inputIndex];
        }
      }

      if (layerIndex === 0) continue;

      const previousDelta = Array(layer.weights[0].length).fill(0);
      for (let out = 0; out < layer.weights.length; out++) {
        for (let inputIndex = 0; inputIndex < layer.weights[out].length; inputIndex++) {
          previousDelta[inputIndex] += layer.weights[out][inputIndex] * delta[out];
        }
      }

      if (useRelu) {
        const previousRaw = state.preActivations[layerIndex - 1];
        for (let i = 0; i < previousDelta.length; i++) {
          previousDelta[i] *= previousRaw[i] > 0 ? 1 : 0;
        }
      }

      delta = previousDelta;
    }
  }

  for (let layerIndex = 0; layerIndex < model.layers.length; layerIndex++) {
    const layer = model.layers[layerIndex];
    for (let out = 0; out < layer.weights.length; out++) {
      layer.bias[out] -= rate * gradients[layerIndex].bias[out] / points.length;
      for (let inputIndex = 0; inputIndex < layer.weights[out].length; inputIndex++) {
        layer.weights[out][inputIndex] -= rate * gradients[layerIndex].weights[out][inputIndex] / points.length;
      }
    }
  }
}

export function collapseDeepLinear(model) {
  let matrix = [[1, 0], [0, 1]];
  let bias = [0, 0];

  for (const layer of model.layers) {
    const nextMatrix = multiplyMatrices(layer.weights, matrix);
    const nextBias = addVectors(multiplyMatrixVector(layer.weights, bias), layer.bias);
    matrix = nextMatrix;
    bias = nextBias;
  }

  return {
    weights: matrix[0],
    bias: bias[0]
  };
}

export function formatDeepLinearMatrices(model, digits = 3) {
  const matrices = model.layers.map((layer, index) => {
    const rows = layer.weights.map((row) => `[${row.map((value) => formatFixed(value, digits)).join(", ")}]`);
    return `W${index + 1} (${layer.weights.length}x${layer.weights[0].length}) = [\n  ${rows.join("\n  ")}\n]`;
  });

  const collapsed = collapseDeepLinear(model);
  const product = `Product W5*W4*W3*W2*W1 (1x2) = [${collapsed.weights
    .map((value) => formatFixed(value, digits))
    .join(", ")}]`;
  const bias = `Collapsed bias = ${formatFixed(collapsed.bias, digits)}`;

  return `${matrices.join("\n\n")}\n\n${product}\n${bias}`;
}

export function traceDeepLinearCalculation(model, input, digits = 3) {
  let current = input;
  const lines = [
    `Input x = [${input.map((value) => formatFixed(value, digits)).join(", ")}]`,
    "",
    "Pass through the 5 linear layers:"
  ];

  for (let i = 0; i < model.layers.length; i++) {
    const previous = current;
    current = affine(model.layers[i], current);
    lines.push(
      `Layer ${i + 1}: W${i + 1} * [${previous
        .map((value) => formatFixed(value, digits))
        .join(", ")}] + b${i + 1} = [${current
        .map((value) => formatFixed(value, digits))
        .join(", ")}]`
    );
  }

  const collapsed = collapseDeepLinear(model);
  const collapsedLogit = collapsed.weights[0] * input[0] + collapsed.weights[1] * input[1] + collapsed.bias;
  const stackedLogit = current[0];

  lines.push(
    "",
    "Collapse the matrices first, then use one layer:",
    `W = W5*W4*W3*W2*W1 = [${collapsed.weights.map((value) => formatFixed(value, digits)).join(", ")}]`,
    `b = ${formatFixed(collapsed.bias, digits)}`,
    `W * x + b = ${formatFixed(collapsedLogit, digits)}`,
    "",
    `5-layer logit       = ${formatFixed(stackedLogit, digits)}`,
    `collapsed logit     = ${formatFixed(collapsedLogit, digits)}`,
    `5-layer probability = ${formatFixed(sigmoid(stackedLogit), digits)}`,
    `collapsed prob.     = ${formatFixed(sigmoid(collapsedLogit), digits)}`,
    `absolute difference = ${Math.abs(stackedLogit - collapsedLogit).toExponential(2)}`
  );

  return lines.join("\n");
}
