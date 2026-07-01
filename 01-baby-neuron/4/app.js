import { RandomSource } from "../shared/js/random.js";
import { dot, sigmoid } from "../shared/js/math.js";
import { $, bindEpochSlider, epochsPerFrame, nextFrame } from "../shared/js/ui.js";

const trainSizes = [20, 200, 2000];
const testSize = 1200;
const featureCount = 90;
const random = new RandomSource(47);

const els = {
  epochs: $("#epochs"),
  epochLabel: $("#epochLabel"),
  reset: $("#reset"),
  train: $("#train"),
  status: $("#status"),
  worstGap: $("#worstGap"),
  resultBody: $("#resultBody"),
  gapCanvas: $("#gapCanvas"),
  setupText: $("#setupText")
};

let experiments = [];
let testSet = [];
let featureMap = [];
let trainingToken = 0;

function targetLogit(x, y) {
  return 1.2 * Math.sin(2.4 * x) + 1.1 * Math.cos(2.1 * y) + 0.85 * Math.sin(1.7 * (x + y)) - 0.15;
}

function samplePoint() {
  const x = random.next() * 4 - 2;
  const y = random.next() * 4 - 2;
  const probability = sigmoid(targetLogit(x, y));
  const noisyProbability = 0.08 + 0.84 * probability;
  return {
    x,
    y,
    label: random.next() < noisyProbability ? 1 : 0
  };
}

function makeDataset(size) {
  return Array.from({ length: size }, samplePoint);
}

function makeFeatureMap() {
  return Array.from({ length: featureCount }, () => ({
    wx: random.normal() * 2.2,
    wy: random.normal() * 2.2,
    phase: random.next() * Math.PI * 2
  }));
}

function features(point) {
  const values = [1];
  const scale = Math.sqrt(2 / featureCount);

  for (const feature of featureMap) {
    values.push(scale * Math.cos(feature.wx * point.x + feature.wy * point.y + feature.phase));
  }

  return values;
}

function makeModel() {
  return {
    weights: Array(featureCount + 1).fill(0)
  };
}

function predict(model, point) {
  const phi = features(point);
  return sigmoid(dot(model.weights, phi));
}

function loss(model, dataset) {
  let total = 0;
  for (const point of dataset) {
    const prediction = Math.min(Math.max(predict(model, point), 1e-6), 1 - 1e-6);
    total += point.label === 1 ? -Math.log(prediction) : -Math.log(1 - prediction);
  }
  return total / dataset.length;
}

function trainEpoch(experiment) {
  const rate = experiment.train.length <= 20 ? 1.6 : experiment.train.length <= 200 ? 0.72 : 0.42;
  const grads = Array(featureCount + 1).fill(0);

  for (const point of experiment.train) {
    const phi = features(point);
    const error = predict(experiment.model, point) - point.label;
    for (let i = 0; i < phi.length; i++) grads[i] += error * phi[i];
  }

  for (let i = 0; i < experiment.model.weights.length; i++) {
    experiment.model.weights[i] -= rate * grads[i] / experiment.train.length;
  }
}

function makeExperiments() {
  random.bump();
  featureMap = makeFeatureMap();
  testSet = makeDataset(testSize);
  experiments = trainSizes.map((size) => ({
    size,
    train: makeDataset(size),
    model: makeModel(),
    trainLoss: null,
    testLoss: null,
    gap: null
  }));
}

function updateMetrics() {
  for (const experiment of experiments) {
    experiment.trainLoss = loss(experiment.model, experiment.train);
    experiment.testLoss = loss(experiment.model, testSet);
    experiment.gap = experiment.testLoss - experiment.trainLoss;
  }
}

function format(value) {
  return value === null ? "--" : value.toFixed(3);
}

function renderTable() {
  els.resultBody.replaceChildren();

  for (const experiment of experiments) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><strong>${experiment.size}</strong></td>
      <td>${format(experiment.trainLoss)}</td>
      <td>${format(experiment.testLoss)}</td>
      <td>${format(experiment.gap)}</td>
    `;
    els.resultBody.append(row);
  }
}

function drawGapChart() {
  const ctx = els.gapCanvas.getContext("2d");
  const { width, height } = els.gapCanvas;
  const pad = 58;
  const chartW = width - pad * 2;
  const chartH = height - pad * 2;
  const maxGap = Math.max(0.2, ...experiments.map((experiment) => experiment.gap ?? 0));

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfaf7";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "rgba(23, 32, 38, 0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad, pad);
  ctx.lineTo(pad, height - pad);
  ctx.lineTo(width - pad, height - pad);
  ctx.stroke();

  experiments.forEach((experiment, index) => {
    const barW = chartW / experiments.length * 0.48;
    const center = pad + chartW * ((index + 0.5) / experiments.length);
    const gap = Math.max(0, experiment.gap ?? 0);
    const barH = (gap / maxGap) * chartH;
    const x = center - barW / 2;
    const y = height - pad - barH;

    ctx.fillStyle = index === 0 ? "#e34f4f" : index === 1 ? "#246bfe" : "#1c8b5a";
    ctx.fillRect(x, y, barW, barH);

    ctx.fillStyle = "#172026";
    ctx.font = "700 16px ui-sans-serif, system-ui";
    ctx.textAlign = "center";
    ctx.fillText(String(experiment.size), center, height - 24);
    ctx.fillText(format(experiment.gap), center, Math.max(26, y - 10));
  });

  ctx.save();
  ctx.translate(20, height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#61707a";
  ctx.font = "700 13px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText("test loss - train loss", 0, 0);
  ctx.restore();

  ctx.fillStyle = "#61707a";
  ctx.font = "700 13px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText("training examples", width / 2, height - 6);
}

function refresh() {
  updateMetrics();
  renderTable();
  drawGapChart();
  const worst = Math.max(...experiments.map((experiment) => experiment.gap ?? 0));
  els.worstGap.textContent = format(worst);
}

function resetDemo() {
  trainingToken++;
  makeExperiments();
  els.train.disabled = false;
  els.status.textContent = "Fresh datasets and untrained models.";
  refresh();
}

async function trainAll() {
  const token = ++trainingToken;
  const epochs = Number(els.epochs.value);
  const frameBatch = epochsPerFrame(epochs, 110);
  let completed = 0;

  els.train.disabled = true;
  els.status.textContent = "Training three models...";

  while (completed < epochs && token === trainingToken) {
    for (let step = 0; step < frameBatch && completed < epochs; step++) {
      for (const experiment of experiments) trainEpoch(experiment);
      completed++;
    }

    refresh();
    els.status.textContent = `Training epoch ${completed} of ${epochs}...`;
    await nextFrame();
  }

  if (token === trainingToken) {
    els.status.textContent = "The gap is largest at 20 examples and shrinks as data grows.";
    els.train.disabled = false;
  }
}

function renderSetup() {
  els.setupText.textContent = [
    "Target: noisy nonlinear binary classification in 2D",
    "p(y=1|x,y) = sigmoid(sin/cos mixture), then 8% label noise",
    "",
    `Feature map: ${featureCount} random Fourier features`,
    "Model: high-capacity logistic classifier on those features",
    `Held-out test examples: ${testSize}`,
    `Train sizes: ${trainSizes.join(", ")}`,
    "",
    "Expected pattern:",
    "20 examples   -> train loss can collapse, test loss stays high",
    "200 examples  -> smaller gap",
    "2000 examples -> train/test losses get much closer"
  ].join("\n");
}

bindEpochSlider(els.epochs, els.epochLabel);

els.reset.addEventListener("click", resetDemo);
els.train.addEventListener("click", trainAll);

renderSetup();
resetDemo();
