import { generateRings } from "../shared/js/data.js";
import { drawPlot } from "../shared/js/draw.js";
import {
  accuracy,
  makeLinearModel,
  makeReluModel,
  predictLinear,
  predictRelu,
  trainLinearEpoch,
  trainReluEpoch
} from "../shared/js/ring-models.js";
import { RandomSource } from "../shared/js/random.js";
import { $, bindEpochSlider, epochsPerFrame, formatPercent, nextFrame } from "../shared/js/ui.js";

const els = {
  epochs: $("#epochs"),
  epochLabel: $("#epochLabel"),
  rerun: $("#rerun"),
  train: $("#train"),
  linearCanvas: $("#linearCanvas"),
  reluCanvas: $("#reluCanvas"),
  linearAccuracy: $("#linearAccuracy"),
  reluAccuracy: $("#reluAccuracy"),
  status: $("#status")
};

const random = new RandomSource(7);

let currentPoints = [];
let linearModel = null;
let reluModel = null;
let trainingToken = 0;

function resetModels() {
  linearModel = makeLinearModel(random);
  reluModel = makeReluModel(random);
}

function refreshView() {
  const linearPredict = (x, y) => predictLinear(linearModel, x, y);
  const reluPredict = (x, y) => predictRelu(reluModel, x, y);

  els.linearAccuracy.textContent = formatPercent(accuracy(currentPoints, linearPredict));
  els.reluAccuracy.textContent = formatPercent(accuracy(currentPoints, reluPredict));

  drawPlot(els.linearCanvas, currentPoints, linearPredict);
  drawPlot(els.reluCanvas, currentPoints, reluPredict);
}

function regenerateData() {
  random.bump();
  currentPoints = generateRings(random);
  resetModels();
  refreshView();
  els.status.textContent = `Generated ${currentPoints.length} new points. Models are untrained.`;
}

async function trainModels() {
  const token = ++trainingToken;
  const epochs = Number(els.epochs.value);
  const frameBatch = epochsPerFrame(epochs);
  let completedEpochs = 0;

  els.status.textContent = "Training...";
  els.train.disabled = true;

  while (completedEpochs < epochs && token === trainingToken) {
    for (let step = 0; step < frameBatch && completedEpochs < epochs; step++) {
      trainLinearEpoch(linearModel, currentPoints);
      trainReluEpoch(reluModel, currentPoints);
      completedEpochs++;
    }

    refreshView();
    els.status.textContent = `Training epoch ${completedEpochs} of ${epochs}...`;

    await nextFrame();
  }

  if (token === trainingToken) {
    els.status.textContent = `Trained ${currentPoints.length} points for ${epochs} epochs.`;
    els.train.disabled = false;
  }
}

bindEpochSlider(els.epochs, els.epochLabel);

els.rerun.addEventListener("click", () => {
  trainingToken++;
  els.train.disabled = false;
  regenerateData();
});

els.train.addEventListener("click", () => {
  trainingToken++;
  resetModels();
  refreshView();
  trainModels();
});

resetModels();
regenerateData();
